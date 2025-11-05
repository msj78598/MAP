import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.spatial import cKDTree

st.set_page_config(page_title="Meterless Transformers (Fast)", layout="wide")

st.title("تحليل المحوّلات التي لا تقع ضمن نطاق أي عدّاد — نسخة سريعة")

st.markdown(
"""
**الفكرة:** نراجع مواقع المحوّلات والعدّادات ونحدّد المحوّلات التي لا يوجد ضمن نطاقها أي عدّاد (حسب مسافة تختارها).  
**المدخلات:** ملف العدّادات + ملف المحوّلات (CSV أو Excel).  
**المخرجات:** **ملف Excel** بالمحوّلات التي **لا** يغطّيها أي عدّاد + ملخص إحصائي سريع.
"""
)

# =========== إعدادات ===========
dist_m = st.number_input("المسافة العتبية (متر)", min_value=1, value=100, step=10)

with st.expander("خيارات متقدمة", expanded=False):
    st.caption("عدد أقرب العدّادات التي نفحصها (بحث تقريبي قبل المسافة الدقيقة). اتركه كما هو غالبًا.")
    max_k = st.slider("عدد أقرب العدّادات للفحص (k)", 1, 50, 10)

# =========== قراءة الملفات ===========
def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        # قراءة أسرع، خصوصاً للملفات الكبيرة
        return pd.read_csv(file, engine="python")
    return pd.read_excel(file)

st.subheader("رفع الملفات")
c1, c2 = st.columns(2)
with c1:
    meters_file = st.file_uploader("ملف العدّادات (CSV/Excel)", type=["csv","xlsx","xls"])
with c2:
    transf_file = st.file_uploader("ملف المحوّلات (CSV/Excel)", type=["csv","xlsx","xls"])

# =========== اكتشاف أسماء الأعمدة تلقائياً ===========
def suggest_col(df, candidates):
    if df is None: 
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def detect_mapping(meters_df, transf_df):
    # العدّادات
    m_lon = suggest_col(meters_df, ["lon","x","longitude"])
    m_lat = suggest_col(meters_df, ["lat","y","latitude"])
    m_id  = suggest_col(meters_df, ["meter_id","meterid","id","رقم_العداد","رقم العداد"])

    # المحوّلات
    t_lon = suggest_col(transf_df, ["Lon","lon","x","longitude"])
    t_lat = suggest_col(transf_df, ["Lat","lat","y","latitude"])
    t_id  = suggest_col(transf_df, ["transformer_id","transformerid","tx_id","txid","id","رقم_المحول","رقم المحول"])

    return (m_lon, m_lat, m_id), (t_lon, t_lat, t_id)

# =========== المسافة ===========
R = 6371000.0  # متر

def haversine_batch(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def sph2cart(lat, lon):
    clat = np.cos(lat)
    return np.column_stack((clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)))

# =========== المعالجة ===========
def process(meters_df, m_lon, m_lat, m_id, transf_df, t_lon, t_lat, t_id, max_meters, dist_threshold):
    keep_m = [c for c in [m_lon, m_lat, m_id] if c is not None]
    keep_t = [c for c in [t_lon, t_lat, t_id] if c is not None]

    meters = meters_df[keep_m].dropna().copy()
    transf  = transf_df[keep_t].dropna().copy()

    rename_m = {m_lon:"lon", m_lat:"lat"}
    if m_id: rename_m[m_id] = "meter_id"
    meters.rename(columns=rename_m, inplace=True)

    rename_t = {t_lon:"Lon", t_lat:"Lat"}
    if t_id: rename_t[t_id] = "transformer_id"
    transf.rename(columns=rename_t, inplace=True)

    # KDTree تقريبي على كرة الوحدة
    meters_rad = np.deg2rad(meters[["lat","lon"]].to_numpy())
    transf_rad = np.deg2rad(transf[["Lat","Lon"]].to_numpy())
    meters_xyz = sph2cart(meters_rad[:,0], meters_rad[:,1])
    transf_xyz  = sph2cart(transf_rad[:,0],  transf_rad[:,1])
    tree = cKDTree(meters_xyz)

    k = min(max_meters, len(meters_xyz)) if len(meters_xyz) > 0 else 1
    d_approx, idxs = tree.query(transf_xyz, k=k)
    if k == 1:
        idxs = idxs.reshape(-1,1)

    nearest_dist = np.full(len(transf), np.inf)
    nearest_meter = np.full(len(transf), np.nan, dtype=object)

    # حساب المسافات الدقيقة لأقرب k عدّادات
    m_lon_arr = meters["lon"].to_numpy()
    m_lat_arr = meters["lat"].to_numpy()

    for i, cand in enumerate(idxs):
        cand = np.atleast_1d(cand)
        dd = haversine_batch(
            transf["Lon"].iloc[i], transf["Lat"].iloc[i],
            m_lon_arr[cand],          m_lat_arr[cand]
        )
        if dd.size:
            j = dd.argmin()
            nearest_dist[i] = dd[j]
            if "meter_id" in meters.columns:
                nearest_meter[i] = meters["meter_id"].to_numpy()[cand][j]

    transf["nearest_meter_dist_m"] = nearest_dist
    if "transformer_id" not in transf.columns:
        transf["transformer_id"] = None
    transf["nearest_meter_id"] = nearest_meter

    missing = transf[ transf["nearest_meter_dist_m"] > dist_threshold ].copy()
    return transf, missing

# =========== تشغيل ===========
if meters_file and transf_file:
    meters_df = read_any(meters_file)
    transf_df  = read_any(transf_file)

    (m_lon, m_lat, m_id), (t_lon, t_lat, t_id) = detect_mapping(meters_df, transf_df)
    need_manual = any(x is None for x in [m_lon, m_lat, t_lon, t_lat])

    if need_manual:
        st.warning("تعذّر التقاط بعض الأعمدة تلقائيًا — اخترها يدويًا.")
        cc1, cc2 = st.columns(2)
        with cc1:
            m_lon = st.selectbox("عمود خط الطول للعداد (lon/x)", options=meters_df.columns)
            m_lat = st.selectbox("عمود خط العرض للعداد (lat/y)", options=meters_df.columns)
            m_id  = st.selectbox("معرّف العدّاد (اختياري)", options=["<لا يوجد>"]+list(meters_df.columns))
            m_id  = None if m_id == "<لا يوجد>" else m_id
        with cc2:
            t_lon = st.selectbox("عمود خط الطول للمحوّل (Lon/x)", options=transf_df.columns)
            t_lat = st.selectbox("عمود خط العرض للمحوّل (Lat/y)", options=transf_df.columns)
            t_id  = st.selectbox("معرّف المحوّل (اختياري)", options=["<لا يوجد>"]+list(transf_df.columns))
            t_id  = None if t_id == "<لا يوجد>" else t_id
    else:
        st.success("تم التقاط الأعمدة تلقائيًا من الملفات.")

    if st.button("ابدأ التحليل"):
        try:
            result_all, result_missing = process(
                meters_df, m_lon, m_lat, m_id,
                transf_df, t_lon, t_lat, t_id,
                max_k, dist_m
            )

            # ====== ملخص إحصائي سريع ======
            total_tx = len(result_all)
            missing_tx = len(result_missing)
            covered_tx = total_tx - missing_tx
            coverage = (covered_tx / total_tx * 100.0) if total_tx else 0.0
            median_dist = float(np.nanmedian(result_all["nearest_meter_dist_m"])) if total_tx else 0.0
            mean_dist   = float(np.nanmean(result_all["nearest_meter_dist_m"]))   if total_tx else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("عدد المحوّلات", f"{total_tx:,}")
            k2.metric("مغطّاة بعدّادات", f"{covered_tx:,}", f"{coverage:.1f}%")
            k3.metric("غير مغطّاة", f"{missing_tx:,}")
            k4.metric("وسيط المسافة (م)", f"{median_dist:,.1f}", f"متوسط {mean_dist:,.1f}")

            # مخطط توزيع المسافات
            st.subheader("توزيع أقرب مسافة لعداد (متر)")
            chart_data = result_all[["nearest_meter_dist_m"]].copy()
            chart_data["nearest_meter_dist_m"] = chart_data["nearest_meter_dist_m"].clip(upper=dist_m*4)
            hist = alt.Chart(chart_data).mark_bar().encode(
                alt.X("nearest_meter_dist_m:Q", bin=alt.Bin(maxbins=40), title="المسافة (م)"),
                alt.Y("count()", title="العدد"),
                tooltip=[alt.Tooltip("count()", title="العدد")]
            ).properties(height=280)
            st.altair_chart(hist, use_container_width=True)

            # قائمة بأبعد 20 محوّل (اختياري)
            st.subheader("أبعد 20 محوّل عن أقرب عدّاد")
            preferred = ["transformer_id","Lat","Lon","nearest_meter_dist_m","nearest_meter_id"]
            cols_all = [c for c in preferred if c in result_all.columns] + [c for c in result_all.columns if c not in preferred]
            st.dataframe(result_all.sort_values("nearest_meter_dist_m", ascending=False).head(20)[cols_all])

            # ====== تنزيل النتائج (المفقودة فقط) ======
            st.subheader("تنزيل")
            cols_mis = [c for c in preferred if c in result_missing.columns] + [c for c in result_missing.columns if c not in preferred]

            # Excel
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                result_missing[cols_mis].to_excel(writer, sheet_name="Missing_Transformers", index=False)
            out_buf.seek(0)
            st.download_button(
                "تنزيل المحوّلات غير المغطّاة (Excel)",
                data=out_buf,
                file_name="missing_transformers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # CSV (أخف وأسرع)
            csv_buf = result_missing[cols_mis].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "تنزيل المحوّلات غير المغطّاة (CSV)",
                data=csv_buf,
                file_name="missing_transformers.csv",
                mime="text/csv"
            )

            st.success("تم الإنهاء بنجاح.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
else:
    st.info("الرجاء رفع ملف العدّادات وملف المحوّلات للمتابعة.")
