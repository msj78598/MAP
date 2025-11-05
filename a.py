import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Meterless Transformers", layout="wide")

st.title("تحليل المحوّلات التي لا تقع ضمن نطاق أي عدّاد")

st.markdown(
"""
**وصف مختصر:**  
يراجع هذا النظام مواقع المحوّلات والعدّادات ويحدّد المحوّلات التي لا يوجد ضمن نطاقها أي عدّاد (حسب مسافة تختارها).  
**المدخلات:** ملف العدّادات + ملف المحوّلات (CSV/Excel).  
**المخرجات:** ملف Excel يتضمن المحوّلات بدون عدّادات قريبة + خريطة تفاعلية.
"""
)

# -------- إعدادات عامّة --------
dist_m = st.number_input("المسافة العتبية (متر)", min_value=1, value=100, step=10)

with st.expander("خيارات متقدمة", expanded=False):
    st.caption("عدد أقرب العدّادات التي نفحصها (للدقّة): يُستخدم في البحث التقريبي قبل حساب المسافة الدقيقة. اتركه كما هو في أغلب الحالات.")
    max_k = st.slider("عدد أقرب العدّادات التي نفحصها (للدقّة)", 1, 50, 10)

# -------- أدوات قراءة الملفات --------
def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

st.subheader("رفع الملفات")
c1, c2 = st.columns(2)
with c1:
    meters_file = st.file_uploader("ملف العدّادات (CSV/Excel)", type=["csv","xlsx","xls"])
with c2:
    transf_file = st.file_uploader("ملف المحوّلات (CSV/Excel)", type=["csv","xlsx","xls"])

# -------- اقتراح/التقاط تلقائي لأسماء الأعمدة --------
def suggest_col(df, candidates):
    if df is None: 
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def detect_mapping(meters_df, transf_df):
    # العدادات
    m_lon = suggest_col(meters_df, ["lon","x","longitude"])
    m_lat = suggest_col(meters_df, ["lat","y","latitude"])
    m_id  = suggest_col(meters_df, ["meter_id","meterid","id","رقم_العداد","رقم العداد"])

    # المحولات
    t_lon = suggest_col(transf_df, ["Lon","lon","x","longitude"])
    t_lat = suggest_col(transf_df, ["Lat","lat","y","latitude"])
    t_id  = suggest_col(transf_df, ["transformer_id","transformerid","tx_id","txid","id","رقم_المحول","رقم المحول"])

    return (m_lon, m_lat, m_id), (t_lon, t_lat, t_id)

# -------- المسافة (Haversine) --------
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

# -------- المعالجة --------
def process(meters_df, m_lon, m_lat, m_id, transf_df, t_lon, t_lat, t_id, max_meters, dist_threshold):
    # الإسقاط والتسمية
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

    # إنشاء KDTree تقريبي
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

    for i, cand in enumerate(idxs):
        cand = np.atleast_1d(cand)
        dd = haversine_batch(
            transf["Lon"].iloc[i], transf["Lat"].iloc[i],
            meters["lon"].to_numpy()[cand], meters["lat"].to_numpy()[cand]
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

# -------- التشغيل --------
if meters_file and transf_file:
    meters_df = read_any(meters_file)
    transf_df  = read_any(transf_file)

    (m_lon, m_lat, m_id), (t_lon, t_lat, t_id) = detect_mapping(meters_df, transf_df)

    # إذا فشل الالتقاط التلقائي نظهر للمستخدم تعيين الأعمدة
    need_manual = any(x is None for x in [m_lon, m_lat, t_lon, t_lat])

    if need_manual:
        st.warning("لم أستطع التعرّف تلقائيًا على بعض الأعمدة. الرجاء اختيارها يدويًا.")
        cc1, cc2 = st.columns(2)
        with cc1:
            m_lon = st.selectbox("عمود خط الطول للعداد (lon/x)", options=meters_df.columns, index=0 if m_lon is None else meters_df.columns.get_loc(m_lon))
            m_lat = st.selectbox("عمود خط العرض للعداد (lat/y)", options=meters_df.columns, index=0 if m_lat is None else meters_df.columns.get_loc(m_lat))
            m_id  = st.selectbox("معرّف العدّاد (اختياري)", options=["<لا يوجد>"]+list(meters_df.columns), index=0)
            m_id  = None if m_id == "<لا يوجد>" else m_id
        with cc2:
            t_lon = st.selectbox("عمود خط الطول للمحوّل (Lon/x)", options=transf_df.columns, index=0 if t_lon is None else transf_df.columns.get_loc(t_lon))
            t_lat = st.selectbox("عمود خط العرض للمحوّل (Lat/y)", options=transf_df.columns, index=0 if t_lat is None else transf_df.columns.get_loc(t_lat))
            t_id  = st.selectbox("معرّف المحوّل (اختياري)", options=["<لا يوجد>"]+list(transf_df.columns), index=0)
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

            st.success(f"تم الانتهاء. عدد المحوّلات: {len(result_all)} — بدون عدّادات قريبة: {len(result_missing)}")

            # ترتيب الأعمدة للإخراج
            preferred_order = ["transformer_id","Lat","Lon","nearest_meter_dist_m","nearest_meter_id"]
            cols_all = [c for c in preferred_order if c in result_all.columns] + [c for c in result_all.columns if c not in preferred_order]
            cols_mis = [c for c in preferred_order if c in result_missing.columns] + [c for c in result_missing.columns if c not in preferred_order]

            st.subheader("المحوّلات بدون عدّادات قريبة (أكبر من العتبة)")
            st.dataframe(result_missing[cols_mis].head(500))

            st.subheader("المحوّلات (مع مسافة أقرب عدّاد)")
            st.dataframe(result_all[cols_all].head(500))

            # تنزيل النتائج
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                result_all[cols_all].to_excel(writer, sheet_name="All_Transformers", index=False)
                result_missing[cols_mis].to_excel(writer, sheet_name="Missing_Transformers", index=False)
            out_buf.seek(0)
            st.download_button(
                "تنزيل النتائج (Excel)",
                data=out_buf,
                file_name="transformers_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # -------- خريطة كبيرة + مشاركة --------
            st.subheader("الخريطة")
            if len(result_all) > 0:
                # حساب حدود لاحتواء جميع النقاط
                lat_all = result_all["Lat"].to_numpy()
                lon_all = result_all["Lon"].to_numpy()
                lat_min, lat_max = float(lat_all.min()), float(lat_all.max())
                lon_min, lon_max = float(lon_all.min()), float(lon_all.max())
                center = [ (lat_min+lat_max)/2.0, (lon_min+lon_max)/2.0 ]

                fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)

                # كل المحوّلات (رمادي)
                for _, r in result_all.iterrows():
                    folium.CircleMarker(
                        [r["Lat"], r["Lon"]],
                        radius=3, color="#999", fill=True, fill_opacity=0.5
                    ).add_to(fmap)

                # المحوّلات بدون عدّادات — نقاط حمراء + رابط مشاركة
                for _, r in result_missing.iterrows():
                    tx_id = r.get("transformer_id", "")
                    gmaps = f"https://www.google.com/maps?q={r['Lat']},{r['Lon']}"
                    popup_html = f"""
                    <b>Transformer:</b> {tx_id if pd.notna(tx_id) else '-'}<br>
                    <b>Distance to nearest meter:</b> {r.get('nearest_meter_dist_m', np.nan):.1f} m<br>
                    <a href="{gmaps}" target="_blank">فتح/مشاركة في Google Maps</a>
                    """
                    folium.CircleMarker(
                        [r["Lat"], r["Lon"]],
                        radius=7, color="#c23b22", fill=True, fill_opacity=0.9,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(fmap)

                # ملاءمة الحدود
                fmap.fit_bounds([[lat_min, lon_min],[lat_max, lon_max]])

                st_folium(fmap, height=660, width=None)  # تملأ العرض تلقائيًا
            else:
                st.info("لا توجد بيانات لرسم خريطة.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
else:
    st.info("الرجاء رفع ملف العدّادات وملف المحوّلات للمتابعة.")
