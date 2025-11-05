import io
import math
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

# ---------------------------
# 0) قوالب الإدخال (زر تنزيل)
# ---------------------------
def make_template(kind="meters"):
    if kind == "meters":
        df = pd.DataFrame({
            "meter_id": [],
            "lon": [],
            "lat": []
        })
    else:
        df = pd.DataFrame({
            "transformer_id": [],
            "Lon": [],
            "Lat": []
        })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Template", index=False)
    buf.seek(0)
    return buf

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "تنزيل قالب العدّادات (Excel)",
        data=make_template("meters"),
        file_name="meters_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
with c2:
    st.download_button(
        "تنزيل قالب المحوّلات (Excel)",
        data=make_template("transformers"),
        file_name="transformers_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.divider()

# ---------------------------
# 1) إعدادات عامّة
# ---------------------------
dist_m = st.number_input("المسافة العتبية (متر)", min_value=1, value=100, step=10)
st.caption("سيتم اعتبار المحوّل 'بدون عدّادات' إذا كان أقرب عدّاد يبعد أكثر من هذه المسافة.")

# ---------------------------
# 2) رفع الملفات
# ---------------------------
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

# ---------------------------
# 3) توحيد أسماء الأعمدة (Mapping)
# ---------------------------
def suggest_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def column_mapper(df, role):
    """
    role: 'meters' or 'transformers'
    """
    st.markdown(f"**تعيين أعمدة { 'العدّادات' if role=='meters' else 'المحوّلات' }:**")
    col1, col2 = st.columns(2)
    if role == "meters":
        lon_guess = suggest_col(df, ["lon", "x", "longitude"])
        lat_guess = suggest_col(df, ["lat", "y", "latitude"])
        with col1:
            lon_col = st.selectbox("عمود خط الطول (lon/x)", options=df.columns, index=(df.columns.get_loc(lon_guess) if lon_guess in df.columns else 0))
        with col2:
            lat_col = st.selectbox("عمود خط العرض (lat/y)", options=df.columns, index=(df.columns.get_loc(lat_guess) if lat_guess in df.columns else 0))
        return lon_col, lat_col
    else:
        lon_guess = suggest_col(df, ["Lon","lon","x","longitude"])
        lat_guess = suggest_col(df, ["Lat","lat","y","latitude"])
        with col1:
            lon_col = st.selectbox("عمود خط الطول للمحوّل (Lon/x)", options=df.columns, index=(df.columns.get_loc(lon_guess) if lon_guess in df.columns else 0))
        with col2:
            lat_col = st.selectbox("عمود خط العرض للمحوّل (Lat/y)", options=df.columns, index=(df.columns.get_loc(lat_guess) if lat_guess in df.columns else 0))
        return lon_col, lat_col

# ---------------------------
# 4) دوال المسافة والحساب
# ---------------------------
R = 6371000.0  # نصف قطر الأرض بالمتر

def haversine_batch(lon1, lat1, lon2, lat2):
    """
    يحسب المسافة (متر) بين نقطتين/مصفوفتين باستخدام Haversine.
    يقبل مصفوفات numpy ويمكنه التعامل مع البث (broadcast).
    """
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def process(meters_df, meters_lon, meters_lat, transf_df, transf_lon, transf_lat, max_meters):
    # إسقاط صفوف ناقصة الإحداثيات
    meters = meters_df[[meters_lon, meters_lat]].dropna().rename(columns={meters_lon:"lon", meters_lat:"lat"})
    transf  = transf_df[[transf_lon,  transf_lat ]].dropna().rename(columns={transf_lon:"Lon", transf_lat:"Lat"})

    # KD-Tree تقريبي على درجات (لغرض البحث الأوّلي السريع)
    meters_rad = np.deg2rad(meters[["lat","lon"]].to_numpy())
    transf_rad = np.deg2rad(transf[["Lat","Lon"]].to_numpy())

    # تحويل (lat,lon) إلى إحداثيات 3D على القشرة الكروية لتسريع البحث التقريبي
    def sph2cart(lat, lon):
        clat = np.cos(lat)
        return np.column_stack((clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)))

    meters_xyz = sph2cart(meters_rad[:,0], meters_rad[:,1])
    transf_xyz  = sph2cart(transf_rad[:,0],  transf_rad[:,1])

    tree = cKDTree(meters_xyz)

    # البحث لأقرب k مرّة تقريبية ثم نتحقق بدقّة بـ Haversine
    k = min(max_meters, len(meters_xyz)) if len(meters_xyz) > 0 else 1
    d_approx, idxs = tree.query(transf_xyz, k=k)

    # تقلّص الخرج إلى مصفوفة 2D دائماً
    if k == 1:
        idxs = idxs.reshape(-1,1)

    nearest_dist = np.full(len(transf), np.inf)
    for i, cand in enumerate(idxs):
        cand = np.atleast_1d(cand)
        # مسافات حقيقية
        dd = haversine_batch(
            transf["Lon"].iloc[i], transf["Lat"].iloc[i],
            meters["lon"].to_numpy()[cand], meters["lat"].to_numpy()[cand]
        )
        nearest_dist[i] = dd.min() if dd.size else np.inf

    transf["nearest_meter_dist_m"] = nearest_dist
    missing = transf[ transf["nearest_meter_dist_m"] > dist_m ].copy()
    return transf, missing

# ---------------------------
# 5) تشغيل المعالجة
# ---------------------------
if meters_file and transf_file:
    meters_df = read_any(meters_file)
    transf_df  = read_any(transf_file)

    st.success("تم تحميل الملفات بنجاح. حدّد أعمدة الإحداثيات أدناه (إن لزم).")

    st.subheader("تعيين الأعمدة")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("العدّادات")
        m_lon, m_lat = column_mapper(meters_df, "meters")
    with c2:
        st.caption("المحوّلات")
        t_lon, t_lat = column_mapper(transf_df, "transformers")

    st.divider()
    st.subheader("تشغيل التحليل")
    max_k = st.slider("عدد أقرب العدّادات التي نفحصها (للدقّة)", 1, 50, 10, help="يُستخدم للبحث التقريبي قبل حساب المسافة الدقيقة")
    if st.button("ابدأ التحليل"):
        try:
            result_all, result_missing = process(
                meters_df, m_lon, m_lat,
                transf_df, t_lon, t_lat,
                max_k
            )

            st.success(f"تم الانتهاء. عدد المحوّلات: {len(result_all)} — بدون عدّادات قريبة: {len(result_missing)}")

            # عرض الجدولين
            st.subheader("المحوّلات (مع مسافة أقرب عدّاد)")
            st.dataframe(result_all.head(500))

            st.subheader("المحوّلات بدون عدّادات قريبة (أكبر من العتبة)")
            st.dataframe(result_missing.head(500))

            # تنزيل النتائج (Excel)
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                result_all.to_excel(writer, sheet_name="All_Transformers", index=False)
                result_missing.to_excel(writer, sheet_name="Missing_Transformers", index=False)
            out_buf.seek(0)
            st.download_button(
                "تنزيل النتائج (Excel)",
                data=out_buf,
                file_name="transformers_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # خريطة مبسطة
            st.subheader("الخريطة")
            if len(result_all) > 0:
                center = [result_all["Lat"].mean(), result_all["Lon"].mean()]
                fmap = folium.Map(location=center, zoom_start=11, control_scale=True)

                # كل المحوّلات كنقاط رمادية صغيرة
                for _, r in result_all.iterrows():
                    folium.CircleMarker([r["Lat"], r["Lon"]], radius=2, color="#999", fill=True, fill_opacity=0.4).add_to(fmap)

                # المحوّلات بدون عدّادات — نقاط حمراء أكبر
                for _, r in result_missing.iterrows():
                    folium.CircleMarker([r["Lat"], r["Lon"]], radius=5, color="#c23b22", fill=True, fill_opacity=0.8, popup=f"{r.get('nearest_meter_dist_m', np.nan):.1f} m").add_to(fmap)

                st_folium(fmap, height=520)
            else:
                st.info("لا توجد بيانات لرسم خريطة.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
else:
    st.info("الرجاء رفع ملف العدّادات وملف المحوّلات للمتابعة.")
