# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from io import BytesIO
from streamlit.components.v1 import html

st.set_page_config(page_title="تحليل تغطية المحوّلات", layout="wide")

st.title("تحليل المحوّلات التي لا تقع ضمن نطاق أي عدّاد")

st.markdown("""
- **ملف العدادات (Meters):** يجب أن يحتوي على عمودين:  
  - `x` = خط الطول (lon)  
  - `y` = خط العرض (lat)
- **ملف المحوّلات (Transformers):** يجب أن يحتوي على عمودين:  
  - `Lon` = خط الطول  
  - `Lat` = خط العرض
""")

m_file = st.file_uploader("ارفع ملف العدادات (CSV)", type=["csv"])
t_file = st.file_uploader("ارفع ملف المحوّلات (CSV)", type=["csv"])

threshold = st.number_input("مسافة العتبة بالمتر", min_value=1, value=100, step=10)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

if st.button("ابدأ التحليل") and m_file and t_file:
    # قراءة الملفات
    meters = pd.read_csv(m_file)
    transf  = pd.read_csv(t_file)

    # توحيد أسماء الأعمدة
    if not {'x','y'}.issubset(meters.columns):
        st.error("ملف العدادات يجب أن يحتوي على الأعمدة x و y")
        st.stop()
    if not {'Lat','Lon'}.issubset(transf.columns):
        st.error("ملف المحوّلات يجب أن يحتوي على الأعمدة Lat و Lon")
        st.stop()

    meters = meters.rename(columns={"x":"lon","y":"lat"})
    meters["lat"] = pd.to_numeric(meters["lat"], errors="coerce")
    meters["lon"] = pd.to_numeric(meters["lon"], errors="coerce")
    meters = meters.dropna(subset=["lat","lon"]).reset_index(drop=True)

    transf["Lat"] = pd.to_numeric(transf["Lat"], errors="coerce")
    transf["Lon"] = pd.to_numeric(transf["Lon"], errors="coerce")
    transf = transf.dropna(subset=["Lat","Lon"]).reset_index(drop=True)

    st.write(f"عدد العدادات: {len(meters)} | عدد المحوّلات: {len(transf)}")

    # حساب أقرب عدّاد لكل محوّل
    mlat = meters["lat"].to_numpy()
    mlon = meters["lon"].to_numpy()

    min_dists = []
    nearest_idx = []
    for _, r in transf.iterrows():
        d = haversine_m(r["Lat"], r["Lon"], mlat, mlon) if len(meters)>0 else np.array([np.nan])
        j = int(np.nanargmin(d)) if len(d)>0 else None
        md = float(d[j]) if len(d)>0 else np.nan
        min_dists.append(md)
        nearest_idx.append(j)

    transf["nearest_meter_index"] = nearest_idx
    transf["nearest_meter_distance_m"] = min_dists

    # اختيار المحوّلات خارج العتبة
    missing = transf[
        transf["nearest_meter_distance_m"].isna() |
        (transf["nearest_meter_distance_m"] > threshold)
    ].copy()

    # إحداثيات أقرب عدّاد (لو موجود)
    def nm_lat(idx):
        return meters.loc[idx,"lat"] if pd.notna(idx) and 0 <= idx < len(meters) else np.nan
    def nm_lon(idx):
        return meters.loc[idx,"lon"] if pd.notna(idx) and 0 <= idx < len(meters) else np.nan

    missing["nearest_meter_lat"] = missing["nearest_meter_index"].apply(nm_lat)
    missing["nearest_meter_lon"] = missing["nearest_meter_index"].apply(nm_lon)

    st.success(f"عدد المحوّلات خارج {threshold} م: {len(missing)}")

    # عرض الجدول + تنزيل CSV
    st.subheader("النتيجة (جدول)")
    st.dataframe(missing)

    csv_bytes = missing.to_csv(index=False).encode("utf-8-sig")
    st.download_button("تنزيل CSV للنتيجة", data=csv_bytes, file_name="transformers_without_meters.csv", mime="text/csv")

    # خريطة تفاعلية
    st.subheader("الخريطة")
    if len(transf) > 0:
        center = [transf["Lat"].mean(), transf["Lon"].mean()]
    else:
        center = [24.7136, 46.6753]  # fallback

    fmap = folium.Map(location=center, zoom_start=11, control_scale=True)

    # جميع المحوّلات بنقاط رمادية صغيرة
    for _, r in transf.iterrows():
        folium.CircleMarker([r["Lat"], r["Lon"]], radius=2, color="#999", fill=True, fill_opacity=0.4).add_to(fmap)

    # المحوّلات خارج العتبة باللون الأحمر + خط يصل لأقرب عدّاد
    for _, r in missing.iterrows():
        folium.Marker(
            [r["Lat"], r["Lon"]],
            icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
            popup=f"Dist: {round(r['nearest_meter_distance_m'],1)} m"
        ).add_to(fmap)
        if pd.notna(r["nearest_meter_lat"]):
            folium.PolyLine(
                [[r["Lat"], r["Lon"]],
                 [r["nearest_meter_lat"], r["nearest_meter_lon"]]],
                color="orange", weight=2, opacity=0.8
            ).add_to(fmap)

    html_str = fmap._repr_html_()
    html(html_str, height=600)
