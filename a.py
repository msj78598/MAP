# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree

# ===== إعداد الصفحة =====
st.set_page_config(page_title="التحقق من تغطية المحوّلات بالعدادات", layout="wide")
st.title("🔎 نظام فرز المحوّلات التي لا يقع ضمن نطاقها عدّادات")

st.write(
    "ارفع ملف العدّادات وملف المحوّلات (CSV أو Excel). سيقوم النظام بحساب أقرب عدّاد لكل محوّل "
    "ثم يعرض المحوّلات التي لا يوجد قربها عدّاد ضمن مسافة العتبة التي تحددها."
)

# ===== قوالب إدخال جاهزة =====
with st.expander("📄 تنزيل قوالب الملفات (اختياري)"):
    meters_tpl = pd.DataFrame({"Meter_ID": ["M-0001","M-0002"], "Lat":[24.7136,24.7150], "Lon":[46.6753,46.6800]})
    trans_tpl  = pd.DataFrame({"Transformer_ID": ["T-1001","T-1002"], "Lat":[24.7148,24.7170], "Lon":[46.6790,46.6820]})

    def dl_excel(df, label, filename):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False)
        st.download_button(label, data=buf.getvalue(),
                           file_name=filename,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    dl_excel(meters_tpl, "📥 تنزيل قالب العدّادات (Excel)", "meter_template.xlsx")
    dl_excel(trans_tpl,  "📥 تنزيل قالب المحوّلات (Excel)", "transformer_template.xlsx")

st.divider()

# ===== رفع الملفات =====
c1, c2 = st.columns(2)
with c1:
    meters_file = st.file_uploader("📤 ملف العدّادات (CSV/Excel)", type=["csv","xlsx","xls"], key="meters")
with c2:
    trans_file  = st.file_uploader("📤 ملف المحوّلات (CSV/Excel)", type=["csv","xlsx","xls"], key="transformers")

# ===== أدوات مساعدة =====
_LAT_CAND   = ["lat","latitude","y","Lat","LAT","Y"]
_LON_CAND   = ["lon","longitude","x","Lon","LON","X"]
_MTR_ID_CAND= ["meter_id","meter","id","subscription","Meter_ID","Number","رقم العداد","رقم المشترك"]
_TRF_ID_CAND= ["transformer_id","transformer","id","Transformer_ID","رقم المحول"]

def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        # جرّب UTF-8 ثم CP1256 تلقائيًا
        for enc in ("utf-8-sig","cp1256","latin1"):
            try:
                return pd.read_csv(file, encoding=enc)
            except Exception:
                file.seek(0)
        # آخر محاولة بدون ترميز محدد
        file.seek(0)
        return pd.read_csv(file)
    return pd.read_excel(file)

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def sanitize_latlon(df, lat_col, lon_col):
    """تحويل إحداثيات إلى أرقام وتصفية أي قيم خارج النطاق."""
    out = df.copy()
    out[lat_col] = to_num(out[lat_col])
    out[lon_col] = to_num(out[lon_col])
    out = out.dropna(subset=[lat_col, lon_col])
    out = out[(out[lat_col].between(-90, 90)) & (out[lon_col].between(-180, 180))]
    return out.reset_index(drop=True)

def standardize(df, lat_cand, lon_cand, id_cand=None):
    lat = find_col(df, lat_cand)
    lon = find_col(df, lon_cand)
    cid = find_col(df, id_cand) if id_cand else None

    if (lat is None) or (lon is None):
        # إعادة None لتجبرنا نستخدم التعيين اليدوي بدل ما نكسر بكود None
        return None

    df = sanitize_latlon(df, lat, lon)
    if df.empty:
        return pd.DataFrame(columns=["id","lat","lon"])

    out = pd.DataFrame()
    out["id"] = df[cid] if cid is not None else np.arange(1, len(df)+1)
    out["lat"] = df[lat].astype(float)
    out["lon"] = df[lon].astype(float)
    out = out.dropna(subset=["lat","lon"]).reset_index(drop=True)
    return out

def deg_to_rad(a):
    return np.deg2rad(a.astype(float).values)

def nearest_distance_meters(meters_df, trans_df):
    if meters_df.empty or trans_df.empty:
        return np.array([])
    m_rad = np.c_[deg_to_rad(meters_df["lat"]), deg_to_rad(meters_df["lon"])]
    t_rad = np.c_[deg_to_rad(trans_df["lat"]),  deg_to_rad(trans_df["lon"])]
    tree = BallTree(m_rad, metric="haversine")
    dist_rad, _ = tree.query(t_rad, k=1)
    return dist_rad.flatten() * 6371000.0  # إلى متر

# ===== إعداد التحليل =====
st.subheader("⚙️ إعداد التحليل")
c3, c4 = st.columns(2)
with c3:
    max_distance_m = st.number_input("مسافة العتبة (متر) لاعتبار المحوّل مغطّى", min_value=10, max_value=10000, value=100, step=10)
with c4:
    manual = st.toggle("تعيين الأعمدة يدويًا (اختياري)", value=False)

meters_df = None
trans_df  = None

if meters_file and trans_file:
    raw_m = read_any(meters_file)
    raw_t = read_any(trans_file)

    if not isinstance(raw_m, pd.DataFrame) or not isinstance(raw_t, pd.DataFrame):
        st.error("تعذّر قراءة الملفات. تأكّد من صحتها.")
        st.stop()

    if not manual:
        # كشف تلقائي مع فحص فشل الكشف
        meters_df = standardize(raw_m, _LAT_CAND, _LON_CAND, _MTR_ID_CAND)
        trans_df  = standardize(raw_t, _LAT_CAND, _LON_CAND, _TRF_ID_CAND)

        if meters_df is None or trans_df is None:
            st.warning("تعذّر الكشف التلقائي عن أعمدة الإحداثيات. فعّل خيار **تعيين الأعمدة يدويًا** بالأسفل.")
            manual = True

    if manual:
        with st.expander("تعيين الأعمدة يدويًا"):
            st.markdown("### العدّادات")
            m_lat = st.selectbox("عمود خط العرض (lat/y) للعدّادات", raw_m.columns)
            m_lon = st.selectbox("عمود خط الطول (lon/x) للعدّادات", raw_m.columns)
            m_id  = st.selectbox("عمود المعرف/الرقم (اختياري)", ["(تسلسل تلقائي)"] + list(raw_m.columns))

            st.markdown("### المحوّلات")
            t_lat = st.selectbox("عمود خط العرض (lat/y) للمحوّلات", raw_t.columns)
            t_lon = st.selectbox("عمود خط الطول (lon/x) للمحوّلات", raw_t.columns)
            t_id  = st.selectbox("عمود معرف المحوّل (اختياري)", ["(تسلسل تلقائي)"] + list(raw_t.columns))

        # تنظيف وتحويل
        raw_m = sanitize_latlon(raw_m, m_lat, m_lon)
        raw_t = sanitize_latlon(raw_t, t_lat, t_lon)

        meters_df = pd.DataFrame({
            "id": raw_m[m_id] if m_id != "(تسلسل تلقائي)" else np.arange(1, len(raw_m)+1),
            "lat": raw_m[m_lat].astype(float),
            "lon": raw_m[m_lon].astype(float),
        })
        trans_df = pd.DataFrame({
            "id": raw_t[t_id] if t_id != "(تسلسل تلقائي)" else np.arange(1, len(raw_t)+1),
            "lat": raw_t[t_lat].astype(float),
            "lon": raw_t[t_lon].astype(float),
        })

# زر التشغيل
run = st.button("▶️ ابدأ التحليل", type="primary",
                disabled=not (isinstance(meters_df, pd.DataFrame) and isinstance(trans_df, pd.DataFrame)))

if run:
    # تحقّقات آمنة
    if meters_df is None or trans_df is None:
        st.error("لا توجد بيانات صالحة للعدّادات/المحوّلات.")
        st.stop()
    if meters_df.empty:
        st.error("ملف العدّادات فارغ بعد التنقية/التعرّف على الأعمدة.")
        st.stop()
    if trans_df.empty:
        st.error("ملف المحوّلات فارغ بعد التنقية/التعرّف على الأعمدة.")
        st.stop()

    # احسب أقرب مسافة لكل محوّل
    dist_m = nearest_distance_meters(meters_df, trans_df)
    if dist_m.size == 0:
        st.warning("تعذّر حساب المسافات. تحقّق من صحة البيانات.")
        st.stop()

    # نتائج
    result = trans_df.copy()
    result["nearest_meter_distance_m"] = dist_m
    result["covered"] = result["nearest_meter_distance_m"] <= float(max_distance_m)

    missing = (
        result.loc[~result["covered"], ["id","lat","lon","nearest_meter_distance_m"]]
        .sort_values("nearest_meter_distance_m", ascending=False)
        .reset_index(drop=True)
    )

    st.success("تم التحليل بنجاح ✅")

    # إحصائيات
    st.subheader("📊 إحصائيات")
    total_t = len(result)
    total_m = len(meters_df)
    uncovered = len(missing)
    pct = (uncovered / total_t * 100.0) if total_t else 0.0
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("عدد المحوّلات", f"{total_t:,}")
    k2.metric("عدد العدّادات", f"{total_m:,}")
    k3.metric(f"غير المغطّاة ≤ {int(max_distance_m)}م", f"{uncovered:,}", f"{pct:.1f}%")
    k4.metric("متوسط أقرب مسافة", f"{result['nearest_meter_distance_m'].mean():.1f} م")

    # تنزيل النتائج (غير المغطّاة فقط)
    st.subheader("⬇️ تنزيل النتائج (المحوّلات غير المغطّاة فقط)")
    cdl1, cdl2 = st.columns(2)
    csv_bytes = missing.to_csv(index=False).encode("utf-8-sig")
    cdl1.download_button("📥 تنزيل CSV", data=csv_bytes,
                         file_name="uncovered_transformers.csv", mime="text/csv")
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as w:
        missing.to_excel(w, index=False, sheet_name="uncovered")
    cdl2.download_button("📥 تنزيل Excel", data=xbuf.getvalue(),
                         file_name="uncovered_transformers.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("#### معاينة (أول 100 صف)")
    st.dataframe(missing.head(100), use_container_width=True)
else:
    st.info("**ارفع ملفي العدّادات والمحوّلات ثم اضغط ‘ابدأ التحليل’.**")
