# -*- coding: utf-8 -*-
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree
import altair as alt

# ----------------------------- الإعدادات العامة -----------------------------
st.set_page_config(page_title="التحقق من تغطية المحوّلات بالعدادات", layout="wide")

st.title("🔎 نظام فرز المحوّلات التي لا يقع ضمن نطاقها عدّادات")
st.write(
    "ارفع ملف العدّادات وملف المحوّلات (CSV أو Excel). سيقوم النظام بحساب أقرب عدّاد لكل محوّل "
    "ثم يعرض المحوّلات التي لا يوجد قربها عدّاد ضمن مسافة العتبة التي تحددها."
)

# ----------------------------- قوالب الإدخال (Templates) -----------------------------
with st.expander("📄 تنزيل قوالب الملفات (اختياري)"):
    st.markdown("قم بتنزيل القوالب، عبّئها بنفس الأعمدة ثم ارفعها في النظام.")
    # قالب العدادات
    meters_tpl = pd.DataFrame(
        {
            "Meter_ID": ["M-0001", "M-0002"],
            "Lat": [24.7136, 24.7150],
            "Lon": [46.6753, 46.6800],
        }
    )
    # قالب المحولات
    trans_tpl = pd.DataFrame(
        {
            "Transformer_ID": ["T-1001", "T-1002"],
            "Lat": [24.7148, 24.7170],
            "Lon": [46.6790, 46.6820],
        }
    )

    def _download_excel_button(df, label, filename):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        st.download_button(label, data=buf.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    _download_excel_button(meters_tpl, "📥 تنزيل قالب العدّادات (Excel)", "meter_template.xlsx")
    _download_excel_button(trans_tpl, "📥 تنزيل قالب المحوّلات (Excel)", "transformer_template.xlsx")

st.divider()

# ----------------------------- رفع الملفات -----------------------------
col_u1, col_u2 = st.columns(2)
with col_u1:
    meters_file = st.file_uploader("📤 ملف العدّادات (CSV/Excel)", type=["csv", "xlsx", "xls"], key="meters")
with col_u2:
    trans_file = st.file_uploader("📤 ملف المحوّلات (CSV/Excel)", type=["csv", "xlsx", "xls"], key="transformers")

# ----------------------------- أدوات مساعدة -----------------------------
_LAT_CAND = ["lat", "latitude", "y", "Lat", "LAT", "Y"]
_LON_CAND = ["lon", "longitude", "x", "Lon", "LON", "X"]
_MTR_ID_CAND = ["meter_id", "meter", "id", "subscription", "Meter_ID", "Number", "رقم العداد", "رقم المشترك"]
_TRF_ID_CAND = ["transformer_id", "transformer", "id", "Transformer_ID", "رقم المحول"]

def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, encoding="utf-8-sig")
    return pd.read_excel(file)

def pick_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def rename_standard(df, lat_cand, lon_cand, id_cand=None):
    lat = pick_col(df, lat_cand)
    lon = pick_col(df, lon_cand)
    if id_cand:
        cid = pick_col(df, id_cand)
    else:
        cid = None
    # نعيد فقط الأعمدة المطلوبة
    out = pd.DataFrame()
    if cid is not None:
        out["id"] = df[cid]
    else:
        out["id"] = np.arange(1, len(df) + 1)
    out["lat"] = pd.to_numeric(df[lat], errors="coerce")
    out["lon"] = pd.to_numeric(df[lon], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out

def deg_to_rad(a):
    return np.deg2rad(a.values.astype(float))

def nearest_distance_meters(meters_df, trans_df):
    """
    يحسب أقرب مسافة (متر) لكل محوّل لأقرب عدّاد باستخدام BallTree مع haversine.
    """
    if meters_df.empty or trans_df.empty:
        return np.array([])

    # حوّل إلى راديان (lat, lon)
    m_lat_r = deg_to_rad(meters_df["lat"])
    m_lon_r = deg_to_rad(meters_df["lon"])
    t_lat_r = deg_to_rad(trans_df["lat"])
    t_lon_r = deg_to_rad(trans_df["lon"])

    meters_rad = np.c_[m_lat_r, m_lon_r]
    trans_rad = np.c_[t_lat_r, t_lon_r]

    tree = BallTree(meters_rad, metric="haversine")
    # k=1 أقرب نقطة
    dist_rad, _ = tree.query(trans_rad, k=1)
    # التحويل إلى متر (نصف قطر الأرض)
    dist_m = dist_rad.flatten() * 6371000.0
    return dist_m

# ----------------------------- واجهة التحليل -----------------------------
st.subheader("⚙️ إعداد التحليل")
col_c1, col_c2 = st.columns(2)
with col_c1:
    max_distance_m = st.number_input("مسافة العتبة (متر) لاعتبار المحوّل مغطّى بعدّاد", min_value=10, max_value=10000, value=100, step=10)
with col_c2:
    show_manual_mapping = st.toggle("تعيين الأعمدة يدويًا (اختياري)", value=False, help="يتم الكشف تلقائيًا عن الأعمدة. فعّل هذا الخيار فقط لو أردت تغييرها يدويًا.")

meters_df = None
trans_df = None

if meters_file and trans_file:
    raw_meters = read_any(meters_file)
    raw_trans  = read_any(trans_file)

    if not show_manual_mapping:
        # كشف تلقائي
        meters_df = rename_standard(raw_meters, _LAT_CAND, _LON_CAND, _MTR_ID_CAND)
        trans_df  = rename_standard(raw_trans,  _LAT_CAND, _LON_CAND, _TRF_ID_CAND)
    else:
        with st.expander("تعيين الأعمدة يدويًا"):
            st.markdown("### العدّادات")
            m_lat = st.selectbox("عمود خط العرض (lat/y) للعدّادات", raw_meters.columns, index=max(0, next((i for i,c in enumerate(raw_meters.columns) if c.lower() in [x.lower() for x in _LAT_CAND]), 0)))
            m_lon = st.selectbox("عمود خط الطول (lon/x) للعدّادات", raw_meters.columns, index=max(0, next((i for i,c in enumerate(raw_meters.columns) if c.lower() in [x.lower() for x in _LON_CAND]), 0)))
            m_id  = st.selectbox("عمود المعرف/الرقم (اختياري)", ["(استخدام تسلسل تلقائي)"] + list(raw_meters.columns))
            st.markdown("### المحوّلات")
            t_lat = st.selectbox("عمود خط العرض (lat/y) للمحوّلات", raw_trans.columns, index=max(0, next((i for i,c in enumerate(raw_trans.columns) if c.lower() in [x.lower() for x in _LAT_CAND]), 0)))
            t_lon = st.selectbox("عمود خط الطول (lon/x) للمحوّلات", raw_trans.columns, index=max(0, next((i for i,c in enumerate(raw_trans.columns) if c.lower() in [x.lower() for x in _LON_CAND]), 0)))
            t_id  = st.selectbox("عمود معرف المحوّل (اختياري)", ["(استخدام تسلسل تلقائي)"] + list(raw_trans.columns))

        # نبني DataFrame موحّد
        meters_df = pd.DataFrame(
            {
                "id": raw_meters[m_id] if m_id != "(استخدام تسلسل تلقائي)" else np.arange(1, len(raw_meters)+1),
                "lat": pd.to_numeric(raw_meters[m_lat], errors="coerce"),
                "lon": pd.to_numeric(raw_meters[m_lon], errors="coerce"),
            }
        ).dropna(subset=["lat","lon"]).reset_index(drop=True)

        trans_df = pd.DataFrame(
            {
                "id": raw_trans[t_id] if t_id != "(استخدام تسلسل تلقائي)" else np.arange(1, len(raw_trans)+1),
                "lat": pd.to_numeric(raw_trans[t_lat], errors="coerce"),
                "lon": pd.to_numeric(raw_trans[t_lon], errors="coerce"),
            }
        ).dropna(subset=["lat","lon"]).reset_index(drop=True)

# زر التشغيل
run = st.button("▶️ ابدأ التحليل", type="primary", disabled=not (meters_df is not None and trans_df is not None))

if run:
    if meters_df.empty or trans_df.empty:
        st.error("الملفات لا تحتوي بيانات صالحة بعد التنقية. تأكد من الأعمدة وقيم الإحداثيات.")
        st.stop()

    # احسب أقرب مسافة لكل محوّل
    dist_m = nearest_distance_meters(meters_df, trans_df)
    if dist_m.size == 0:
        st.warning("تعذّر حساب المسافات. تحقّق من صحة البيانات.")
        st.stop()

    # نتيجة موحّدة
    result = trans_df.copy()
    result["nearest_meter_distance_m"] = dist_m
    result["covered"] = result["nearest_meter_distance_m"] <= float(max_distance_m)

    # محوّلات غير مغطّاة
    missing = result.loc[~result["covered"], ["id", "lat", "lon", "nearest_meter_distance_m"]]\
                    .sort_values("nearest_meter_distance_m", ascending=False)\
                    .reset_index(drop=True)

    st.success("تم التحليل بنجاح ✅")

    # ----------------------------- إحصائيات سريعة -----------------------------
    st.subheader("📊 إحصائيات")
    total_t = len(result)
    total_m = len(meters_df)
    uncovered = len(missing)
    pct = (uncovered / total_t * 100.0) if total_t else 0.0
    cols = st.columns(4)
    cols[0].metric("عدد المحوّلات", f"{total_t:,}")
    cols[1].metric("عدد العدّادات", f"{total_m:,}")
    cols[2].metric("غير المغطّاة ≤ "+str(int(max_distance_m))+"م", f"{uncovered:,}", f"{pct:.1f}%")
    cols[3].metric("متوسط أقرب مسافة", f"{result['nearest_meter_distance_m'].mean():.1f} م")

    # هيستوجرام للمسافات
    st.markdown("#### توزيع المسافات لأقرب عدّاد (لكل المحوّلات)")
    dist_df = pd.DataFrame({"distance_m": result["nearest_meter_distance_m"]})
    chart = (
        alt.Chart(dist_df)
        .mark_bar()
        .encode(
            x=alt.X("distance_m:Q", bin=alt.Bin(maxbins=40), title="المسافة إلى أقرب عدّاد (م)"),
            y=alt.Y("count()", title="العدد"),
            tooltip=[alt.Tooltip("count()", title="العدد")]
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

    # ----------------------------- تنزيل النتائج -----------------------------
    st.subheader("⬇️ تنزيل النتائج (المحوّلات غير المغطّاة فقط)")
    st.caption("الملف يحتوي فقط على المحوّلات التي أقرب عدّاد لها **أبعد** من مسافة العتبة المحدّدة.")
    cdl1, cdl2 = st.columns(2)

    # CSV
    csv_bytes = missing.to_csv(index=False).encode("utf-8-sig")
    cdl1.download_button(
        "📥 تنزيل CSV",
        data=csv_bytes,
        file_name="uncovered_transformers.csv",
        mime="text/csv",
    )

    # Excel
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        missing.to_excel(writer, index=False, sheet_name="uncovered")
    cdl2.download_button(
        "📥 تنزيل Excel",
        data=xbuf.getvalue(),
        file_name="uncovered_transformers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # معاينة سريعة لأول 100 صف
    st.markdown("#### معاينة (أول 100 صف)")
    st.dataframe(missing.head(100), use_container_width=True)
else:
    st.info("**ارفع ملفي العدّادات والمحوّلات ثم اضغط ‘ابدأ التحليل’.**")
