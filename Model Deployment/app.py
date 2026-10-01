"""Streamlit inference UI for the supplied Olist seller churn pipeline."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


THRESHOLD = 0.5
MODEL_PATH = Path(__file__).with_name("model_parameters.json")
FEATURE_LABELS = {
    "total_orders": "Total order historis",
    "days_since_last_order": "Hari sejak order terakhir",
    "avg_gap_antar_order": "Rata-rata jeda antarorder (hari)",
    "total_active_order_days": "Jumlah hari aktif bertransaksi",
    "unique_products": "Jumlah produk unik terjual",
    "total_product_price": "Total harga item terjual (BRL)",
    "min_product_price": "Harga item terendah (BRL)",
    "max_payment_value": "Nilai pembayaran order tertinggi (BRL)",
    "max_payment_installments": "Jumlah cicilan tertinggi",
    "avg_product_name_length": "Rata-rata panjang nama produk",
    "unique_product_categories": "Jumlah unik kategori produk",
    "min_seller_customer_distance_km": "Minimum jarak seller–customer (km)",
    "median_seller_customer_distance_km": "Median jarak seller–customer (km)",
    "max_seller_customer_distance_km": "Maksimum jarak seller–customer (km)",
}
INTEGER_FEATURES = {
    "total_orders", "total_active_order_days", "unique_products",
    "max_payment_installments", "unique_product_categories",
}


@st.cache_resource
def load_model():
    # Parameters exported from the supplied fitted pipeline; no pickle loading at runtime.
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    features = model.get("features", [])
    if (model.get("format") != "olist-seller-churn-logreg-v1"
            or model.get("positive_class") != 1
            or set(features) != set(FEATURE_LABELS)):
        raise ValueError("Parameter model tidak sesuai dengan aplikasi.")
    for key in ("imputer_median", "robust_center", "robust_scale", "coefficients"):
        values = np.asarray(model.get(key, []), dtype=float)
        if len(values) != len(features) or not np.isfinite(values).all():
            raise ValueError(f"Parameter model '{key}' tidak valid.")
    if (np.asarray(model["robust_scale"]) <= 0).any():
        raise ValueError("Skala model harus positif.")
    return model


def validate_input(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = [feature for feature in features if feature not in frame.columns]
    if missing_columns:
        raise ValueError("Kolom wajib belum ada: " + ", ".join(missing_columns))

    values = frame.loc[:, features].copy()

    def parse_number(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        raw = str(value).strip().replace("\u00a0", " ").replace("\u202f", " ")
        if raw.lower() in {"", "nan", "na", "n/a", "null", "none", "-"}:
            return np.nan
        # Accept decimal comma (12,5), BR thousands (1.234,56), and US (1,234.56).
        raw = re.sub(r"\s+", "", raw)
        if "," in raw and "." in raw:
            raw = raw.replace(".", "").replace(",", ".") if raw.rfind(",") > raw.rfind(".") else raw.replace(",", "")
        elif "," in raw:
            raw = raw.replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            return np.nan

    for feature in features:
        original = values[feature]
        values[feature] = original.map(parse_number)
        empty_markers = original.astype("string").str.strip().str.lower().isin(
            ["", "nan", "na", "n/a", "null", "none", "-"]
        )
        invalid = original.notna() & ~empty_markers & values[feature].isna()
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
            examples = original[invalid].astype(str).head(3).tolist()
            raise ValueError(
                f"'{feature}' berisi teks/non-numerik pada baris CSV {rows}. "
                f"Nilai yang terbaca: {examples}. Gunakan angka, misalnya 12.5 atau 12,5."
            )
        if np.isinf(values[feature].to_numpy(dtype=float)).any():
            raise ValueError(f"'{feature}' mengandung nilai tak terhingga.")
        if (values[feature].dropna() < 0).any():
            raise ValueError(f"'{feature}' tidak boleh negatif.")
        if feature in INTEGER_FEATURES:
            non_missing = values[feature].dropna()
            if (non_missing % 1 != 0).any():
                raise ValueError(f"'{feature}' harus bilangan bulat.")

    bounds = json.loads(MODEL_PATH.read_text(encoding="utf-8"))["bounds"]
    for feature in features:
        lower, upper = bounds[feature]["min"], bounds[feature]["max"]
        outside = values[feature].notna() & ((values[feature] < lower) | (values[feature] > upper))
        if outside.any():
            rows = (np.flatnonzero(outside.to_numpy())[:5] + 2).tolist()
            raise ValueError(f"'{feature}' harus dalam rentang {lower:.12g}–{upper:.12g}. Nilai di luar batas pada baris CSV {rows}.")

    warnings = []
    if values.isna().any().any():
        warnings.append(
            "Nilai kosong pada fitur numerik diisi dengan median data latih oleh pipeline model."
        )
    if "days_since_last_order" in values and (values["days_since_last_order"] >= 30).any():
        warnings.append(
            "Ada seller yang telah 30 hari atau lebih tanpa order. Berdasarkan definisi status proyek, "
            "seller tersebut sudah memenuhi kriteria tidak aktif; probabilitas model bukan prediksi dini."
        )
    return values, pd.DataFrame({"Peringatan": warnings})


def score(model: dict, frame: pd.DataFrame, features: list[str], threshold: float):
    values, warnings = validate_input(frame, features)
    arr = values.to_numpy(dtype=float)
    median = np.asarray(model["imputer_median"], dtype=float)
    center = np.asarray(model["robust_center"], dtype=float)
    scale = np.asarray(model["robust_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    arr = np.where(np.isnan(arr), median, arr)
    scaled = (arr - center) / scale
    linear_score = scaled @ coefficients + model["intercept"]
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear_score, -700, 700)))
    result = frame.copy()
    result["probability_churn"] = probability
    result["predicted_churn"] = (probability >= threshold).astype(int)
    result["status_prediksi"] = np.where(result["predicted_churn"] == 1, "Churn", "Aktif")
    return result, warnings


st.set_page_config(page_title="Olist | Seller Churn", page_icon="📦", layout="wide")
st.markdown("""
<style>
    :root { --navy:#14233d; --teal:#782339; --gold:#e9ad43; --pale:#f4f8fb; }
    .stApp { background: linear-gradient(180deg,#edf5f8 0%,#f8fbfd 320px,#ffffff 700px); }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { background: #14233d; }
    [data-testid="stSidebar"] * { color: #f5f9fb !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] * { color: #f5f9fb !important; }
    .hero { background: linear-gradient(115deg,#132642,#651e35 75%,#8b2539);
            border-radius: 22px; padding: 34px 42px; color: white; margin-bottom: 22px;
            box-shadow: 0 16px 36px rgba(19,38,66,.16); }
    .brand-lockup { display:inline-flex; align-items:center; background:white;
                    border-radius:11px; padding:8px 14px; margin-bottom:20px; min-width:120px; }
    .brand-lockup img { display:block; width:91px; height:33px; object-fit:contain; }
    .eyebrow { display:inline-block; color:#ffe0a0; font-size:.76rem; font-weight:800;
               letter-spacing:.15em; text-transform:uppercase; margin-bottom:8px; }
    .hero h1 { margin:0 0 8px; font-size:2.5rem; line-height:1.12; color:white; }
    .hero p { margin:0; max-width:740px; color:#d9eef1; font-size:1.02rem; }
    .mini-card { padding:18px 20px; background:#fff; border:1px solid #dce9ed;
                 border-radius:16px; min-height:112px; box-shadow:0 5px 16px rgba(16,45,65,.045); }
    .mini-card .icon { font-size:1.35rem; display:block; margin-bottom:7px; }
    .mini-card strong { color:#152b46; font-size:.94rem; }
    .mini-card small { display:block; color:#597184; line-height:1.45; margin-top:4px; }
    .section-kicker { color:#087f8c; font-weight:800; letter-spacing:.12em;
                      font-size:.75rem; text-transform:uppercase; margin:18px 0 0; }
    div[data-testid="stMetric"] { background:white; border:1px solid #dce9ed;
      border-radius:16px; padding:17px 20px; box-shadow:0 5px 16px rgba(16,45,65,.045); }
    div[data-testid="stMetricValue"] { color:#132d48; }
    div[data-testid="stForm"] { background:white; border:1px solid #dce9ed;
      border-radius:18px; padding:24px; box-shadow:0 8px 26px rgba(16,45,65,.06); }
    .stTabs [data-baseweb="tab-list"] { gap:12px; }
    .stTabs [data-baseweb="tab"] { border-radius:12px; padding:10px 18px; }
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
      background:#782339; border-color:#782339; border-radius:10px; font-weight:700; }
</style>
<div class="hero">
  <span class="brand-lockup"><img src="https://d3hw41hpah8tvx.cloudfront.net/images/logo_olist_rebrand_62df2f9c0a.svg" alt="Olist"></span><br>
  <span class="eyebrow">Olist · Seller Analytics</span>
  <h1>Seller Churn Predictor</h1>
  <p>Ubah riwayat transaksi menjadi skor risiko untuk membantu prioritas pemantauan seller.</p>
</div>
""", unsafe_allow_html=True)

cards = st.columns(3)
for column, icon, title, caption in zip(
    cards,
    ["◉", "▦", "↗"],
    ["Prediksi individual", "Analisis CSV", "Hasil siap diunduh"],
    ["Masukkan profil satu seller.", "Proses banyak seller sekaligus.", "Urutkan dan tindak lanjuti skor risiko."],
):
    column.markdown(
        f'<div class="mini-card"><span class="icon">{icon}</span>'
        f'<strong>{title}</strong><small>{caption}</small></div>',
        unsafe_allow_html=True,
    )

try:
    pipeline = load_model()
    feature_names = pipeline["features"]
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Informasi prediksi")
    st.info("Threshold default 0,5 (50%). Probabilitas churn ≥50% diklasifikasikan sebagai Churn, sedangkan <50% sebagai Active.")
    st.caption("Model final menggunakan 14 fitur historis pada level seller.")
    st.caption("Batas input mengikuti minimum dan maksimum dari data train.")

threshold = THRESHOLD

st.markdown('<div class="section-kicker">Workspace prediksi</div>', unsafe_allow_html=True)
single_tab, batch_tab, guide_tab = st.tabs(["👤 Satu seller", "📁 Unggah CSV", "📖 Panduan fitur"])

with single_tab:
    st.subheader("Profil seller")
    st.caption("Isi data historis seller. Kolom yang tidak tersedia dapat dikosongkan; model menggunakan median data latih.")
    with st.form("seller_prediction"):
        inputs = {}
        groups = [
            ("01 / Aktivitas transaksi", feature_names[:5]),
            ("02 / Nilai produk & pembayaran", feature_names[5:9]),
            ("03 / Produk & jangkauan", feature_names[9:]),
        ]
        for heading, group_features in groups:
            st.markdown(f"#### {heading}")
            columns = st.columns(2)
            for i, feature in enumerate(group_features):
                with columns[i % 2]:
                    raw = st.text_input(FEATURE_LABELS[feature], value="", key=feature,
                                        placeholder="Masukkan angka",
                                        help=f"Rentang data train: {pipeline['bounds'][feature]['min']:.12g}–{pipeline['bounds'][feature]['max']:.12g}")
                    st.caption(f"Min: {pipeline['bounds'][feature]['min']:.12g} · Max: {pipeline['bounds'][feature]['max']:.12g}")
                    inputs[feature] = raw.strip() or np.nan
            st.divider()
        submitted = st.form_submit_button("🔎 Hitung risiko churn", type="primary", width="stretch")

    if submitted:
        try:
            row = pd.DataFrame([inputs], columns=feature_names)
            result, warnings = score(pipeline, row, feature_names, threshold)
            for warning in warnings["Peringatan"]:
                st.warning(warning)
            probability = float(result.loc[0, "probability_churn"])
            predicted = int(result.loc[0, "predicted_churn"])
            st.markdown("### Hasil prediksi")
            a, b, c = st.columns(3)
            a.metric("Skor risiko churn", f"{probability:.1%}")
            b.metric("Ambang keputusan", f"{threshold:.0%}")
            c.metric("Status prediksi", "Perlu perhatian" if predicted else "Aktif")
            st.progress(probability, text=f"Skor risiko {probability:.1%}")
            if predicted:
                st.warning("Seller melewati threshold default 50%. Prioritaskan pemeriksaan riwayat dan tindak lanjut.")
            else:
                st.success("Seller berada di bawah threshold default 50%. Lanjutkan pemantauan rutin.")
            st.caption("Probabilitas ini adalah keluaran model; belum tentu terkalibrasi sebagai peluang kejadian sebenarnya.")
        except Exception as exc:
            st.error(str(exc))

with batch_tab:
    st.subheader("Prediksi dalam satu unggahan")
    st.caption("CSV wajib memiliki 14 kolom fitur sesuai template. Kolom tambahan seperti `seller_id` tetap disertakan pada hasil.")
    template = pd.DataFrame(columns=["seller_id", *feature_names])
    st.download_button("Unduh template CSV", template.to_csv(index=False).encode("utf-8-sig"),
                       file_name="template_olist_seller_churn.csv", mime="text/csv")
    uploaded = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded is not None:
        try:
            raw_data = pd.read_csv(io.BytesIO(uploaded.getvalue()), dtype={"seller_id": "string"})
            if raw_data.empty:
                raise ValueError("CSV kosong. Tambahkan setidaknya satu seller.")
            if raw_data.columns.duplicated().any():
                raise ValueError("CSV mempunyai nama kolom ganda.")
            result, warnings = score(pipeline, raw_data, feature_names, threshold)
            for warning in warnings["Peringatan"]:
                st.warning(warning)
            a, b, c = st.columns(3)
            a.metric("Total seller", f"{len(result):,}")
            b.metric("Prediksi churn", f"{result['predicted_churn'].sum():,}")
            c.metric("Persentase ditandai", f"{result['predicted_churn'].mean():.1%}")
            st.markdown("#### Daftar prioritas seller")
            result = result.sort_values("probability_churn", ascending=False, kind="stable")
            st.dataframe(result, width="stretch", hide_index=True,
                         column_config={
                             "probability_churn": st.column_config.ProgressColumn(
                                 "Probabilitas churn", min_value=0, max_value=1, format="percent"
                             ),
                             "predicted_churn": st.column_config.NumberColumn("Label churn", format="%d"),
                         })
            st.download_button("Unduh hasil prediksi", result.to_csv(index=False).encode("utf-8-sig"),
                               file_name="hasil_prediksi_olist_seller_churn.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")

with guide_tab:
    st.subheader("Kamus fitur")
    st.caption("Nama kolom fitur harus sama persis. Semua nilai berupa angka pada level seller.")
    st.dataframe(pd.DataFrame({"Nama Fitur": feature_names,
                               "Definisi": [FEATURE_LABELS[x] for x in feature_names],
                               "Minimum": [pipeline['bounds'][x]['min'] for x in feature_names],
                               "Maksimum": [pipeline['bounds'][x]['max'] for x in feature_names]}),
                 width="stretch", hide_index=True)
    st.caption("`total_product_price` adalah jumlah harga item, tanpa ongkos kirim. "
               "`max_payment_installments` adalah jumlah cicilan terbesar. ")
