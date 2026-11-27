import base64
import io
import os
import pickle
import tempfile
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

# ==========================================
# KONFIGURASI MODEL FUNDUS (.H5)
# ==========================================
MODEL_PATH = os.path.join(tempfile.gettempdir(), "best_resnet50_model.h5")
MODEL_RELEASE_URL = (
    "https://github.com/Alfizh05-git/fundus-diagnosis-bdc/releases/download/DL-Model/best_resnet50_model.h5"
)
IMG_SIZE = (300, 300)
CLASS_NAMES = ["Normal", "Atrofi Total", "Atrofi Parsial"]

# ==========================================
# KONFIGURASI MODEL PARAMETER KLINIS (.PKL)
# ==========================================
# Sesuaikan path lokal atau URL GitHub Releases untuk model .pkl Anda
RF_LOCAL_MODEL_PATH = os.path.join("RandomForest_Scoring_System", "RandomForest_Final_Model.pkl")
RF_TEMP_MODEL_PATH = os.path.join(tempfile.gettempdir(), "RandomForest_Final_Model.pkl")
#RF_MODEL_RELEASE_URL = ""  # Isi URL GitHub Releases jika file .pkl tidak di-commit ke repo

RF_EXPLAINER_PATH = os.path.join("RandomForest_Scoring_System", "RandomForest_SHAP_Explainer.pkl")
RF_WEIGHT_CSV_PATH = os.path.join("RandomForest_Scoring_System", "RandomForest_Score_Weight.csv")

# Urutan fitur wajib sama persis dengan X.columns saat training
CLINICAL_FEATURES = [
    "Usia_tahun",
    "Jenis_Kelamin",
    "Keluhan_Utama",
    "Sejak_Kapan_bulan",
    "BCVA_Desimal",
    "Pola_Defek_Lapang_Pandang",
    "MD_tuliskan_tanda_jika_ada_dB",
    "VFI",
]

# Default Score Weight (Permutation Importance) sesuai tabel referensi Anda
DEFAULT_SCORE_WEIGHTS = {
    "Keluhan_Utama": 100.0000,
    "Sejak_Kapan_bulan": 81.0076,
    "MD_tuliskan_tanda_jika_ada_dB": 57.2776,
    "Usia_tahun": 27.4389,
    "Pola_Defek_Lapang_Pandang": 25.3580,
    "VFI": 22.6140,
    "Jenis_Kelamin": 0.0000,
    "BCVA_Desimal": 0.0000,
}

# Mapping pilihan UI ke nilai numerik yang dipelajari model
# (Sesuaikan angka mapping di bawah jika encoding label di dataset Anda berbeda)
GENDER_MAP = {
    "Perempuan": 0,
    "Laki-laki": 1,
}

KELUHAN_MAP = {
    "Tidak ada keluhan / Pemeriksaan rutin": 0,
    "Penglihatan kabur (Blurred vision)": 1,
    "Penurunan tajam penglihatan / Visual loss": 2,
    "Gangguan penglihatan warna / Lapang pandang": 3,
    "Lainnya": 4,
}

POLA_DEFEK_MAP = {
    "Normal / Tidak ada defek": 0.0,
    "Skotoma sentral / sekosentral": 1.0,
    "Hemianopia / Kuadranopia": 2.0,
    "Konstriksi perifer / Defek arkuata": 3.0,
    "Defek difus / Total": 4.0,
}

st.set_page_config(
    page_title="Sistem Diagnosis Atrofi Papil - dr. Sita",
    page_icon="🩺",
    layout="wide",
)

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "active_menu" not in st.session_state:
    st.session_state.active_menu = "Profil Dokter"

if "history" not in st.session_state:
    st.session_state.history = []

if "clinical_result" not in st.session_state:
    st.session_state.clinical_result = None

if "clinical_form_key" not in st.session_state:
    st.session_state.clinical_form_key = 0


# ==========================================
# CSS KUSTOM UNTUK TAMPILAN KARTU HASIL
# ==========================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        .result-banner {
            background-color: #e6f4ea;
            border: 1px solid #ceead6;
            border-radius: 12px;
            padding: 24px 16px;
            text-align: center;
            margin-bottom: 16px;
        }
        .result-banner-title {
            color: #137333;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin: 0;
        }
        .result-banner-sub {
            color: #1e8e3e;
            font-size: 1.05rem;
            font-weight: 500;
            margin-top: 4px;
        }
        .class-card {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }
        .class-card-active {
            background-color: #e6f4ea;
            border: 2px solid #34a853;
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }
        .class-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #202124;
            margin-bottom: 2px;
        }
        .class-card-sub {
            font-size: 0.82rem;
            color: #5f6368;
        }
        .prob-row {
            margin-top: 10px;
            margin-bottom: 12px;
        }
        .prob-label-wrap {
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .prob-bar-bg {
            width: 100%;
            background-color: #e8eaed;
            border-radius: 6px;
            height: 10px;
            overflow: hidden;
        }
        .prob-bar-fill-active {
            background-color: #2e7d32;
            height: 100%;
            border-radius: 6px;
        }
        .prob-bar-fill-other {
            background-color: #5c7cfa;
            height: 100%;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# UTILITAS GAMBAR & MODEL FUNDUS (.H5)
# ==========================================
def get_doctor_avatar() -> Image.Image:
    img = Image.new("RGB", (240, 240), color=(224, 242, 254))
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 35, 180, 155), fill=(30, 64, 175))
    draw.pieslice((30, 145, 210, 275), start=180, end=360, fill=(37, 99, 235))
    return img


def pil_to_png_bytes(img_pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    img_pil.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def render_bytes_image(png_bytes: bytes, caption: str = "", width: int | None = None):
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    style_width = f"width:{width}px; max-width:100%;" if width else "width:100%;"
    caption_html = (
        f"<p style='text-align:center; color:#9ca3af; font-size:0.85rem; margin-top:4px;'>{caption}</p>"
        if caption
        else ""
    )
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{b64_str}" style="{style_width} border-radius:8px;" />
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pil_image(img_pil: Image.Image, caption: str = "", width: int | None = None):
    render_bytes_image(pil_to_png_bytes(img_pil), caption=caption, width=width)


def is_valid_hdf5(filepath: str) -> bool:
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024 * 1024:
        return False
    with open(filepath, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def download_file_from_url(dest_path: str, release_url: str, label: str = "model") -> bool:
    clean_url = (release_url or "").strip()
    if not clean_url:
        return False

    status_placeholder = st.empty()
    status_placeholder.info(f"Mengunduh {label} dari server...")
    try:
        with requests.get(clean_url, stream=True, timeout=300) as resp:
            if resp.status_code != 200:
                status_placeholder.error(f"Gagal mengunduh {label} (HTTP {resp.status_code}).")
                return False
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        status_placeholder.empty()
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception as exc:
        status_placeholder.error(f"Gagal mengunduh {label}: {exc}")
        return False


@st.cache_resource
def load_fundus_model():
    if not is_valid_hdf5(MODEL_PATH):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        with st.spinner("Mengunduh model fundus dari GitHub Releases, mohon tunggu..."):
            downloaded = download_file_from_url(MODEL_PATH, MODEL_RELEASE_URL, label="model fundus (.h5)")
            if not downloaded or not is_valid_hdf5(MODEL_PATH):
                st.error(f"File model `{MODEL_PATH}` tidak tersedia atau tidak valid.")
                return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = model(dummy, training=False)
        return model
    except Exception as exc:
        st.error(f"Gagal memuat model fundus: {exc}")
        return None


def preprocess_fundus_image(img_pil: Image.Image) -> np.ndarray:
    img_resized = img_pil.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype(np.float32)
    return np.expand_dims(img_array, axis=0)


def _find_last_conv_layer_recursive(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return layer
    if hasattr(layer, "layers"):
        for sub_layer in reversed(layer.layers):
            found = _find_last_conv_layer_recursive(sub_layer)
            if found is not None:
                return found
    return None


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, pred_index: int) -> np.ndarray:
    backbone = None
    for layer in model.layers:
        if "resnet50" in layer.name.lower() or "backbone" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        backbone = model

    try:
        conv_layer = backbone.get_layer("conv5_block3_out")
    except Exception:
        conv_layer = _find_last_conv_layer_recursive(backbone)

    if conv_layer is None:
        return np.zeros((10, 10), dtype=np.float32)

    grad_model = tf.keras.models.Model(
        inputs=[backbone.inputs],
        outputs=[conv_layer.output, backbone.output],
    )
    x_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array.copy())

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_preprocessed, training=False)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((10, 10), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy().astype(np.float32)


def overlay_gradcam(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    img = np.array(img_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1.0 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(superimposed)


# ==========================================
# UTILITAS MODEL PARAMETER KLINIS (.PKL + SHAP)
# ==========================================
@st.cache_resource
def load_clinical_scoring_assets():
    """Memuat RandomForest_Final_Model.pkl beserta SHAP Explainer dan Score Weights."""
    model_file = None
    if os.path.exists(RF_LOCAL_MODEL_PATH):
        model_file = RF_LOCAL_MODEL_PATH
    elif os.path.exists("RandomForest_Final_Model.pkl"):
        model_file = "RandomForest_Final_Model.pkl"
    elif os.path.exists(RF_TEMP_MODEL_PATH):
        model_file = RF_TEMP_MODEL_PATH
    elif RF_MODEL_RELEASE_URL.strip():
        if download_file_from_url(RF_TEMP_MODEL_PATH, RF_MODEL_RELEASE_URL, label="model klinis (.pkl)"):
            model_file = RF_TEMP_MODEL_PATH

    if model_file is None:
        return None, None, None

    try:
        with open(model_file, "rb") as f:
            rf_pipeline = pickle.load(f)

        # Muat atau bangun SHAP TreeExplainer
        explainer = None
        if os.path.exists(RF_EXPLAINER_PATH):
            with open(RF_EXPLAINER_PATH, "rb") as f:
                explainer = pickle.load(f)
        else:
            # Ekstrak estimator RandomForest jika dibungkus sklearn Pipeline
            if hasattr(rf_pipeline, "named_steps"):
                rf_estimator = list(rf_pipeline.named_steps.values())[-1]
            else:
                rf_estimator = rf_pipeline
            explainer = shap.TreeExplainer(rf_estimator)

        # Muat Score Weight CSV jika ada, jika tidak gunakan default
        weight_lookup = DEFAULT_SCORE_WEIGHTS.copy()
        if os.path.exists(RF_WEIGHT_CSV_PATH):
            df_w = pd.read_csv(RF_WEIGHT_CSV_PATH)
            if {"Feature", "Score Weight"}.issubset(df_w.columns):
                weight_lookup.update(df_w.set_index("Feature")["Score Weight"].to_dict())

        return rf_pipeline, explainer, weight_lookup
    except Exception as exc:
        st.error(f"Gagal memuat aset model parameter klinis: {exc}")
        return None, None, None


def compute_clinical_shap_values(explainer, transformed_df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Menormalisasi output SHAP menjadi array (1, n_features, n_classes)."""
    raw_values = explainer.shap_values(transformed_df)
    if isinstance(raw_values, list):
        values = np.stack(raw_values, axis=-1)
    else:
        values = np.asarray(raw_values)
        if values.ndim == 2:
            values = values[:, :, np.newaxis]
        elif values.ndim == 3 and values.shape[0] == n_classes:
            values = np.transpose(values, (1, 2, 0))
    return values


def run_clinical_scoring(
    patient_dict: dict,
    rf_pipeline,
    explainer,
    weight_lookup: dict,
) -> dict:
    """Menjalankan prediksi probabilitas & tabel kontribusi SHAP sesuai notebook training."""
    patient_data = pd.DataFrame([patient_dict])[CLINICAL_FEATURES]

    probabilities = np.asarray(rf_pipeline.predict_proba(patient_data)[0], dtype=np.float64)
    predicted_id = int(rf_pipeline.predict(patient_data)[0])
    class_order = list(getattr(rf_pipeline, "classes_", [0, 1, 2]))

    # Transformasi fitur jika pipeline memiliki preprocessor sebelum classifier
    X_for_shap = patient_data.copy()
    if hasattr(rf_pipeline, "steps") and len(rf_pipeline.steps) > 1:
        for step_name, step_obj in rf_pipeline.steps[:-1]:
            if hasattr(step_obj, "transform"):
                X_for_shap = step_obj.transform(X_for_shap)

    shap_values = compute_clinical_shap_values(explainer, X_for_shap, len(class_order))[0]
    expected_values = np.asarray(explainer.expected_value).reshape(-1)
    
    class_tables = {}
    for idx, class_id in enumerate(class_order):
        label = CLASS_NAMES[int(class_id)]
        contribution = shap_values[:, idx]

        table = pd.DataFrame(
            {
                "Feature": CLINICAL_FEATURES,
                "Input Value": patient_data.iloc[0].values,
                "Score Weight": [weight_lookup.get(col, 0.0) for col in CLINICAL_FEATURES],
                "SHAP Contribution": contribution,
                "Contribution (%)": contribution * 100.0,
            }
        )
        table = table.sort_values("Score Weight", ascending=False).reset_index(drop=True)
        class_tables[label] = table

    summary_df = pd.DataFrame(
        {
            "Class": [CLASS_NAMES[int(c)] for c in class_order],
            "SHAP Baseline": expected_values[: len(class_order)],
            "Total SHAP Contribution": shap_values.sum(axis=0),
            "Final Score (%)": probabilities * 100.0,
        }
    )

    pred_idx_in_order = class_order.index(predicted_id)
    predicted_label = CLASS_NAMES[int(predicted_id)]

    return {
        "predicted_class": predicted_label,
        "predicted_idx": pred_idx_in_order,
        "probabilities": probabilities,
        "class_order": class_order,
        "class_tables": class_tables,
        "summary_df": summary_df,
    }


# ==========================================
# HALAMAN LOGIN
# ==========================================
def login_page():
    _, col_mid, _ = st.columns([1, 1.4, 1])
    with col_mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("Welcome to OptiDisc")
        st.caption("Silakan masuk menggunakan akun dokter")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if username.strip() == "drsita" and password == "123456":
                    st.session_state.logged_in = True
                    st.success("Login berhasil!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("Username atau password salah.")


# ==========================================
# HALAMAN MENU 3: DIAGNOSIS PARAMETER KLINIS
# ==========================================
def clinical_diagnosis_page():
    st.header("📋 Diagnosis Parameter Klinis")
    st.caption("Evaluasi risiko atrofi papil optik berbasis data pemeriksaan klinis (Random Forest + SHAP Scoring System)")
    st.divider()

    rf_pipeline, explainer, weight_lookup = load_clinical_scoring_assets()
    if rf_pipeline is None:
        st.error(
            "File model `RandomForest_Final_Model.pkl` tidak ditemukan. "
            "Pastikan file tersimpan di folder `RandomForest_Scoring_System/RandomForest_Final_Model.pkl` "
            "atau isi `RF_MODEL_RELEASE_URL` pada kode."
        )
        return

    fkey = st.session_state.clinical_form_key

    # --- FORM INPUT DATA PASIEN ---
    with st.form(key=f"clinical_form_{fkey}"):
        st.subheader("A. Informasi Pasien")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            patient_id = st.text_input("Nomor Pasien", placeholder="Contoh: Pasien-01")
        with col_a2:
            usia_tahun = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=30, step=1)
        with col_a3:
            jenis_kelamin_label = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])

        st.subheader("B. Riwayat Klinis")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            keluhan_label = st.selectbox("Keluhan Utama", options=list(KELUHAN_MAP.keys()), index=2)
        with col_b2:
            sejak_kapan_bulan = st.number_input(
                "Sejak Kapan (bulan)", min_value=0.0, max_value=600.0, value=3.0, step=0.5
            )

        st.subheader("C. Fungsi Visual & Lapang Pandang")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            bcva_desimal = st.number_input(
                "BCVA (desimal)", min_value=0.000, max_value=2.000, value=0.133, step=0.01, format="%.3f"
            )
            md_db = st.number_input(
                "MD — Mean Deviation (tuliskan tanda jika ada, dB)",
                min_value=-40.00,
                max_value=10.00,
                value=-16.68,
                step=0.10,
                format="%.2f",
            )
        with col_c2:
            pola_defek_label = st.selectbox(
                "Pola Defek Lapang Pandang", options=list(POLA_DEFEK_MAP.keys()), index=2
            )
            vfi = st.number_input(
                "VFI — Visual Field Index (%)", min_value=0.0, max_value=100.0, value=54.0, step=1.0, format="%.1f"
            )

        submitted = st.form_submit_button("Diagnosis Parameter Klinis", type="primary", use_container_width=True)

    if submitted:
        patient_numeric_input = {
            "Usia_tahun": int(usia_tahun),
            "Jenis_Kelamin": GENDER_MAP[jenis_kelamin_label],
            "Keluhan_Utama": KELUHAN_MAP[keluhan_label],
            "Sejak_Kapan_bulan": float(sejak_kapan_bulan),
            "BCVA_Desimal": float(bcva_desimal),
            "Pola_Defek_Lapang_Pandang": POLA_DEFEK_MAP[pola_defek_label],
            "MD_tuliskan_tanda_jika_ada_dB": float(md_db),
            "VFI": float(vfi),
        }

        display_summary = {
            "ID Pasien": patient_id.strip() if patient_id.strip() else "Tanpa ID",
            "Usia": f"{usia_tahun} tahun",
            "Jenis Kelamin": jenis_kelamin_label,
            "Keluhan Utama": keluhan_label,
            "Sejak Kapan": f"{sejak_kapan_bulan} bulan",
            "BCVA (Desimal)": f"{bcva_desimal:.3f}",
            "Pola Defek Lapang Pandang": pola_defek_label,
            "MD (dB)": f"{md_db:.2f} dB",
            "VFI": f"{vfi:.1f}%",
        }

        with st.spinner("Menghitung skor dan kontribusi parameter klinis..."):
            scoring_output = run_clinical_scoring(
                patient_dict=patient_numeric_input,
                rf_pipeline=rf_pipeline,
                explainer=explainer,
                weight_lookup=weight_lookup,
            )

        res_payload = {
            "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "jenis": "Parameter Klinis",
            "pasien": display_summary["ID Pasien"],
            "prediksi": scoring_output["predicted_class"],
            "confidence": f"{np.max(scoring_output['probabilities']) * 100:.2f}%",
            "display_summary": display_summary,
            "scoring_output": scoring_output,
        }
        st.session_state.clinical_result = res_payload
        st.session_state.history.append(res_payload)

    # --- TAMPILAN HASIL DIAGNOSIS (SESUAI REFERENSI UI) ---
    if st.session_state.clinical_result is not None:
        res = st.session_state.clinical_result
        out = res["scoring_output"]
        pred_label = out["predicted_class"]
        probs = out["probabilities"]
        class_order = out["class_order"]

        banner_map = {
            "Normal": ("NORMAL OPTIC DISC", "Papil Normal"),
            "Atrofi Parsial": ("PARTIAL OPTIC DISC ATROPHY", "Papil Atrofi Parsial"),
            "Atrofi Total": ("TOTAL OPTIC DISC ATROPHY", "Papil Atrofi Total"),
        }
        en_title, id_sub = banner_map.get(pred_label, (pred_label.upper(), pred_label))

        st.markdown("---")
        st.subheader("AI Analysis Result")
        st.caption("Clinical Parameter Analysis (Random Forest + SHAP)")

        # 1. Banner Hasil Utama
        st.markdown(
            f"""
            <div class="result-banner">
                <div class="result-banner-title">{en_title}</div>
                <div class="result-banner-sub">{id_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Tiga Kartu Kelas (Normal / Partial / Total)
        card_cols = st.columns(3)
        card_meta = [
            ("Normal", "Normal", "Class 1"),
            ("Atrofi Parsial", "Partial", "Class 2"),
            ("Atrofi Total", "Total", "Class 3"),
        ]
        for col, (internal_name, title_en, sub_cls) in zip(card_cols, card_meta):
            is_active = pred_label == internal_name
            css_cls = "class-card-active" if is_active else "class-card"
            with col:
                st.markdown(
                    f"""
                    <div class="{css_cls}">
                        <div class="class-card-title">{title_en}</div>
                        <div class="class-card-sub">{sub_cls}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 3. Classification Probabilities dengan Progress Bar
        st.markdown("#### Classification probabilities")
        prob_lookup = {
            CLASS_NAMES[int(cid)]: float(probs[i] * 100.0)
            for i, cid in enumerate(class_order)
        }
        prob_display_order = [
            ("Normal optic disc", "Normal"),
            ("Partial optic disc atrophy", "Atrofi Parsial"),
            ("Total optic disc atrophy", "Atrofi Total"),
        ]
        for label_en, key_id in prob_display_order:
            pct = prob_lookup.get(key_id, 0.0)
            bar_cls = "prob-bar-fill-active" if key_id == pred_label else "prob-bar-fill-other"
            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-label-wrap">
                        <span>{label_en}</span>
                        <span><strong>{pct:.2f}%</strong></span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="{bar_cls}" style="width: {min(max(pct, 0.0), 100.0):.2f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4. Clinical Data Summary
        st.markdown("#### Clinical Data Summary")
        summary_table = pd.DataFrame(
            list(res["display_summary"].items()),
            columns=["Parameter", "Data"],
        )
        st.table(summary_table)

        # 5. Detail SHAP Scoring System (Expandable)
        with st.expander("Lihat Detail SHAP Scoring System per Kelas"):
            st.markdown("**Final Scoring Summary**")
            st.dataframe(out["summary_df"].round(4), use_container_width=True, hide_index=True)

            tabs = st.tabs([f"Scoring - {c}" for c in CLASS_NAMES])
            for tab, cname in zip(tabs, CLASS_NAMES):
                with tab:
                    if cname in out["class_tables"]:
                        st.dataframe(out["class_tables"][cname].round(4), use_container_width=True, hide_index=True)

        # 6. Tombol Aksi Lanjutan: Riwayat Diagnosis & Diagnosis Pasien Lain
        st.markdown("---")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("🕒 Riwayat Diagnosis", use_container_width=True):
                st.session_state.active_menu = "Riwayat Diagnosis"
                st.rerun()
        with act_col2:
            if st.button("🔄 Diagnosis Pasien Lain", type="primary", use_container_width=True):
                st.session_state.clinical_result = None
                st.session_state.clinical_form_key += 1
                st.rerun()


# ==========================================
# APLIKASI UTAMA
# ==========================================
def main_app():
    inject_custom_css()

    menu_options = [
        "Profil Dokter",
        "Diagnosis Foto Fundus",
        "Diagnosis Parameter Klinis",
        "Riwayat Diagnosis",
    ]

    with st.sidebar:
        st.title("Navigasi")
        st.write("Login sebagai: **dr. Sita**")
        st.divider()
        current_index = menu_options.index(st.session_state.active_menu)
        selected_menu = st.radio(
            "Pilih Menu",
            menu_options,
            index=current_index,
        )
        if selected_menu != st.session_state.active_menu:
            st.session_state.active_menu = selected_menu
            st.rerun()

        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    menu = st.session_state.active_menu

    if menu == "Profil Dokter":
        st.header("👨‍⚕️ Profil Dokter")
        st.divider()
        col1, col2 = st.columns([1, 2.5])
        with col1:
            render_pil_image(get_doctor_avatar(), width=200)
        with col2:
            st.subheader("dr. Sita")
            st.write("**Username:** `drsita`")
            st.write("**Bidang:** Analisis Citra Fundus & Parameter Klinis Atrofi Papil (Normal, Atrofi Total, Atrofi Parsial)")
            st.write("**Model Citra Fundus:** ResNet50 + Grad-CAM")
            st.write("**Model Parameter Klinis:** Random Forest + SHAP Scoring System")

    elif menu == "Diagnosis Foto Fundus":
        st.header("🔬 Diagnosis Foto Fundus")
        st.divider()

        model = load_fundus_model()
        if model is None:
            st.error(
                "Model belum dapat dimuat. Periksa kembali link release di GitHub Releases serta log aplikasi."
            )
            return

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            patient_id = st.text_input("Nomor Pasien ", placeholder="Contoh: Pasien-01")
        with col_in2:
            uploaded_file = st.file_uploader("Upload Foto Fundus", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            render_pil_image(pil_image, caption="Preview Foto Fundus", width=260)

            if st.button("Diagnosis Foto", type="primary"):
                with st.spinner("Gambar Anda sedang dianalisis..."):
                    input_tensor = preprocess_fundus_image(pil_image)
                    probs = model.predict(input_tensor, verbose=0)[0]
                    probs = np.asarray(probs, dtype=np.float32)

                    if len(probs) != len(CLASS_NAMES):
                        st.error(
                            f"Jumlah output model ({len(probs)}) tidak sama dengan jumlah CLASS_NAMES ({len(CLASS_NAMES)})."
                        )
                        return

                    pred_idx = int(np.argmax(probs))
                    pred_label = CLASS_NAMES[pred_idx]
                    confidence = float(probs[pred_idx] * 100.0)

                    heatmap = make_gradcam_heatmap(input_tensor, model, pred_index=pred_idx)
                    gradcam_image = overlay_gradcam(pil_image, heatmap)

                st.success("Analisis selesai.")
                st.subheader(f"Prediksi: **{pred_label}** ({confidence:.2f}%)")

                prob_cols = st.columns(len(CLASS_NAMES))
                for idx, cname in enumerate(CLASS_NAMES):
                    prob_cols[idx].metric(cname, f"{probs[idx] * 100:.2f}%")

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    render_pil_image(pil_image, caption="Gambar Fundus Asli")
                with img_col2:
                    render_pil_image(gradcam_image, caption="Grad-CAM Heatmap")

                st.session_state.history.append({
                    "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "jenis": "Foto Fundus",
                    "pasien": patient_id.strip() if patient_id.strip() else "Tanpa ID",
                    "file": uploaded_file.name,
                    "prediksi": pred_label,
                    "confidence": f"{confidence:.2f}%",
                    "asli_bytes": pil_to_png_bytes(pil_image),
                    "gradcam_bytes": pil_to_png_bytes(gradcam_image),
                })

    elif menu == "Diagnosis Parameter Klinis":
        clinical_diagnosis_page()

    elif menu == "Riwayat Diagnosis":
        st.header("🕒 Riwayat Diagnosis")
        st.divider()

        if not st.session_state.history:
            st.info("Belum ada riwayat diagnosis pada sesi ini.")
        else:
            if st.button("Bersihkan Riwayat"):
                st.session_state.history = []
                st.rerun()

            for i, item in enumerate(reversed(st.session_state.history), start=1):
                jenis = item.get("jenis", "Foto Fundus")
                title = f"{i}. [{item['waktu']}] [{jenis}] {item['pasien']} — {item['prediksi']} ({item['confidence']})"
                with st.expander(title):
                    if jenis == "Parameter Klinis":
                        st.markdown("**Ringkasan Parameter Klinis:**")
                        st.table(
                            pd.DataFrame(
                                list(item["display_summary"].items()),
                                columns=["Parameter", "Data"],
                            )
                        )
                        st.markdown("**Final Scoring Summary:**")
                        st.dataframe(
                            item["scoring_output"]["summary_df"].round(4),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            render_bytes_image(item["asli_bytes"], caption=f"Asli ({item['file']})")
                        with c2:
                            render_bytes_image(item["gradcam_bytes"], caption="Grad-CAM Heatmap")


if st.session_state.logged_in:
    main_app()
else:
    login_page()
