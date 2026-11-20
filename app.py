import io
import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

# ==========================================
# KONFIGURASI SESUAI TRAINING & GITHUB RELEASES
# ==========================================
MODEL_PATH = "best_resnet50_model.keras"
MODEL_RELEASE_URL = (
    "https://github.com/Alfizh05-git/fundus-diagnosis-bdc/releases/download/DL-Model/best_resnet50_model.keras"
)

# Samakan dengan IMG_SIZE = (height, width) pada script training Anda
IMG_SIZE = (224, 224)
# Samakan urutan indeks 0, 1, 2 dengan CLASS_TO_IDX / IDX_TO_CLASS saat training
CLASS_NAMES = ["Normal", "Atrofi Total", "Atrofi Parsial"]

st.set_page_config(
    page_title="Sistem Diagnosis Fundus - Dr. Sita",
    page_icon="👁️",
    layout="wide",
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# UTILITAS GAMBAR & MODEL
# ==========================================
def get_doctor_avatar() -> Image.Image:
    """Membuat avatar lokal sederhana agar tidak bergantung pada URL eksternal."""
    img = Image.new("RGB", (240, 240), color=(224, 242, 254))
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 35, 180, 155), fill=(30, 64, 175))
    draw.pieslice((30, 145, 210, 275), start=180, end=360, fill=(37, 99, 235))
    return img


def pil_to_png_bytes(img_pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    img_pil.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


<<<<<<< HEAD
def download_model_from_release(dest_path: str, release_url: str) -> bool:
    """Mengunduh file model langsung dari aset GitHub Releases."""
    clean_url = (release_url or "").strip()
    if not clean_url:
        st.error("`MODEL_RELEASE_URL` kosong pada kode `app.py`.")
        return False

    try:
        with requests.get(clean_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        st.error(f"Gagal mengunduh model dari GitHub Releases: {exc}")
        return False

    if not os.path.exists(dest_path):
        st.error("File model tidak ditemukan setelah proses unduh.")
        return False

    size_bytes = os.path.getsize(dest_path)
    if size_bytes < 1024 * 1024:
        st.error(
            f"Ukuran file model terlalu kecil ({size_bytes} bytes), kemungkinan file tidak terunduh penuh."
        )
=======
import requests

def _download_gdrive_fallback_requests(dest_path: str, file_id: str) -> bool:
    """Cadangan unduhan file besar Google Drive dengan konfirmasi token bila diperlukan."""
    session = requests.Session()
    base_url = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}

    resp = session.get(base_url, params=params, stream=True, timeout=120)
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        params["confirm"] = token
        resp.close()
        resp = session.get(base_url, params=params, stream=True, timeout=120)

    if resp.status_code != 200:
        st.warning(f"Unduhan Google Drive mengembalikan status HTTP {resp.status_code}.")
        return False

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    resp.close()
    return os.path.exists(dest_path)


def download_model_from_gdrive(dest_path: str, file_id: str) -> bool:
    """Mengunduh file model dari Google Drive menggunakan gdown lalu fallback ke requests."""
    clean_id = (file_id or "").strip()
    if not clean_id or "MASUKKAN_FILE_ID" in clean_id:
        st.error("`GDRIVE_FILE_ID` belum diisi pada kode `app.py`.")
        return False

    url = f"https://drive.google.com/uc?id={clean_id}"
    downloaded_ok = False

    try:
        output = gdown.download(url, dest_path, quiet=False, fuzzy=True)
        downloaded_ok = output is not None and os.path.exists(dest_path)
    except Exception as exc:
        st.warning(f"`gdown` gagal: {exc}. Mencoba metode unduhan cadangan...")

    if not downloaded_ok or (os.path.exists(dest_path) and os.path.getsize(dest_path) < 1024 * 1024):
        try:
            downloaded_ok = _download_gdrive_fallback_requests(dest_path, clean_id)
        except Exception as exc:
            st.error(f"Unduhan cadangan juga gagal: {exc}")
            return False

    if not os.path.exists(dest_path):
        st.error("File model tidak berhasil dibuat setelah proses unduh.")
        return False

    size_bytes = os.path.getsize(dest_path)
    if size_bytes < 1024 * 1024:
        with open(dest_path, "rb") as f:
            preview = f.read(300).decode("utf-8", errors="replace")
        st.error(
            f"File yang terunduh terlalu kecil ({size_bytes} bytes), kemungkinan bukan file model `.keras` "
            "(misalnya halaman izin akses/konfirmasi Google Drive). Cek apakah file di Google Drive sudah diatur ke "
            "**Anyone with the link**."
        )
        st.code(preview, language="text")
>>>>>>> b04ea1eefa988d72290d73a2284814cc32ccf8cb
        return False

    return True


@st.cache_resource
def load_fundus_model():
    need_download = (
        not os.path.exists(MODEL_PATH)
        or os.path.getsize(MODEL_PATH) < 1024 * 1024
    )
    if need_download:
        with st.spinner("Mengunduh model dari GitHub Releases, mohon tunggu..."):
            downloaded = download_model_from_release(MODEL_PATH, MODEL_RELEASE_URL)
            if not downloaded:
                return None

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        st.error(f"Gagal memuat model `{MODEL_PATH}`: {exc}")
        return None


def preprocess_fundus_image(img_pil: Image.Image) -> np.ndarray:
    """Menyiapkan tensor input (1, H, W, 3) float32; preprocessing ResNet50 ada di dalam model."""
    img_resized = img_pil.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype(np.float32)
    return np.expand_dims(img_array, axis=0)


def _find_last_conv_layer_recursive(layer):
    """Mencari layer Conv2D terakhir baik di model utama maupun sub-model."""
    if isinstance(layer, tf.keras.layers.Conv2D):
        return layer
    if hasattr(layer, "layers"):
        for sub_layer in reversed(layer.layers):
            found = _find_last_conv_layer_recursive(sub_layer)
            if found is not None:
                return found
    return None


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, pred_index: int) -> np.ndarray:
    """Menghasilkan heatmap Grad-CAM dengan menelusuri sub-model ResNet50 secara aman."""
    try:
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
                backbone = layer
                break

        if backbone is not None:
            last_conv = _find_last_conv_layer_recursive(backbone)
            if last_conv is None:
                return np.zeros((7, 7), dtype=np.float32)

            conv_extractor = tf.keras.Model(
                inputs=backbone.input,
                outputs=last_conv.output,
            )

            post_layers = []
            found_backbone = False
            for layer in model.layers:
                if found_backbone:
                    post_layers.append(layer)
                elif layer is backbone:
                    found_backbone = True

            pre_layers = []
            for layer in model.layers:
                if layer is backbone:
                    break
                if not isinstance(layer, tf.keras.layers.InputLayer):
                    pre_layers.append(layer)

            with tf.GradientTape() as tape:
                x = tf.cast(img_array, tf.float32)
                for layer in pre_layers:
                    x = layer(x, training=False)
                conv_outputs = conv_extractor(x, training=False)
                tape.watch(conv_outputs)

                backbone_out = conv_outputs
                passed_conv = False
                for sub_layer in backbone.layers:
                    if passed_conv:
                        backbone_out = sub_layer(backbone_out, training=False)
                    elif sub_layer is last_conv:
                        passed_conv = True

                y = backbone_out
                for layer in post_layers:
                    y = layer(y, training=False)

                class_channel = y[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                return np.zeros((7, 7), dtype=np.float32)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_map = conv_outputs[0]
            heatmap = conv_map @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
        else:
            last_conv = _find_last_conv_layer_recursive(model)
            if last_conv is None:
                return np.zeros((7, 7), dtype=np.float32)

            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[last_conv.output, model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array, training=False)
                tape.watch(conv_outputs)
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                return np.zeros((7, 7), dtype=np.float32)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_map = conv_outputs[0]
            heatmap = conv_map @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy().astype(np.float32)

    except Exception:
        return np.zeros((7, 7), dtype=np.float32)


def overlay_gradcam(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Menggabungkan heatmap Grad-CAM dengan gambar fundus asli."""
    img = np.array(img_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img, 1.0 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(superimposed)


# ==========================================
# HALAMAN LOGIN
# ==========================================
def login_page():
    _, col_mid, _ = st.columns([1, 1.4, 1])
    with col_mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("👁️ Portal Diagnosis Fundus")
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
# APLIKASI UTAMA
# ==========================================
def main_app():
    with st.sidebar:
        st.title("Navigasi")
        st.write("Login sebagai: **dr. Sita**")
        st.divider()
        menu = st.radio(
            "Pilih Menu",
            ["Profil Dokter", "Diagnosis", "Riwayat Diagnosis"],
        )
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    if menu == "Profil Dokter":
        st.header("👨‍⚕️ Profil Dokter")
        st.divider()
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.image(get_doctor_avatar(), width=200)
        with col2:
            st.subheader("dr. Sita")
            st.write("**Username:** `drsita`")
            st.write("**Bidang:** Analisis Citra Fundus (Normal, Atrofi Total, Atrofi Parsial)")
            st.write("**Arsitektur Model:** ResNet50 + Grad-CAM")

    elif menu == "Diagnosis":
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
            patient_id = st.text_input("ID / Nama Pasien (Opsional)", placeholder="Contoh: Pasien-01")
        with col_in2:
            uploaded_file = st.file_uploader("Upload Foto Fundus", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="Preview Foto Fundus", width=260)

            if st.button("diagnosis foto", type="primary"):
                with st.spinner("gambar anda sedang dianalisis"):
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
                    st.image(pil_image, caption="Gambar Fundus Asli", use_container_width=True)
                with img_col2:
                    st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)

                st.session_state.history.append({
                    "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pasien": patient_id.strip() if patient_id.strip() else "Tanpa ID",
                    "file": uploaded_file.name,
                    "prediksi": pred_label,
                    "confidence": f"{confidence:.2f}%",
                    "asli_bytes": pil_to_png_bytes(pil_image),
                    "gradcam_bytes": pil_to_png_bytes(gradcam_image),
                })

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
                title = f"{i}. [{item['waktu']}] {item['pasien']} — {item['prediksi']} ({item['confidence']})"
                with st.expander(title):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(item["asli_bytes"], caption=f"Asli ({item['file']})", use_container_width=True)
                    with c2:
                        st.image(item["gradcam_bytes"], caption="Grad-CAM Heatmap", use_container_width=True)


if st.session_state.logged_in:
    main_app()
else:
    login_page()
