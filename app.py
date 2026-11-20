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


import traceback


def download_model_from_release(dest_path: str, release_url: str) -> bool:
    """Mengunduh file model langsung dari aset GitHub Releases dengan log status & error detail."""
    clean_url = (release_url or "").strip()
    if not clean_url:
        msg = "`MODEL_RELEASE_URL` kosong pada kode `app.py`."
        print(f"[MODEL DOWNLOAD ERROR] {msg}", flush=True)
        st.error(msg)
        return False

    print(f"[MODEL DOWNLOAD] Mengunduh dari: {clean_url} -> {dest_path}", flush=True)
    status_placeholder = st.empty()
    status_placeholder.info(f"Menghubungkan ke GitHub Releases: `{clean_url}`")

    try:
        with requests.get(clean_url, stream=True, timeout=300) as resp:
            print(f"[MODEL DOWNLOAD] HTTP Status: {resp.status_code}", flush=True)
            total_bytes = int(resp.headers.get("Content-Length", 0))
            if total_bytes > 0:
                print(
                    f"[MODEL DOWNLOAD] Ukuran file di server: {total_bytes / (1024 * 1024):.2f} MB",
                    flush=True,
                )

            if resp.status_code != 200:
                body_preview = resp.text[:300] if resp.text else ""
                err_msg = (
                    f"Gagal mengunduh model (HTTP {resp.status_code}). "
                    f"Pastikan release & aset `best_resnet50_model.keras` tersedia publik."
                )
                print(f"[MODEL DOWNLOAD ERROR] {err_msg} | Preview: {body_preview}", flush=True)
                status_placeholder.error(err_msg)
                if body_preview:
                    st.code(body_preview, language="text")
                return False

            downloaded_bytes = 0
            last_logged_mb = 0
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        current_mb = downloaded_bytes // (1024 * 1024)
                        if current_mb >= last_logged_mb + 50:
                            last_logged_mb = current_mb
                            print(
                                f"[MODEL DOWNLOAD] Terunduh: {downloaded_bytes / (1024 * 1024):.1f} MB",
                                flush=True,
                            )
                            status_placeholder.info(
                                f"Mengunduh model: {downloaded_bytes / (1024 * 1024):.1f} MB..."
                            )

    except Exception as exc:
        tb_str = traceback.format_exc()
        print(f"[MODEL DOWNLOAD EXCEPTION] {exc}\n{tb_str}", flush=True)
        st.error(f"Terjadi exception saat mengunduh model: {exc}")
        st.exception(exc)
        return False

    if not os.path.exists(dest_path):
        msg = f"File `{dest_path}` tidak ditemukan di direktori kerja ({os.getcwd()}) setelah unduh."
        print(f"[MODEL DOWNLOAD ERROR] {msg}", flush=True)
        st.error(msg)
        return False

    size_bytes = os.path.getsize(dest_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"[MODEL DOWNLOAD] Selesai. Ukuran file lokal: {size_mb:.2f} MB ({size_bytes} bytes)", flush=True)

    if size_bytes < 1024 * 1024:
        with open(dest_path, "rb") as f:
            preview = f.read(300).decode("utf-8", errors="replace")
        err_msg = (
            f"Ukuran file model terlalu kecil ({size_bytes} bytes), kemungkinan bukan file `.keras` yang valid."
        )
        print(f"[MODEL DOWNLOAD ERROR] {err_msg} | Preview: {preview}", flush=True)
        status_placeholder.error(err_msg)
        st.code(preview, language="text")
        return False

    status_placeholder.success(f"Model berhasil diunduh ({size_mb:.2f} MB).")
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

    # Guard tambahan agar load_model() tidak dipanggil jika file tetap tidak ada atau korup/terpotong
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024 * 1024:
        st.error(f"File model `{MODEL_PATH}` tidak tersedia atau tidak lengkap di disk.")
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