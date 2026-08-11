import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


# ============================================================
# KONFIGURASI — disamakan dengan CNN Improved-v1 8 detik
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
WEIGHTS_PATH = APP_DIR / "model.weights.h5"
METADATA_PATH = APP_DIR / "metadata.csv"

SAMPLE_RATE = 16000
SILENCE_TOP_DB = 30
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
MAX_FRAMES = 501

# Threshold terbaik seed 42 Improved-v1 8 detik,
# dipilih hanya dari validation set.
CLASSIFICATION_THRESHOLD = 0.51

LABELS = {
    0: "BENAR",
    1: "SALAH",
}

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ============================================================
# LOAD MODEL / DATA
# ============================================================
@st.cache_resource
def load_cnn_model():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # config.json berasal dari format serialisasi Keras.
    model = keras.models.model_from_json(json.dumps(model_config))
    model.load_weights(WEIGHTS_PATH)
    return model


@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_PATH)


# ============================================================
# PREPROCESSING
# ============================================================
def load_audio_trimmed(audio_path):
    original_audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    trimmed_audio, _ = librosa.effects.trim(
        original_audio,
        top_db=SILENCE_TOP_DB,
    )

    # Sama dengan notebook: jika hasil trimming kosong,
    # gunakan audio asli.
    if trimmed_audio.size == 0:
        trimmed_audio = original_audio

    return trimmed_audio.astype(np.float32)


def pad_or_crop_spectrogram(logmel, max_frames=MAX_FRAMES):
    n_frames = logmel.shape[1]

    if n_frames > max_frames:
        # Improved-v1 8s: ambil bagian AWAL spectrogram.
        logmel = logmel[:, :max_frames]

    elif n_frames < max_frames:
        right_padding = max_frames - n_frames

        # Sama dengan notebook: padding memakai nilai minimum
        # spectrogram yang sudah dinormalisasi.
        pad_value = float(logmel.min())

        logmel = np.pad(
            logmel,
            ((0, 0), (0, right_padding)),
            mode="constant",
            constant_values=pad_value,
        )

    return logmel


def extract_logmel(y):
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )

    logmel = librosa.power_to_db(
        mel_power,
        ref=np.max,
    )

    # Z-score per audio SEBELUM crop/padding.
    logmel_mean = float(logmel.mean())
    logmel_std = float(logmel.std())
    logmel = (logmel - logmel_mean) / (logmel_std + 1e-8)

    logmel = pad_or_crop_spectrogram(logmel)
    return logmel.astype(np.float32)


def preprocess_audio(audio_path):
    y = load_audio_trimmed(audio_path)
    logmel = extract_logmel(y)

    # (64, 501) -> (1, 64, 501, 1)
    x = np.expand_dims(logmel, axis=-1)
    x = np.expand_dims(x, axis=0).astype(np.float32)

    return x, y, logmel


# ============================================================
# INFERENCE
# ============================================================
def predict_audio(model, audio_path):
    x, y_trimmed, logmel = preprocess_audio(audio_path)

    prob_salah = float(
        np.asarray(model(x, training=False)).reshape(-1)[0]
    )

    pred_label = int(prob_salah >= CLASSIFICATION_THRESHOLD)

    return {
        "label": pred_label,
        "label_name": LABELS[pred_label],
        "prob_salah": prob_salah,
        "prob_benar": 1.0 - prob_salah,
        "trimmed_duration": len(y_trimmed) / SAMPLE_RATE,
        "logmel": logmel,
    }


# ============================================================
# FEEDBACK
# ============================================================
def fallback_feedback(result):
    if result["label"] == 0:
        return (
            "Bacaan terdeteksi **BENAR** oleh model CNN. "
            "Pertahankan dengungan ghunnah agar tetap jelas dan konsisten. "
            "Hasil ini adalah prediksi model dan bukan penilaian tajwid dari guru secara langsung."
        )

    return (
        "Bacaan terdeteksi **SALAH** oleh model CNN. "
        "Fokuskan latihan pada kejelasan dan kestabilan dengungan ghunnah, "
        "lalu rekam ulang dengan suara yang jelas. "
        "Model CNN ini hanya menentukan BENAR/SALAH dan tidak menentukan jenis kesalahan yang lebih spesifik."
    )


def get_hf_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")


def qwen_feedback(result, metadata):
    token = get_hf_token()

    if not token or InferenceClient is None:
        return None

    if result["label"] == 0:
        examples = (
            metadata.loc[metadata["label"] == 0, "error_explanation"]
            .dropna()
            .drop_duplicates()
            .head(3)
            .tolist()
        )
    else:
        examples = (
            metadata.loc[metadata["label"] == 1, "error_explanation"]
            .dropna()
            .drop_duplicates()
            .head(5)
            .tolist()
        )

    references = "\n".join(f"- {x}" for x in examples)

    system_prompt = (
        "Kamu adalah asisten latihan bacaan ghunnah. "
        "Berikan feedback singkat dalam Bahasa Indonesia. "
        "Jangan mengarang jenis kesalahan yang tidak diberikan oleh model. "
        "CNN hanya menghasilkan label BENAR atau SALAH. "
        "Jangan menyatakan hasil sebagai diagnosis atau keputusan guru. "
        "Gunakan bahasa yang jelas, sopan, dan praktis."
    )

    user_prompt = f"""
Hasil CNN:
- Label: {result["label_name"]}
- Probabilitas SALAH: {result["prob_salah"]:.4f}
- Threshold klasifikasi: {CLASSIFICATION_THRESHOLD:.2f}

Contoh gaya feedback dari metadata dataset:
{references}

Buat feedback 2–3 kalimat untuk pengguna.
Jika label SALAH, sarankan latihan ghunnah secara umum tanpa menebak detail kesalahan lain.
Jika label BENAR, beri penguatan singkat tanpa menyatakan bacaan pasti sempurna.
""".strip()

    client = InferenceClient(
        model=QWEN_MODEL,
        provider="auto",
        token=token,
        timeout=60,
    )

    output = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=180,
        temperature=0.2,
        top_p=0.9,
    )

    return output.choices[0].message.content.strip()


# ============================================================
# UI
# ============================================================
st.set_page_config(
    page_title="Deteksi Ghunnah",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Deteksi Ghunnah")
st.caption(
    "CNN Improved-v1 8 detik • Log-Mel Spectrogram • "
    "threshold validation 0.51"
)

with st.expander("Tentang model"):
    st.markdown(
        """
- **Input:** audio WAV
- **Sample rate:** 16 kHz
- **Silence trimming:** `top_db=30`
- **Fitur:** Log-Mel Spectrogram, 64 Mel bands
- **FFT / hop:** 1024 / 256
- **Normalisasi:** Z-score per audio
- **Panjang input CNN:** 501 frame
- **Kelas:** `0 = BENAR`, `1 = SALAH`
- **Threshold:** `0.51`
- **Catatan:** SpecAugment hanya digunakan saat training, bukan saat inference.
        """
    )

uploaded_file = st.file_uploader(
    "Upload rekaman bacaan (.wav)",
    type=["wav"],
    help="Gunakan rekaman yang cukup jelas dan minim noise.",
)

if uploaded_file is None:
    st.info("Upload file WAV untuk mulai melakukan prediksi.")
    st.stop()

audio_bytes = uploaded_file.getvalue()
st.audio(audio_bytes, format="audio/wav")

if st.button("Analisis bacaan", type="primary", use_container_width=True):
    try:
        with st.spinner("Memproses audio dan menjalankan CNN..."):
            suffix = Path(uploaded_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                model = load_cnn_model()
                metadata = load_metadata()
                result = predict_audio(model, tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if result["label"] == 0:
            st.success("Hasil: BENAR")
        else:
            st.error("Hasil: SALAH")

        col1, col2 = st.columns(2)
        col1.metric(
            "Probabilitas BENAR",
            f'{result["prob_benar"] * 100:.2f}%',
        )
        col2.metric(
            "Probabilitas SALAH",
            f'{result["prob_salah"] * 100:.2f}%',
        )

        st.progress(
            min(max(result["prob_salah"], 0.0), 1.0),
            text=f'P(SALAH) = {result["prob_salah"]:.4f}',
        )

        st.caption(
            f'Threshold = {CLASSIFICATION_THRESHOLD:.2f} • '
            f'Durasi setelah trimming = {result["trimmed_duration"]:.2f} detik'
        )

        st.subheader("Feedback")

        feedback = None
        hf_token = get_hf_token()

        if hf_token:
            try:
                with st.spinner("Membuat feedback dengan Qwen2.5-1.5B..."):
                    feedback = qwen_feedback(result, metadata)
            except Exception as exc:
                st.warning(
                    "Qwen API tidak dapat digunakan saat ini. "
                    "Feedback lokal ditampilkan sebagai cadangan."
                )

        if not feedback:
            feedback = fallback_feedback(result)

        st.markdown(feedback)

        if not hf_token:
            st.caption(
                "Qwen belum aktif. Tambahkan `HF_TOKEN` pada "
                "Streamlit Community Cloud → App settings → Secrets "
                "untuk mengaktifkan feedback SLM."
            )

        with st.expander("Detail teknis"):
            st.write(
                {
                    "label_numeric": result["label"],
                    "label": result["label_name"],
                    "probability_error": round(result["prob_salah"], 6),
                    "threshold": CLASSIFICATION_THRESHOLD,
                    "input_shape": [1, N_MELS, MAX_FRAMES, 1],
                    "sample_rate": SAMPLE_RATE,
                }
            )

    except Exception as exc:
        st.error("Prediksi gagal.")
        st.exception(exc)
