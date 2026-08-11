import io
import json
import re
import os
import subprocess
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

SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "aac", "webm"]

AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}


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
# KONVERSI AUDIO
# ============================================================
def convert_to_standard_wav(input_path):
    """
    Konversi format audio umum ke WAV mono 16 kHz menggunakan ffmpeg.
    Ini membuat MP3/M4A/FLAC/OGG/AAC/WEBM diproses dengan jalur
    preprocessing yang sama seperti WAV.
    """
    input_path = str(input_path)
    output_path = f"{input_path}.converted.wav"

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        output_path,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg belum tersedia. Pastikan file packages.txt berisi `ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Format audio gagal dikonversi oleh ffmpeg. {detail}"
        ) from exc

    return output_path


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
        return "Pertahankan dengungan ghunnah agar tetap jelas dan konsisten."

    return "Latih kembali kejelasan dan kestabilan dengungan ghunnah, lalu rekam ulang dengan suara yang jelas."


def get_hf_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")


def keep_one_feedback(text):
    """Pastikan output akhir hanya satu feedback."""
    if not text:
        return text

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return str(text).strip()

    first = lines[0]

    # Hapus awalan bullet / nomor jika model masih mengeluarkannya.
    first = re.sub(r"^\s*(?:[-*•]+|\d+[\.\)])\s*", "", first).strip()

    # Jika masih berupa beberapa kalimat, tampilkan kalimat pertama saja.
    match = re.match(r"(.+?[.!?])(?:\s|$)", first)
    if match:
        first = match.group(1).strip()

    return first


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
        "Berikan tepat SATU feedback singkat dalam Bahasa Indonesia. "
        "Jangan mengarang jenis kesalahan yang tidak diberikan oleh model. "
        "CNN hanya menghasilkan label BENAR atau SALAH. "
        "Jangan menyatakan hasil sebagai diagnosis atau keputusan guru. "
        "Gunakan bahasa yang jelas, sopan, dan praktis. Jangan gunakan bullet, daftar, atau penomoran."
    )

    user_prompt = f"""
Hasil CNN:
- Label: {result["label_name"]}
- Probabilitas SALAH: {result["prob_salah"]:.4f}
- Threshold klasifikasi: {CLASSIFICATION_THRESHOLD:.2f}

Contoh gaya feedback dari metadata dataset:
{references}

Buat tepat SATU feedback saja dalam SATU kalimat.
Jangan gunakan bullet, daftar, atau penomoran.
Jika label SALAH, berikan satu saran latihan ghunnah secara umum tanpa menebak detail kesalahan lain.
Jika label BENAR, berikan satu penguatan singkat tanpa menyatakan bacaan pasti sempurna.
""".strip()

    client = InferenceClient(
        model=QWEN_MODEL,
        provider="featherless-ai",
        token=token,
        timeout=120,
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

    return keep_one_feedback(output.choices[0].message.content.strip())


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
- **Input:** WAV, MP3, M4A, FLAC, OGG, AAC, atau WEBM
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
    "Upload rekaman bacaan",
    type=SUPPORTED_AUDIO_TYPES,
    help="Format: WAV, MP3, M4A, FLAC, OGG, AAC, atau WEBM. Gunakan rekaman yang cukup jelas dan minim noise.",
)

if uploaded_file is None:
    st.info("Upload rekaman audio untuk mulai melakukan prediksi.")
    st.stop()

audio_bytes = uploaded_file.getvalue()
uploaded_suffix = Path(uploaded_file.name).suffix.lower()
audio_mime = AUDIO_MIME_TYPES.get(uploaded_suffix, "audio/wav")
st.audio(audio_bytes, format=audio_mime)

if st.button("Analisis bacaan", type="primary", use_container_width=True):
    try:
        with st.spinner("Memproses audio dan menjalankan CNN..."):
            suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
            converted_path = None

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                # Semua format distandarkan menjadi WAV mono 16 kHz
                # sebelum masuk ke preprocessing CNN.
                converted_path = convert_to_standard_wav(tmp_path)

                model = load_cnn_model()
                metadata = load_metadata()
                result = predict_audio(model, converted_path)
            finally:
                for path_to_remove in [tmp_path, converted_path]:
                    if path_to_remove:
                        try:
                            os.remove(path_to_remove)
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
            f'Durasi setelah trimming = {result["trimmed_duration"]:.2f} detik • '
            f'CNN menggunakan maksimal 501 frame (~8 detik awal)'
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
                    "Qwen API gagal, sehingga feedback lokal digunakan. "
                    "Lihat detail error di bawah."
                )
                st.code(f"{type(exc).__name__}: {exc}")

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
