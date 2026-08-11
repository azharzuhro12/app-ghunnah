# Deteksi Ghunnah — Streamlit Community Cloud

Aplikasi inference untuk CNN Improved-v1 8 detik.

## Isi repository

```text
.
├── app.py
├── requirements.txt
├── config.json
├── metadata.json
├── model.weights.h5
├── metadata.csv
└── .streamlit/
    └── config.toml
```

## Konfigurasi model

- Sample rate: 16 kHz
- Silence trimming: `top_db=30`
- Log-Mel Spectrogram
- `n_mels=64`
- `n_fft=1024`
- `hop_length=256`
- Z-score per audio sebelum crop/padding
- Crop dari awal / padding di akhir
- 501 time frames
- Input CNN: `(64, 501, 1)`
- Output sigmoid = probabilitas kelas SALAH
- Label `0 = BENAR`, `1 = SALAH`
- Threshold final = `0.51`

SpecAugment tidak digunakan saat inference.

## Menjalankan lokal

Disarankan memakai Python 3.12.

```bash
python -m venv .venv
```

Aktifkan virtual environment, lalu:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload seluruh isi folder ini ke repository.
3. Buka Streamlit Community Cloud.
4. Pilih **Create app**.
5. Pilih repository dan branch GitHub.
6. Main file path: `app.py`.
7. Di **Advanced settings**, pilih **Python 3.12**.
8. Deploy.

## Mengaktifkan feedback Qwen

Aplikasi tetap dapat berjalan tanpa Qwen. Jika `HF_TOKEN` tidak tersedia,
feedback lokal akan dipakai.

Untuk mengaktifkan Qwen2.5-1.5B-Instruct melalui Hugging Face Inference Providers,
tambahkan secret pada Streamlit:

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
```

Masukkan token melalui:

**App settings → Secrets**

Jangan commit token ke GitHub.

## Catatan metodologi

CNN ini hanya melakukan klasifikasi biner BENAR/SALAH. Ia tidak dilatih untuk
menentukan tipe kesalahan tajwid yang lebih rinci. Karena itu aplikasi tidak
mengarang `error_type` untuk rekaman baru.

`metadata.csv` digunakan sebagai sumber contoh gaya feedback, bukan sebagai
ground truth untuk rekaman baru yang diunggah pengguna.
