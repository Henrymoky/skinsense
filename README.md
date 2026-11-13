# SkinSense: Klasifikasi Jenis Kulit Wajah dengan Custom CNN

Proyek Akhir Praktikum Kecerdasan Buatan. Henry Moky (2409106070), Informatika Universitas Mulawarman.

Web app yang mengidentifikasi **jenis kulit wajah** (berminyak / kering / normal) dari foto menggunakan
**custom Convolutional Neural Network** yang dilatih dari nol (tanpa transfer learning), lalu memberi rekomendasi skincare.

- **Dataset:** [Oily, Dry and Normal Skin Types Dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset) (Kaggle, asal Roboflow)
- **Notebook training:** `PA_KB_Klasifikasi_Jenis_Kulit_CNN.ipynb` (pengumpulan data, pembersihan, EDA, prapemrosesan, model, evaluasi, deployment)
- **Model:** Residual CNN 4 blok (±1,3 juta parameter), input 224×224, Adam + label smoothing, EarlyStopping & ReduceLROnPlateau

## Struktur
```
app.py              backend Flask (API /api/predict, /api/info)
static/index.html   tampilan web (upload / kamera, hasil, rekomendasi)
model/              skin_type_cnn.keras + labels.json (hasil notebook)
Dockerfile          untuk deploy (Hugging Face Spaces / Render)
```

## Menjalankan
```bash
pip install -r requirements.txt
python app.py        # buka http://localhost:5000
```
