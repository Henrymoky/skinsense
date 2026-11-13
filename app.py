"""SkinSense: web app deteksi jenis kulit wajah (custom CNN) + rekomendasi skincare.

Jalankan lokal:
    pip install -r requirements.txt
    python app.py            # buka http://localhost:5000

File model (hasil notebook, folder PA_outputs/model/) harus ada di folder ./model/:
    model/skin_type_cnn.keras
    model/labels.json
"""
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "model"))
MAX_UPLOAD_MB = 10
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOW_CONFIDENCE = 0.50

RECOMMENDATIONS = {
    "oily": dict(
        judul="Kulit Berminyak",
        ciri="Produksi sebum tinggi, wajah tampak mengkilap terutama di zona T (dahi, hidung, dagu), pori-pori besar, dan rentan komedo/jerawat.",
        rutin=["Gentle foaming atau gel cleanser 2x sehari",
               "Exfoliant BHA (salicylic acid) 2-3x seminggu",
               "Serum niacinamide untuk mengontrol minyak dan pori",
               "Moisturizer gel oil-free, non-comedogenic",
               "Sunscreen bertekstur gel/fluid dengan hasil matte"],
        hindari=["Produk berbahan minyak berat atau oklusif tebal",
                 "Scrub kasar berlebihan",
                 "Mencuci muka terlalu sering (memicu minyak berlebih)"]),
    "dry": dict(
        judul="Kulit Kering",
        ciri="Terasa kencang setelah cuci muka, tampak kusam, bersisik atau mengelupas, dan garis halus lebih terlihat.",
        rutin=["Cleanser creamy atau low-pH tanpa SLS",
               "Hydrating toner/essence dengan hyaluronic acid",
               "Serum ceramide atau panthenol untuk memperkuat skin barrier",
               "Moisturizer krim yang kaya (ceramide, shea butter, squalane)",
               "Sunscreen yang melembapkan"],
        hindari=["Toner dengan kandungan alkohol tinggi",
                 "Cleanser berbusa kuat",
                 "Eksfoliasi terlalu sering",
                 "Mencuci muka dengan air terlalu panas"]),
    "normal": dict(
        judul="Kulit Normal",
        ciri="Seimbang, tidak terlalu berminyak maupun kering, pori-pori kecil, dan jarang bermasalah.",
        rutin=["Gentle cleanser",
               "Moisturizer ringan (lotion)",
               "Antioksidan seperti vitamin C di pagi hari",
               "Sunscreen SPF 30+ setiap hari",
               "Eksfoliasi ringan AHA 1x seminggu (opsional)"],
        hindari=["Mencoba banyak produk baru sekaligus",
                 "Melewatkan sunscreen"]),
    "combination": dict(
        judul="Kulit Kombinasi",
        ciri="Zona T berminyak, sedangkan area pipi normal atau kering.",
        rutin=["Gentle gel cleanser", "Serum niacinamide",
               "Moisturizer ringan; krim lebih kaya di area kering", "Sunscreen ringan"],
        hindari=["Produk terlalu berat di zona T"]),
    "sensitive": dict(
        judul="Kulit Sensitif",
        ciri="Mudah kemerahan, perih, atau iritasi.",
        rutin=["Cleanser fragrance-free", "Moisturizer dengan centella/ceramide", "Sunscreen mineral (zinc oxide)"],
        hindari=["Pewangi, alkohol, dan eksfoliasi keras", "Langsung memakai produk baru tanpa patch test"]),
}


def recommendation_for(label):
    return RECOMMENDATIONS.get(label, dict(judul=label.title(), ciri="-", rutin=[], hindari=[]))


class KerasPredictor:
    """Memuat model .keras dan melakukan prapemrosesan yang SAMA dengan notebook (load_image)."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        model_path, meta_path = model_dir / "skin_type_cnn.keras", model_dir / "labels.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan di '{model_dir}'. Salin skin_type_cnn.keras dan labels.json "
                "dari PA_outputs/model/ hasil notebook.")
        import tensorflow as tf  # import di sini agar server cepat menyala
        import keras
        self.tf = tf
        self.meta = json.loads(meta_path.read_text())
        self.classes = list(self.meta["class_names"])
        self.img_size = int(self.meta["img_size"])
        self.model = keras.models.load_model(model_path, compile=False)
        self.n_params = int(self.model.count_params())
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), np.float32), verbose=0)  # warm-up

    def preprocess(self, img):
        # RGB -> resize bilinear antialias -> bulatkan ke 0..255 -> float32 (normalisasi 1/255 ada DI DALAM model)
        x = self.tf.image.resize(np.asarray(img.convert("RGB")), [self.img_size, self.img_size], antialias=True)
        return self.tf.clip_by_value(self.tf.round(x), 0, 255).numpy().astype("float32")

    def predict(self, img, tta=True):
        x = self.preprocess(img)[None]
        if tta:  # test-time augmentation: rata-rata prediksi citra asli dan citra yang di-flip horizontal
            batch = np.concatenate([x, np.ascontiguousarray(x[:, :, ::-1, :])])
            return self.model.predict(batch, verbose=0).mean(axis=0)
        return self.model.predict(x, verbose=0)[0]


def create_app(predictor=None, model_dir=MODEL_DIR):
    app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    state = {"predictor": predictor, "error": None}
    lock = threading.Lock()

    def get_predictor():
        if state["predictor"] is None and state["error"] is None:
            with lock:
                if state["predictor"] is None and state["error"] is None:
                    try:
                        state["predictor"] = KerasPredictor(model_dir)
                    except Exception as e:  # pesan error ditampilkan di UI
                        state["error"] = f"{type(e).__name__}: {e}"
                        app.logger.error("Gagal memuat model: %s", state["error"])
        return state["predictor"]

    def error(msg, code):
        return jsonify(ok=False, error=msg), code

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/info")
    def info():
        p = get_predictor()
        if p is None:
            return jsonify(ready=False, error=state["error"])
        m = getattr(p, "meta", {})
        return jsonify(
            ready=True,
            classes=[dict(label=c, judul=recommendation_for(c)["judul"]) for c in p.classes],
            img_size=p.img_size,
            params=getattr(p, "n_params", None),
            test_accuracy=m.get("test_accuracy"),
            test_accuracy_tta=m.get("test_accuracy_tta"),
            dataset=m.get("dataset"),
            created=m.get("created"),
        )

    @app.post("/api/predict")
    def predict():
        p = get_predictor()
        if p is None:
            return error(f"Model belum siap. {state['error'] or ''}".strip(), 503)
        f = request.files.get("image")
        if f is None or not f.filename:
            return error("Tidak ada gambar yang dikirim.", 400)
        ext = Path(f.filename).suffix.lower()
        if ext and ext not in ALLOWED_EXT:
            return error("Format tidak didukung. Gunakan JPG, PNG, atau WEBP.", 400)
        try:
            img = Image.open(io.BytesIO(f.read()))
            img = ImageOps.exif_transpose(img)
            img.load()
        except (UnidentifiedImageError, OSError):
            return error("File bukan gambar yang valid atau rusak.", 400)
        if min(img.size) < 64:
            return error("Resolusi gambar terlalu kecil (minimal 64 px).", 400)

        tta = request.form.get("tta", "1") not in ("0", "false", "False")
        t0 = time.perf_counter()
        probs = np.asarray(p.predict(img, tta=tta), dtype=float)
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(-probs)
        top = p.classes[int(order[0])]
        conf = float(probs[order[0]])
        return jsonify(
            ok=True,
            label=top,
            judul=recommendation_for(top)["judul"],
            confidence=conf,
            low_confidence=conf < LOW_CONFIDENCE,
            probabilities=[dict(label=p.classes[int(i)], judul=recommendation_for(p.classes[int(i)])["judul"],
                                prob=float(probs[i])) for i in order],
            recommendation=recommendation_for(top),
            tta=tta,
            inference_ms=round(ms, 1),
            image_size=list(img.size),
        )

    @app.errorhandler(413)
    def too_large(_):
        return error(f"Ukuran file terlalu besar (maksimal {MAX_UPLOAD_MB} MB).", 413)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"SkinSense berjalan di http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
