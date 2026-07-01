# app.py — Flask Web Server untuk Klasifikasi Kualitas Alpukat
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from avocado_engine import predict_image
import cv2
import numpy as np

app = Flask(__name__)
# Mengizinkan akses Cross-Origin agar JavaScript dari frontend bisa menembak API
CORS(app)  

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"status": "error", "error_msg": "Tidak ada file yang dikirim."}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"status": "error", "error_msg": "Nama file kosong."}), 400
        
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"status": "error", "error_msg": f"Format .{ext} tidak didukung."}), 400

    try:
        # Mengubah file input gambar menjadi matriks OpenCV array BGR
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"status": "error", "error_msg": "File gambar corrupt atau tidak valid."}), 400

        # Jalankan mesin prediksi dengan mengembalikan visualisasi overlay gambar Base64
        result = predict_image(img_bgr, return_image_b64=True)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "error_msg": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    # Menjalankan server lokal pada port 5000 dengan fitur auto-reload (debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
