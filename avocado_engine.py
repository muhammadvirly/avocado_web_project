# avocado_engine.py — Modul inti klasifikasi alpukat (Versi Tekstur Sensitif & Performa Tinggi)
import cv2
import json
import base64
import os
import numpy as np
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'avocado_config_v2.json')

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config_data = json.load(f)
    IMG_SIZE_DIM = tuple(config_data['img_size'])
    MIN_MASK_AREA_VAL = config_data['min_mask_area']
    RATIO_THRESHOLD_VAL = config_data['ratio_threshold']
    
    bounds = config_data['hsv_bounds']
    V_LOWER_BAGUS = np.array(bounds['lower_bagus'])
    V_UPPER_BAGUS = np.array(bounds['upper_bagus'])
    V_LOWER_BUSUK = np.array(bounds['lower_busuk'])
    V_UPPER_BUSUK = np.array(bounds['upper_busuk'])
else:
    IMG_SIZE_DIM = (300, 300)
    MIN_MASK_AREA_VAL = 3500  # Diturunkan ke 3500 agar objek bintik/busuk yang areanya terpecah di awal tetap lolos
    RATIO_THRESHOLD_VAL = 0.65
    
    # KUNCI PERBAIKAN 1: Melonggarkan Saturation (S) dan Value (V) Bagus untuk menoleransi bintik kulit
    V_LOWER_BAGUS = np.array([30, 25, 30])    # S diturunkan ke 25, V ke 30
    V_UPPER_BAGUS = np.array([90, 255, 255])
    
    # KUNCI PERBAIKAN 2: Memperluas jangkauan Busuk (Warna cokelat tua kering hingga mendekati hitam legam)
    V_LOWER_BUSUK = np.array([0, 30, 25])     # S diturunkan ke 30 agar hitam/cokelat pudar ikut masuk
    V_UPPER_BUSUK = np.array([25, 255, 160])   # V dinaikkan ke 160 untuk menangkap busuk yang agak terang

LABELS = {
    'bagus': {'nama': 'Alpukat Bagus', 'warna_hex': '#10B981'},
    'busuk': {'nama': 'Alpukat Busuk', 'warna_hex': '#EF4444'},
    'unknown': {'nama': 'MENUNGGU BUAH...', 'warna_hex': '#6B7280'}
}

def process_avocado_image(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None: return None
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        return None

    # 1. Prapemrosesan Skala Citra
    img_resized = cv2.resize(img, IMG_SIZE_DIM)

    # 2. CLAHE Adaptif untuk Meratakan Refleksi Kilap Layar HP / Cahaya Lampu
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    img_normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Ekstraksi HSV
    hsv = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2HSV)

    # 4. Segmentasi Warna Awal
    mask_bagus_raw = cv2.inRange(hsv, V_LOWER_BAGUS, V_UPPER_BAGUS)
    mask_busuk_raw = cv2.inRange(hsv, V_LOWER_BUSUK, V_UPPER_BUSUK)

    # 5. KUNCI PERBAIKAN 3: OPERASI MORFOLOGI AGRESIF PENUTUP BINTIK (CLOSING)
    # Menggunakan kernel besar berbentuk lingkaran untuk menyatukan bintik putih / lubang kilap di dalam buah
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    mask_bagus_filled = cv2.morphologyEx(mask_bagus_raw, cv2.MORPH_CLOSE, kernel_close)
    mask_busuk_filled = cv2.morphologyEx(mask_busuk_raw, cv2.MORPH_CLOSE, kernel_close)
    
    # Gabungkan masker yang sudah mulus untuk mencari geometri buah
    mask_gabungan = cv2.bitwise_or(mask_bagus_filled, mask_busuk_filled)
    
    # Pembersihan noise luar bodi buah menggunakan MORPH_OPEN ringan
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_gabungan = cv2.morphologyEx(mask_gabungan, cv2.MORPH_OPEN, kernel_open)
    
    kontur, _ = cv2.findContours(mask_gabungan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_alpukat_saja = np.zeros_like(mask_gabungan)
    is_valid_fruit = False

    if kontur:
        kontur_terbesar = max(kontur, key=cv2.contourArea)
        area_kontur = cv2.contourArea(kontur_terbesar)

        if area_kontur >= MIN_MASK_AREA_VAL:
            hull = cv2.convexHull(kontur_terbesar)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_kontur) / hull_area if hull_area > 0 else 0
            
            x, y, w, h = cv2.boundingRect(kontur_terbesar)
            aspect_ratio = float(w) / h

            # Solidity diturunkan aman ke 0.50 karena buah busuk bentuk strukturnya sering tidak rata / pecah kontur
            if solidity > 0.50 and (0.3 <= aspect_ratio <= 2.2):
                is_valid_fruit = True
                cv2.drawContours(mask_alpukat_saja, [kontur_terbesar], -1, 255, -1)

    # 6. Ekstraksi Piksel Berdasarkan Kontur Solid Terpilih
    if is_valid_fruit:
        mask_bagus_clean = cv2.bitwise_and(mask_bagus_raw, mask_alpukat_saja)
        mask_busuk_clean = cv2.bitwise_and(mask_busuk_raw, mask_alpukat_saja)
        img_final_processing = cv2.bitwise_and(img_normalized, img_normalized, mask=mask_alpukat_saja)
    else:
        mask_bagus_clean = np.zeros_like(mask_bagus_raw)
        mask_busuk_clean = np.zeros_like(mask_busuk_raw)
        img_final_processing = np.zeros_like(img_normalized)

    hsv_clean = cv2.cvtColor(img_final_processing, cv2.COLOR_BGR2HSV)

    # 7. Hitung Statistik Piksel Real-Time
    area_bagus = int(cv2.countNonZero(mask_bagus_clean))
    area_busuk = int(cv2.countNonZero(mask_busuk_clean))
    total_area = area_bagus + area_busuk

    h_mean_g, s_mean_g, v_mean_g = 0.0, 0.0, 0.0
    if area_bagus > 0:
        mean_g = cv2.mean(hsv_clean, mask=mask_bagus_clean)
        h_mean_g, s_mean_g, v_mean_g = round(mean_g[0], 2), round(mean_g[1], 2), round(mean_g[2], 2)

    h_mean_k, s_mean_k, v_mean_k = 0.0, 0.0, 0.0
    if area_busuk > 0:
        mean_k = cv2.mean(hsv_clean, mask=mask_busuk_clean)
        h_mean_k, s_mean_k, v_mean_k = round(mean_k[0], 2), round(mean_k[1], 2), round(mean_k[2], 2)

    # 8. LOGIKA PENENTUAN KEPUTUSAN STRATEGIS (Lebih Sensitif & Konsisten)
    if not is_valid_fruit or total_area < MIN_MASK_AREA_VAL:
        label_pred = 'unknown'
        confidence = 0.0
    else:
        pct_busuk = (area_busuk / total_area) * 100.0
        ratio = area_bagus / (area_busuk + 1.0)

        # AMBANG BATAS ADAPTIF: Jika luas noda cokelat/gelap melebihi 8% dari total bodi, langsung tandai BUSUK
        if pct_busuk > 8.0:
            label_pred = 'busuk'
            # Skala interpolasi confidence agar stabil di angka tinggi saat objek jelas busuk
            confidence = 50.0 + (pct_busuk * 4.5)
        elif 10.0 <= h_mean_g <= 75.0 and ratio > 0.25:
            label_pred = 'bagus'
            confidence = (area_bagus / total_area) * 100.0
        else:
            if ratio > RATIO_THRESHOLD_VAL:
                label_pred = 'bagus'
                confidence = (area_bagus / total_area) * 100.0
            else:
                label_pred = 'busuk'
                confidence = (area_busuk / total_area) * 100.0

    # Batasi rasio interpolasi agar nilai confidence tidak melonjak tidak menentu (Statis & Halus)
    confidence = max(50.0 if label_pred != 'unknown' else 0.0, min(99.4, float(confidence)))
    nama_label = LABELS[label_pred]['nama']
    warna_bgr = (0, 200, 0) if label_pred == 'bagus' else ((0, 0, 200) if label_pred == 'busuk' else (128, 128, 128))

    return {
        'original': img, 'resized': img_final_processing, 'blurred': img_final_processing, 'hsv': hsv_clean,
        'raw_masks': {'mask_bagus': mask_bagus_raw, 'mask_busuk': mask_busuk_raw},
        'clean_masks': {'mask_bagus': mask_bagus_clean, 'mask_busuk': mask_busuk_clean},
        'result': {
            'label': label_pred, 'nama': nama_label, 'confidence': round(confidence, 2),
            'area_bagus': area_bagus, 'area_busuk': area_busuk, 'warna_bgr': warna_bgr
        },
        'features': {
            'mask_bagus': {'area': area_bagus, 'pct': round((area_bagus/max(1, total_area))*100, 2), 'h_mean': h_mean_g, 's_mean': s_mean_g, 'v_mean': v_mean_g},
            'mask_busuk': {'area': area_busuk, 'pct': round((area_busuk/max(1, total_area))*100, 2), 'h_mean': h_mean_k, 's_mean': s_mean_k, 'v_mean': v_mean_k}
        }
    }

def predict_image(image_input, return_image_b64=False):
    response = {
        'status'      : 'error',
        'label'       : 'unknown',
        'label_nama'  : 'MENUNGGU BUAH...',
        'confidence'  : 0.0,
        'area_bagus'  : 0,
        'area_busuk'  : 0,
        'pct_bagus'   : 0.0,
        'pct_busuk'   : 0.0,
        'h_mean_bagus': 0.0, 's_mean_bagus': 0.0, 'v_mean_bagus': 0.0,
        'h_mean_busuk': 0.0, 's_mean_busuk': 0.0, 'v_mean_busuk': 0.0,
        'warna_hex'   : '#6B7280',
        'timestamp'   : datetime.now().isoformat(timespec='seconds'),
        'image_b64'   : None,
        'error_msg'   : None
    }

    try:
        out = process_avocado_image(image_input)
        if out is None:
            response['error_msg'] = 'Gagal memuat atau memproses gambar.'
            return response

        r  = out['result']
        f  = out['features']
        fb = f.get('mask_bagus', {})
        fk = f.get('mask_busuk', {})

        response['status']       = 'success'
        response['label']        = r['label']
        response['label_nama']   = r['nama']
        response['confidence']   = r['confidence']
        response['area_bagus']   = r['area_bagus']
        response['area_busuk']   = r['area_busuk']
        response['pct_bagus']    = fb.get('pct',    0.0)
        response['pct_busuk']    = fk.get('pct',    0.0)
        response['h_mean_bagus'] = fb.get('h_mean', 0.0)
        response['s_mean_bagus'] = fb.get('s_mean', 0.0)
        response['v_mean_bagus'] = fb.get('v_mean', 0.0)
        response['h_mean_busuk'] = fk.get('h_mean', 0.0)
        response['s_mean_busuk'] = fk.get('s_mean', 0.0)
        response['v_mean_busuk'] = fk.get('v_mean', 0.0)
        response['warna_hex']    = LABELS[r['label']]['warna_hex']

        if return_image_b64:
            img_overlay = out['resized'].copy()
            color_bgr   = r['warna_bgr']
            
            cv2.rectangle(img_overlay, (0, 0), (img_overlay.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(img_overlay, r['nama'].upper(), (8, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255) if r['label']=='unknown' else color_bgr, 2, cv2.LINE_AA)
            
            if r['label'] != 'unknown':
                cv2.rectangle(img_overlay, (2, 2), (img_overlay.shape[1]-2, img_overlay.shape[0]-2), color_bgr, 3)

            _, buf = cv2.imencode('.png', img_overlay)
            response['image_b64'] = base64.b64encode(buf).decode('utf-8')

    except Exception as e:
        response['error_msg'] = str(e)

    return response