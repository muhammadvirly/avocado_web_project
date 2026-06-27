// script.js - Logika Manajemen Media Aliran Webcam & Sinkronisasi UI Dinamis
document.addEventListener('DOMContentLoaded', () => {
    const liveWebcam = document.getElementById('live-webcam');
    const hiddenCanvas = document.getElementById('hidden-frame-canvas');
    const canvasCtx = hiddenCanvas.getContext('2d');

    // Deklarasi Node Komponen UI Antarmuka
    const webcamCardContainer = document.getElementById('webcam-card-container');
    const statusPanelContainer = document.getElementById('status-panel-container');
    const appHeader = document.getElementById('app-header');
    const uiNamaLabel = document.getElementById('ui-nama-label');
    const uiBadgeStatus = document.getElementById('ui-badge-status');
    const uiTextConfidence = document.getElementById('ui-text-confidence');
    const uiBarConfidence = document.getElementById('ui-bar-confidence');
    const uiAreaBagus = document.getElementById('ui-area-bagus');
    const uiAreaBusuk = document.getElementById('ui-area-busuk');
    const uiApiStatus = document.getElementById('ui-api-status');
    const uiTimestamp = document.getElementById('ui-timestamp');

    // 1. Ambil Aliran Data Kamera Menggunakan Browser WebRTC API
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
            .then((stream) => {
                liveWebcam.srcObject = stream;
                if (uiApiStatus) {
                    uiApiStatus.innerText = "Server Status: Kamera Online";
                    uiApiStatus.className = "text-emerald-400 font-bold";
                }
                
                // Memicu siklus transfer pengiriman gambar setiap 400ms konstan (Real-time)
                setInterval(captureAndPostFrame, 400);
            })
            .catch((err) => {
                console.error("Akses izin kamera ditolak:", err);
                if (uiApiStatus) {
                    uiApiStatus.innerText = "Server Status: Kamera Diblokir";
                    uiApiStatus.className = "text-red-500 font-bold";
                }
            });
    }

    // 2. Tangkap Gambar Frame Canvas & Kirim POST FormData ke Flask
    function captureAndPostFrame() {
        if (liveWebcam.readyState !== liveWebcam.HAVE_ENOUGH_DATA) return;

        // Atur dimensi canvas tersembunyi agar pas dengan aspek rasio video stream
        hiddenCanvas.width = liveWebcam.videoWidth;
        hiddenCanvas.height = liveWebcam.videoHeight;

        // Gambar frame dari video stream ke objek canvas tersembunyi
        canvasCtx.drawImage(liveWebcam, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        
        // SINKRONISASI FLASK: Ubah data matriks canvas menjadi Blob File virtual (.png)
        hiddenCanvas.toBlob((blob) => {
            if (!blob) return;

            let formData = new FormData();
            // Sesuaikan key "file" dengan yang diminta oleh request.files di app.py
            formData.append("file", blob, "webcam_frame.png");

            // Menembak alamat endpoint Flask server eksternal port 5000 secara eksplisit
            fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData // Mengirim multipart/form-data biner asli
            })
            .then(response => {
                if (!response.ok) throw new Error("API Offline / Bad Response");
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    // Berikan objek data root utama karena variabel tidak lagi bersarang di data.result
                    renderResponseDataToUI(data);
                    if (uiApiStatus) {
                        uiApiStatus.innerText = "Server Status: Terhubung";
                        uiApiStatus.className = "text-emerald-400 font-bold";
                    }
                }
            })
            .catch(error => {
                console.error("Gagal mengirim data streaming ke server:", error);
                if (uiApiStatus) {
                    uiApiStatus.innerText = "Server Status: Flask Offline";
                    uiApiStatus.className = "text-red-500 font-bold";
                }
            });
        }, 'image/png');
    }

    // 3. Mutasi Warna Elemen UI Secara Real-time Sesuai Aturan Mutu Buah
    function renderResponseDataToUI(data) {
        // ---- PERBAIKAN UTAMA: Ambil langsung dari 'data', bukan dari 'result' ----
        if (uiNamaLabel && data.label_nama) {
            uiNamaLabel.innerText = data.label_nama.toUpperCase();
        }
        
        if (uiTextConfidence && typeof data.confidence === 'number') {
            uiTextConfidence.innerText = `${data.confidence.toFixed(1)}%`;
        }
        
        if (uiBarConfidence && typeof data.confidence === 'number') {
            uiBarConfidence.style.width = `${data.confidence}%`;
        }
        
        if (uiAreaBagus && typeof data.area_bagus === 'number') {
            uiAreaBagus.innerHTML = `${data.area_bagus.toLocaleString()} <span class="text-xs font-normal text-gray-500">px</span>`;
        }
        
        if (uiAreaBusuk && typeof data.area_busuk === 'number') {
            uiAreaBusuk.innerHTML = `${data.area_busuk.toLocaleString()} <span class="text-xs font-normal text-gray-500">px</span>`;
        }
        
        const waktuSekarang = new Date();
        if (uiTimestamp) {
            uiTimestamp.innerText = `Waktu Ambil: ${waktuSekarang.toLocaleTimeString('id-ID')}`;
        }

        // Alur Seleksi Perubahan Warna Tema Dasbor Komputer Vision
        if (data.label === 'bagus') {
            // Tema Warna Bagus (Nuansa Estetik Hijau Alpukat)
            if (webcamCardContainer) webcamCardContainer.className = "relative bg-gray-800 rounded-2xl p-4 shadow-2xl border-4 border-emerald-500 transition-smooth";
            if (statusPanelContainer) statusPanelContainer.className = "bg-emerald-950/70 rounded-2xl p-6 shadow-2xl border border-emerald-500/40 transition-smooth";
            if (appHeader) appHeader.className = "bg-emerald-800 border-b border-emerald-700 shadow-md transition-smooth";
            if (uiNamaLabel) uiNamaLabel.className = "text-3xl font-black tracking-tight text-emerald-400 transition-smooth";
            
            if (uiBadgeStatus) {
                uiBadgeStatus.innerText = "BAGUS";
                uiBadgeStatus.className = "bg-emerald-500/20 text-emerald-400 text-xs px-3 py-1.5 rounded-lg font-black border border-emerald-500/30 tracking-widest uppercase";
            }
            if (uiBarConfidence) uiBarConfidence.className = "bg-emerald-500 h-full transition-all duration-300 rounded-full shadow shadow-emerald-400";
            
        } else if (data.label === 'busuk') {
            // Tema Warna Busuk (Nuansa Tegas Merah Cacat)
            if (webcamCardContainer) webcamCardContainer.className = "relative bg-gray-800 rounded-2xl p-4 shadow-2xl border-4 border-red-600 transition-smooth";
            if (statusPanelContainer) statusPanelContainer.className = "bg-red-950/60 rounded-2xl p-6 shadow-2xl border border-red-600/40 transition-smooth";
            if (appHeader) appHeader.className = "bg-red-900 border-b border-red-800 shadow-md transition-smooth";
            if (uiNamaLabel) uiNamaLabel.className = "text-3xl font-black tracking-tight text-red-400 transition-smooth";
            
            if (uiBadgeStatus) {
                uiBadgeStatus.innerText = "BUSUK / CACAT";
                uiBadgeStatus.className = "bg-red-500/20 text-red-400 text-xs px-3 py-1.5 rounded-lg font-black border border-red-500/30 tracking-widest uppercase";
            }
            if (uiBarConfidence) uiBarConfidence.className = "bg-red-600 h-full transition-all duration-300 rounded-full shadow shadow-red-500";
            
        } else {
            // Tema Standby Idle (Nuansa Abu-abu - Menunggu Buah / ROI Guard Terkunci)
            if (webcamCardContainer) webcamCardContainer.className = "relative bg-gray-800 rounded-2xl p-4 shadow-2xl border-4 border-gray-700 transition-smooth";
            if (statusPanelContainer) statusPanelContainer.className = "bg-gray-800 rounded-2xl p-6 shadow-2xl border border-gray-700/80 transition-smooth";
            if (appHeader) appHeader.className = "bg-emerald-800 border-b border-emerald-700 shadow-md transition-smooth";
            if (uiNamaLabel) uiNamaLabel.className = "text-3xl font-black tracking-tight text-gray-300 transition-smooth";
            
            if (uiBadgeStatus) {
                uiBadgeStatus.innerText = "IDLE";
                uiBadgeStatus.className = "bg-gray-700 text-gray-400 text-xs px-3 py-1.5 rounded-lg font-black border border-gray-600 tracking-wider";
            }
            if (uiBarConfidence) uiBarConfidence.className = "bg-gray-500 h-full transition-all duration-300 rounded-full";
        }
    }
});