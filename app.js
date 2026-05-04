// ==========================================
// DOM ELEMENTS
// ==========================================

// Landing Page Elements
const landingPage = document.getElementById('landing-page');
const mainApp = document.getElementById('main-app');
const landingUploadBtn = document.getElementById('landingUploadBtn');
const landingCameraBtn = document.getElementById('landingCameraBtn');

// Main App Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const scanBtn = document.getElementById('scanBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const detectionCircle = document.getElementById('detectionCircle');
const countDisplay = document.getElementById('countDisplay');
const personCount = document.getElementById('personCount');
const threatValue = document.getElementById('threatValue');
const threatFill = document.getElementById('threatFill');
const fpsValue = document.getElementById('fpsValue');
const logContainer = document.getElementById('logContainer');
const detectionList = document.getElementById('detectionList');
const resultsImage = document.getElementById('resultsImage');
const annotatedImage = document.getElementById('annotatedImage');
const alertSound = document.getElementById('alertSound');

// Mode switching
const uploadModeBtn = document.getElementById('uploadModeBtn');
const cameraModeBtn = document.getElementById('cameraModeBtn');
const uploadSection = document.getElementById('uploadSection');
const cameraSection = document.getElementById('cameraSection');

// Camera elements
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');

let selectedFile = null;
let cameraStream = null;
let currentMode = 'upload'; // 'upload' or 'camera'

// ==========================================
// LANDING PAGE LOGIC
// ==========================================

function transitionToApp(mode) {
    // 1. Fade out the landing page
    landingPage.style.opacity = '0';
    landingPage.style.pointerEvents = 'none'; // Prevent double clicks

    // 2. Wait for CSS transition (0.8s), then switch views
    setTimeout(() => {
        landingPage.style.display = 'none';
        mainApp.style.display = 'block'; // Show Main App
        
        // Simple fade-in animation for main app
        mainApp.style.opacity = '0';
        requestAnimationFrame(() => {
            mainApp.style.transition = 'opacity 0.5s ease';
            mainApp.style.opacity = '1';
        });

        // 3. Initialize the correct mode based on user choice
        if (mode === 'upload') {
            uploadModeBtn.click(); // Triggers existing logic
        } else {
            cameraModeBtn.click(); // Triggers existing logic
            // Optional: Auto-start camera if they clicked "Live Camera"
            // startCameraBtn.click(); 
        }
        
        addLog('› SYSTEM ACCESSED VIA SECURE LOGIN');
    }, 800);
}

// Event Listeners for Landing Page
landingUploadBtn.addEventListener('click', () => transitionToApp('upload'));
landingCameraBtn.addEventListener('click', () => transitionToApp('camera'));


// ==========================================
// MAIN APP LOGIC
// ==========================================

// Mode Switching
uploadModeBtn.addEventListener('click', () => {
    currentMode = 'upload';
    uploadModeBtn.classList.add('active');
    cameraModeBtn.classList.remove('active');
    uploadSection.style.display = 'block';
    cameraSection.style.display = 'none';
    stopCamera();
    addLog('› SWITCHED TO UPLOAD MODE');
});

cameraModeBtn.addEventListener('click', () => {
    currentMode = 'camera';
    cameraModeBtn.classList.add('active');
    uploadModeBtn.classList.remove('active');
    cameraSection.style.display = 'block';
    uploadSection.style.display = 'none';
    selectedFile = null;
    scanBtn.disabled = true;
    addLog('› SWITCHED TO CAMERA MODE');
});

// Camera Controls
startCameraBtn.addEventListener('click', async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        cameraVideo.srcObject = cameraStream;
        startCameraBtn.style.display = 'none';
        stopCameraBtn.style.display = 'block';
        scanBtn.disabled = false;
        addLog('› CAMERA STARTED');
    } catch (error) {
        addLog(`ERROR: Camera access denied - ${error.message}`);
        alert('Camera access denied. Please allow camera permissions.');
    }
});

stopCameraBtn.addEventListener('click', () => {
    stopCamera();
});

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        cameraVideo.srcObject = null;
        startCameraBtn.style.display = 'block';
        stopCameraBtn.style.display = 'none';
        scanBtn.disabled = true;
        addLog('› CAMERA STOPPED');
    }
}

// Upload Area Click
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#00fff7';
    uploadArea.style.background = 'rgba(0, 255, 247, 0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'rgba(0, 255, 247, 0.4)';
    uploadArea.style.background = 'transparent';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(0, 255, 247, 0.4)';
    uploadArea.style.background = 'transparent';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File Input Change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Selection
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        addLog('ERROR: Invalid file type. Please upload an image.');
        return;
    }

    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        scanBtn.disabled = false;
        addLog(`› IMAGE LOADED: ${file.name}`);
    };
    reader.readAsDataURL(file);
}

// Remove Image
removeBtn.addEventListener('click', () => {
    selectedFile = null;
    previewImage.src = '';
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    scanBtn.disabled = true;
    resultsImage.style.display = 'none';
    resetResults();
    addLog('› IMAGE REMOVED');
});

// Scan Button Click
scanBtn.addEventListener('click', () => {
    if (currentMode === 'upload' && !selectedFile) return;
    if (currentMode === 'camera' && !cameraStream) return;
    
    performScan();
});

// Perform Scan
async function performScan() {
    loadingOverlay.classList.add('active');
    addLog('› INITIATING SCAN...');
    
    const formData = new FormData();
    
    if (currentMode === 'upload') {
        formData.append('image', selectedFile);
    } else if (currentMode === 'camera') {
        // Capture frame from camera
        const canvas = cameraCanvas;
        const video = cameraVideo;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Convert canvas to blob
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
        formData.append('image', blob, 'camera-capture.jpg');
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayResults(data);
        } else {
            addLog(`ERROR: ${data.message}`);
        }
    } catch (error) {
        addLog(`ERROR: ${error.message}`);
    } finally {
        loadingOverlay.classList.remove('active');
    }
}

// Display Results
function displayResults(data) {
    // Update count
    countDisplay.textContent = data.people_count;
    personCount.textContent = data.people_count;
    
    // Update threat level
    threatValue.textContent = data.threat_level;
    threatValue.className = 'threat-value ' + data.threat_level.toLowerCase();
    threatFill.className = 'threat-fill ' + data.threat_level.toLowerCase();
    
    // Update FPS
    fpsValue.textContent = data.fps.toFixed(1);
    
    // Update detection circle
    if (data.people_count > 0) {
        detectionCircle.classList.add('alert');
        alertSound.play().catch(() => {}); // Play alert sound
    } else {
        detectionCircle.classList.remove('alert');
    }
    
    // Update logs
    data.logs.forEach(log => {
        addLog(`› ${log}`);
    });
    
    // Display annotated image
    if (data.annotated_image) {
        annotatedImage.src = data.annotated_image;
        resultsImage.style.display = 'block';
    }
    
    // Display detection list
    displayDetectionList(data.detections);
    
    // Final log
    addLog(`› SCAN COMPLETE | ${data.people_count} PERSON(S) DETECTED`);
}

// Display Detection List
function displayDetectionList(detections) {
    detectionList.innerHTML = '';
    
    if (detections.length === 0) {
        detectionList.innerHTML = '<div style="text-align:center;opacity:0.5;padding:20px;">NO DETECTIONS</div>';
        return;
    }
    
    detections.forEach((detection, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        item.innerHTML = `
            <div class="detection-info">
                <div class="detection-label">PERSON ${index + 1}</div>
                <div class="detection-conf">CONFIDENCE: ${(detection.confidence * 100).toFixed(1)}%</div>
            </div>
        `;
        detectionList.appendChild(item);
    });
}

// Add Log Entry
function addLog(message) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = message;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Reset Results
function resetResults() {
    countDisplay.textContent = '0';
    personCount.textContent = '0';
    threatValue.textContent = 'LOW';
    threatValue.className = 'threat-value low';
    threatFill.className = 'threat-fill low';
    fpsValue.textContent = '0.0';
    detectionCircle.classList.remove('alert');
    detectionList.innerHTML = '';
    logContainer.innerHTML = '<div class="log-entry">› SYSTEM READY</div><div class="log-entry">› AWAITING INPUT...</div>';
}

// Initialize
addLog('› SYSTEM INITIALIZED');
addLog('› WAITING FOR LANDING SELECTION...');
