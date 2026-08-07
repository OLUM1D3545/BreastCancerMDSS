// ═══════════════════════════════════════
// MDSS - Breast Cancer Detection System
// Main JavaScript File
// ═══════════════════════════════════════

// ── Image Upload Preview ──
// When a doctor uploads a mammogram, this shows a preview
// of the image so they can confirm it is the right file

const imageInput = document.getElementById('imageInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const uploadArea = document.getElementById('uploadArea');

if (imageInput) {
    imageInput.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            // Show the image preview
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImg.src = e.target.result;
                fileName.textContent = file.name;
                uploadPlaceholder.style.display = 'none';
                uploadPreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
}

// ── Drag and Drop Support ──
// Allows the doctor to drag a mammogram image
// directly onto the upload area instead of clicking

if (uploadArea) {
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function () {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            imageInput.files = e.dataTransfer.files;
            const reader = new FileReader();
            reader.onload = function (ev) {
                previewImg.src = ev.target.result;
                fileName.textContent = file.name;
                uploadPlaceholder.style.display = 'none';
                uploadPreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Make the upload area clickable anywhere
    uploadArea.addEventListener('click', function (e) {
        if (e.target !== imageInput && !e.target.closest('.upload-preview')) {
            imageInput.click();
        }
    });
}

// ── Loading Overlay ──
// When the doctor clicks Run Analysis, this shows
// a loading screen while the AI model is processing

const analysisForm = document.getElementById('analysisForm');
const submitBtn = document.getElementById('submitBtn');

if (analysisForm) {
    analysisForm.addEventListener('submit', function (e) {

        // Check if an image has been uploaded
        if (!imageInput || !imageInput.files[0]) {
            e.preventDefault();
            alert('Please upload a mammogram image before running the analysis.');
            return;
        }

        // Show loading overlay
        showLoading();

        // Change button text
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Analysing...</span>';
        }
    });
}

function showLoading() {
    // Create loading overlay
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay active';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div class="loading-spinner"></div>
        <div class="loading-text">Running AI Analysis...</div>
        <div class="loading-sub">CNN feature extraction in progress. Please wait.</div>
    `;
    document.body.appendChild(overlay);
}

// ── Confidence Bar Animation ──
// Animates the confidence percentage bar on the results page

function animateConfidenceBar() {
    const fill = document.querySelector('.confidence-fill');
    if (fill) {
        const width = fill.getAttribute('data-width');
        setTimeout(() => {
            fill.style.width = width + '%';
        }, 300);
    }
}

// ── SHAP Bar Animation ──
// Animates the SHAP importance bars on the results page

function animateSHAPBars() {
    const bars = document.querySelectorAll('.shap-bar-fill');
    bars.forEach((bar, index) => {
        const width = bar.getAttribute('data-width');
        setTimeout(() => {
            bar.style.width = width + '%';
        }, 300 + index * 100);
    });
}

// ── Print Report ──
// Allows the clinician to print the results page
// as a clinical report

function printReport() {
    window.print();
}

// ── Run animations when page loads ──
window.addEventListener('load', function () {
    animateConfidenceBar();
    animateSHAPBars();
});

// ── Dark Mode Toggle ──
// This adds a dark mode button to the website
// When clicked it switches between light and dark mode
// and remembers the user's preference

function createDarkModeToggle() {
    // Create the toggle button
    const toggle = document.createElement('button');
    toggle.id = 'darkModeToggle';
    toggle.innerHTML = '🌙';
    toggle.title = 'Toggle Dark Mode';
    toggle.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: none;
        background: #002060;
        color: white;
        font-size: 20px;
        cursor: pointer;
        z-index: 9999;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: all 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    document.body.appendChild(toggle);

    // Check if user previously chose dark mode
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
        toggle.innerHTML = '☀️';
    }

    // Toggle dark mode when button is clicked
    toggle.addEventListener('click', function () {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('darkMode', isDark);
        toggle.innerHTML = isDark ? '☀️' : '🌙';
    });
}

// Run when page loads
createDarkModeToggle();