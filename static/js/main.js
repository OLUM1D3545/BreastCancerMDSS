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
        const files = this.files;
        if (files.length > 0) {
            uploadPlaceholder.style.display = 'none';
            uploadPreview.style.display = 'block';

            const previewGrid = document.getElementById('previewGrid');
            previewGrid.innerHTML = '';

            // Show preview for each uploaded image (max 4)
            const maxFiles = Math.min(files.length, 4);
            for (let i = 0; i < maxFiles; i++) {
                const file = files[i];
                const reader = new FileReader();
                reader.onload = function(e) {
                    const imgWrapper = document.createElement('div');
                    imgWrapper.style.cssText = 'position:relative;';
                    imgWrapper.innerHTML = `
                        <img src="${e.target.result}" style="width:100%;height:120px;object-fit:cover;border-radius:8px;background:#000;">
                        <div style="font-size:11px;color:#475569;margin-top:4px;text-align:center;">${file.name.length > 20 ? file.name.substring(0, 20) + '...' : file.name}</div>
                    `;
                    previewGrid.appendChild(imgWrapper);
                };
                reader.readAsDataURL(file);
            }

            fileName.textContent = `${files.length} file${files.length > 1 ? 's' : ''} selected`;
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
// ── Live Risk Score Preview ──
// Updates the risk score in real time as the doctor
// fills in the clinical data fields

function updateLiveRisk() {
    const age = parseInt(document.querySelector('[name="age"]')?.value) || 0;
    const birads = parseInt(document.querySelector('[name="birads"]')?.value) || 0;
    const familyHistory = parseInt(document.querySelector('[name="family_history"]')?.value) || 0;
    const density = parseInt(document.querySelector('[name="density"]')?.value) || 1;
    const menopause = parseInt(document.querySelector('[name="menopause"]')?.value) || 0;
    const priorBiopsy = parseInt(document.querySelector('[name="prior_biopsy"]')?.value) || 0;

    // Calculate live risk score
    let risk = 0;
    const biradsFactor = birads * 0.15;
    const ageFactor = age > 0 ? (age - 18) / 82 * 0.25 : 0;
    const familyFactor = familyHistory * 0.20;
    const densityFactor = density / 4 * 0.12;
    const menopauseFactor = menopause * 0.10;
    const biopsyFactor = priorBiopsy * 0.08;

    risk = biradsFactor + ageFactor + familyFactor + densityFactor + menopauseFactor + biopsyFactor;
    risk = Math.min(Math.max(risk, 0), 0.99);

    const riskPercent = Math.round(risk * 100);
    const scoreEl = document.getElementById('liveRiskScore');
    const labelEl = document.getElementById('liveRiskLabel');
    const fillEl = document.getElementById('liveRiskFill');
    const factorsEl = document.getElementById('liveRiskFactors');

    if (!scoreEl) return;

    // Only show if at least one field is filled
    if (age === 0 && birads === 0) {
        scoreEl.textContent = '--';
        labelEl.textContent = 'Fill in patient details to see live risk estimate';
        fillEl.style.width = '0%';
        factorsEl.innerHTML = '';
        return;
    }

    // Update score display
    scoreEl.textContent = riskPercent + '%';
    fillEl.style.width = riskPercent + '%';

    // Update label based on risk level
    if (riskPercent < 30) {
        labelEl.textContent = '🟢 LOW RISK — Routine annual screening recommended';
        scoreEl.style.color = '#2C6E49';
    } else if (riskPercent < 70) {
        labelEl.textContent = '🟡 INTERMEDIATE RISK — Follow-up in 6 months recommended';
        scoreEl.style.color = '#F5A623';
    } else {
        labelEl.textContent = '🔴 HIGH RISK — Immediate biopsy referral recommended';
        scoreEl.style.color = '#C84B31';
    }

    // Show factor breakdown
    const total = biradsFactor + ageFactor + familyFactor + densityFactor + menopauseFactor + biopsyFactor + 0.001;
    const factors = [
        { name: 'BI-RADS Score', val: biradsFactor, pct: Math.round(biradsFactor / total * 100) },
        { name: 'Patient Age', val: ageFactor, pct: Math.round(ageFactor / total * 100) },
        { name: 'Family History', val: familyFactor, pct: Math.round(familyFactor / total * 100) },
        { name: 'Breast Density', val: densityFactor, pct: Math.round(densityFactor / total * 100) },
        { name: 'Menopausal Status', val: menopauseFactor, pct: Math.round(menopauseFactor / total * 100) },
        { name: 'Prior Biopsy', val: biopsyFactor, pct: Math.round(biopsyFactor / total * 100) },
    ];

    factorsEl.innerHTML = factors.map(f => `
        <div class="live-factor-item">
            <span>${f.name}</span>
            <div class="live-factor-bar">
                <div class="live-factor-fill" style="width: ${f.pct}%"></div>
            </div>
            <span class="live-factor-val">${f.pct}%</span>
        </div>
    `).join('');
}

// Listen for changes on all clinical fields
document.querySelectorAll('[name="age"],[name="birads"],[name="family_history"],[name="density"],[name="menopause"],[name="prior_biopsy"]').forEach(function(el) {
    el.addEventListener('input', updateLiveRisk);
    el.addEventListener('change', updateLiveRisk);
});
// ── Tooltip Toggle for Mobile ──
document.querySelectorAll('.tooltip-icon').forEach(function(icon) {
    icon.addEventListener('click', function(e) {
        e.stopPropagation();
        const wrap = this.parentElement;
        const isActive = wrap.classList.contains('active');
        document.querySelectorAll('.tooltip-wrap').forEach(w => w.classList.remove('active'));
        if (!isActive) wrap.classList.add('active');
    });
});

document.addEventListener('click', function() {
    document.querySelectorAll('.tooltip-wrap').forEach(w => w.classList.remove('active'));
});
// ── Voice Summary ──
// Reads out the result automatically when the results page loads
// Uses the browser's built-in text-to-speech

function speakResult() {
    const resultBadge = document.querySelector('.result-badge');
    const riskLevel = document.querySelector('.result-card-title');

    if (!resultBadge) return; // Only run on results page

    const prediction = resultBadge.textContent.trim().split('—')[0].trim();
    const confidence = resultBadge.textContent.trim().split('%')[0].split('—')[1]?.trim() || '';
    const riskEl = document.querySelectorAll('.risk-tier.active')[0];
    const risk = riskEl ? riskEl.textContent.trim().split('\n')[0].trim() : '';
    const actionEl = document.querySelector('.risk-action');
    const action = actionEl ? actionEl.textContent.replace('Recommended Action:', '').trim() : '';

    const message = `Analysis complete. 
        The AI model predicts this case as ${prediction}. 
        Confidence level is ${confidence} percent. 
        The overall risk assessment is ${risk} RISK. 
        ${action}. 
        Please consult with a qualified healthcare professional before making any clinical decisions.`;

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.rate = 0.85;
        utterance.pitch = 1;
        utterance.volume = 1;

        // Try to use a clear English voice
        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find(v =>
            v.lang.startsWith('en') && v.name.includes('Female')
        ) || voices.find(v => v.lang.startsWith('en')) || voices[0];

        if (preferred) utterance.voice = preferred;
        window.speechSynthesis.speak(utterance);
    }
}

// ── Voice control buttons ──
function addVoiceControls() {
    const actionsBar = document.querySelector('.actions-bar');
    if (!actionsBar) return;

    const voiceDiv = document.createElement('div');
    voiceDiv.style.cssText = 'text-align:center;margin-top:16px;';
    voiceDiv.innerHTML = `
        <button onclick="speakResult()" style="
            background: #7B2D8B;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            margin-right: 8px;
            font-family: Inter, sans-serif;
        ">🔊 Read Results Aloud</button>
        <button onclick="window.speechSynthesis.cancel()" style="
            background: transparent;
            color: #475569;
            border: 1.5px solid #e2e8f0;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            font-family: Inter, sans-serif;
        ">⏹ Stop</button>
    `;
    actionsBar.appendChild(voiceDiv);
}

// Run voice controls on results page
window.addEventListener('load', function() {
    addVoiceControls();

    // Auto speak after 1.5 seconds on results page
    if (document.querySelector('.result-badge')) {
        setTimeout(speakResult, 1500);
    }
});