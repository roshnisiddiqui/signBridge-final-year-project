/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║  app.js — SignBridge Frontend Application                    ║
 * ║                                                              ║
 * ║  Responsibilities:                                           ║
 * ║  1. Camera access via getUserMedia API                       ║
 * ║  2. WebSocket connection to FastAPI backend                  ║
 * ║  3. Frame capture & encoding (canvas → base64 JPEG)          ║
 * ║  4. Real-time UI updates from backend responses              ║
 * ║  5. Translation text management (add/delete/clear/speak)     ║
 * ║  6. Session analytics display                                ║
 * ║  7. ASL reference guide population                           ║
 * ╚══════════════════════════════════════════════════════════════╝
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

/**
 * API base URL — change this to your deployed backend URL
 * In development: http://localhost:8000
 * In production: https://your-app.railway.app
 */
const API_BASE = window.location.origin;  // Same origin as frontend
const WS_BASE  = API_BASE.replace('http', 'ws'); // ws:// or wss://

// ═══════════════════════════════════════════════════════════════════
// APPLICATION STATE
// Central state object — easy to debug and reason about
// ═══════════════════════════════════════════════════════════════════

const STATE = {
  // Session
  sessionId: generateSessionId(),  // Unique ID for this browser session
  isRunning: false,                 // Whether camera + WS are active
  startTime: null,                  // When session started (for timer)
  sessionInterval: null,            // setInterval handle for session timer

  // WebSocket
  ws: null,                         // WebSocket instance
  wsRetries: 0,                     // Reconnection attempts
  maxRetries: 5,

  // Camera
  stream: null,                     // MediaStream from getUserMedia
  frameInterval: null,              // setInterval for frame capture (fallback)

  // Detection
  translatedText: '',               // The full translated text string
  lastSign: null,                   // Last detected sign
  holdStart: null,                  // When current hold started
  holdDuration: 1800,               // ms to hold sign before confirming
  cooldownMs: 2000,                 // ms to wait after confirm — STOPS SPAM completely
  holdTimer: null,                  // setTimeout for hold confirmation
  lastAdded: 0,                     // Timestamp of last confirmed sign
  noHandTimer: null,                // Timer for auto-space on hand removal
  detectionCount: 0,                // Total confirmed signs this session
  signHistory: {},                  // {sign: count} frequency map

  // UI
  currentMode: 'letter',            // Active detection mode
  settingsOpen: false,              // Whether settings panel is visible
  currentRef: 'alpha',              // Active reference tab

  // Performance
  fps: 0,
  inferenceMs: 0,
};

// Word bank for auto-complete suggestions
// In production: load from backend /api/wordbank endpoint
const WORD_BANK = [
  'HELLO','HELP','HOME','HAND','HAVE','HERE','HI','HOW',
  'GOOD','GOING','GREAT','GIVE','GET',
  'THANKS','THANK','THE','THAT','THIS','TIME','TODAY','TOGETHER',
  'PLEASE','SORRY','STOP','SCHOOL','SIGN','SPEAK','SEE',
  'YES','YOU','YOUR',
  'NO','NAME','NICE','NEED','NEW','NEXT',
  'LOVE','LIKE','LOOK','LEARN','LISTEN',
  'WHAT','WHERE','WHEN','WHO','WHY','WANT','WAIT','WORK','WRITE',
  'CALL','CAN','COME','DO','DONE','DAY','DIFFERENT',
  'BYE','BAD','BACK','BEFORE','BEST','BOTH',
  'ASL','AMERICAN','AGAIN','ALWAYS',
  'FOOD','FEEL','FINE','FRIEND','FAMILY','FAST','FIRST','FORGET',
  'READY','REAL','RIGHT','REMEMBER',
  'UNDERSTAND','US','UNIT','UNTIL',
  'MORE','MY','MAYBE','MEET','MEAN','MORNING',
  'KNOW','KEEP',
];

// ASL sign emoji fallbacks (shown if image fails to load)
const SIGN_EMOJI = {
  'A':'✊','B':'🖐','C':'🤏','D':'👆','E':'🤞','F':'👌','G':'👉',
  'H':'✌️','I':'🤙','J':'🤙','K':'✌️','L':'🤙','M':'✊','N':'✊',
  'O':'👌','P':'👇','Q':'👇','R':'🤞','S':'✊','T':'✊','U':'✌️',
  'V':'✌️','W':'🖖','X':'☝️','Y':'🤙','Z':'☝️',
  '0':'👌','1':'☝️','2':'✌️','3':'🤟','4':'🖐','5':'🖐',
  '6':'🤙','7':'👆','8':'🤌','9':'👌',
};


// ═══════════════════════════════════════════════════════════════════
// INITIALIZATION
// Runs once when page loads
// ═══════════════════════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  updateLoadingMsg('Connecting to backend...');

  // Try to connect to backend health check
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();

    if (data.status === 'healthy') {
      updateLoadingMsg('Loading ASL reference data...');
      await loadReferenceData();  // Fetch sign data from API
    }
  } catch (e) {
    // Backend not available — use built-in fallback data
    console.warn('Backend not available, using fallback data:', e);
    loadFallbackReference();
  }

  updateLoadingMsg('Ready!');

  // Dismiss loading screen after brief delay
  setTimeout(() => {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.opacity = '0';
    overlay.style.pointerEvents = 'none';
    setTimeout(() => { overlay.style.display = 'none'; }, 500);
  }, 800);

  // Set up UI event listeners
  setupEventListeners();

  // Render reference guide
  renderReferenceGrid('alpha');

  // Initial display update
  updateTranslationDisplay();

  console.log('✅ SignBridge App initialized — Session:', STATE.sessionId);
}

function updateLoadingMsg(msg) {
  const el = document.getElementById('loadingMsg');
  if (el) el.textContent = msg;
}


// ═══════════════════════════════════════════════════════════════════
// WEBSOCKET MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

/**
 * Connect to the FastAPI WebSocket endpoint.
 * The WS receives detections + annotated frames in real-time.
 */
function connectWebSocket() {
  if (STATE.ws && STATE.ws.readyState === WebSocket.OPEN) return;

  const url = `${WS_BASE}/ws/stream/${STATE.sessionId}`;
  console.log('🔌 Connecting WebSocket:', url);

  STATE.ws = new WebSocket(url);

  STATE.ws.onopen = () => {
    console.log('✅ WebSocket connected');
    STATE.wsRetries = 0;
    setStatus('active', 'LIVE — Show hand signs to the camera');
    document.getElementById('connectionBadge').textContent = 'LIVE';
    document.getElementById('connectionBadge').classList.add('online');
    document.getElementById('inferenceBadge').style.display = 'flex';
  };

  STATE.ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWSMessage(msg);
    } catch (e) {
      console.error('WS message parse error:', e);
    }
  };

  STATE.ws.onclose = () => {
    console.log('WebSocket closed');
    document.getElementById('connectionBadge').textContent = 'OFFLINE';
    document.getElementById('connectionBadge').classList.remove('online');
    document.getElementById('inferenceBadge').style.display = 'none';

    // Auto-reconnect with exponential backoff
    if (STATE.isRunning && STATE.wsRetries < STATE.maxRetries) {
      const delay = Math.pow(2, STATE.wsRetries) * 1000;  // 1s, 2s, 4s...
      STATE.wsRetries++;
      console.log(`Reconnecting in ${delay}ms (attempt ${STATE.wsRetries})`);
      setTimeout(connectWebSocket, delay);
    }
  };

  STATE.ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    setStatus('error', 'Connection error — retrying...');
  };
}

/**
 * Handle incoming WebSocket messages from the backend.
 * 
 * Message types:
 * - "detection": Main inference result per frame
 * - "pong": Keepalive response
 * - "error": Server-side error
 */
function handleWSMessage(msg) {
  if (msg.type === 'detection') {
    // ── Update FPS & inference time ────────────────────────
    if (msg.fps) {
      document.getElementById('statsFPS').textContent = msg.fps;
    }
    if (msg.inference_ms) {
      STATE.inferenceMs = msg.inference_ms;
      document.getElementById('inferenceMs').textContent = msg.inference_ms;
    }

    // ── Update session stats from backend ──────────────────
    if (msg.session_stats) {
      const s = msg.session_stats;
      document.getElementById('statsDetections').textContent = STATE.detectionCount;
      document.getElementById('aAvgConf').textContent =
        s.avg_conf ? (s.avg_conf * 100).toFixed(0) + '%' : '--';
      document.getElementById('aSPM').textContent = s.spm || '--';
    }

    // ── FIX: Always draw landmarks directly on canvas (no blinking!)
    // We NO LONGER render the annotated frame from backend because:
    // 1. It arrives ~80ms late (encode → send → decode) causing flicker
    // 2. Instead we draw landmarks from JSON data instantly on live video
    // This gives smooth 30fps visuals even if backend is at 12fps
    if (msg.landmarks) {
      drawLandmarks(msg.landmarks);
    }

    // ── Process sign detection ──────────────────────────────
    if (msg.hand_detected && msg.sign) {
      processDetection(msg.sign, msg.confidence || 0.8);
    } else if (!msg.hand_detected) {
      handleNoHand();
    }

  } else if (msg.type === 'pong') {
    // Keepalive acknowledged — do nothing
  } else if (msg.type === 'error') {
    console.error('Server error:', msg.message);
    showToast('⚠️ ' + msg.message);
  }
}

/**
 * Send a keepalive ping every 30 seconds.
 * Prevents the WebSocket from timing out on some hosting providers.
 */
setInterval(() => {
  if (STATE.ws && STATE.ws.readyState === WebSocket.OPEN) {
    STATE.ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 30000);


// ═══════════════════════════════════════════════════════════════════
// CAMERA & SESSION
// ═══════════════════════════════════════════════════════════════════

/**
 * Start the camera and establish WebSocket connection.
 * Called when user clicks "Start" button.
 */
async function startSession() {
  try {
    setStatus('', 'Requesting camera access...');

    // Request camera access
    // ideal constraints — browser will use best available
    STATE.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',        // Front camera
        frameRate: { ideal: 30 }
      }
    });

    const video = document.getElementById('videoEl');
    video.srcObject = STATE.stream;
    video.style.display = 'block';

    // Wait for video metadata to load before starting
    await new Promise(res => video.addEventListener('loadedmetadata', res, { once: true }));

    // Mirror the canvas if mirror toggle is on
    applyMirror();

    // Show camera UI elements
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('hudOverlay').style.display = 'block';
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';

    STATE.isRunning = true;
    STATE.startTime = Date.now();

    // Connect to backend via WebSocket
    connectWebSocket();

    // Start frame capture loop
    // We use requestAnimationFrame for smooth 30fps capture
    startFrameCapture(video);

    // Start session timer
    STATE.sessionInterval = setInterval(updateSessionTimer, 1000);

    setStatus('active', 'LIVE — Show hand signs to the camera');
    showToast('🎥 Camera started! Hold a sign steady to detect');

  } catch (err) {
    console.error('Camera error:', err);
    setStatus('error', 'Camera access denied — check browser permissions');
    showToast('❌ Camera denied. Allow camera in browser settings.');
  }
}

/**
 * Stop camera and disconnect WebSocket.
 */
function stopSession() {
  // Stop all camera tracks
  if (STATE.stream) {
    STATE.stream.getTracks().forEach(t => t.stop());
    STATE.stream = null;
  }

  // Cancel frame capture
  if (STATE.animFrame) {
    cancelAnimationFrame(STATE.animFrame);
    STATE.animFrame = null;
  }

  // Close WebSocket
  if (STATE.ws) {
    STATE.ws.close();
    STATE.ws = null;
  }

  STATE.isRunning = false;

  // Clear timers
  clearInterval(STATE.sessionInterval);
  if (STATE.holdTimer) { clearTimeout(STATE.holdTimer); STATE.holdTimer = null; }
  if (STATE.noHandTimer) { clearTimeout(STATE.noHandTimer); STATE.noHandTimer = null; }

  // Reset UI
  const video = document.getElementById('videoEl');
  video.style.display = 'none';
  video.srcObject = null;

  const canvas = document.getElementById('canvasEl');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  document.getElementById('placeholder').style.display = 'flex';
  document.getElementById('hudOverlay').style.display = 'none';
  document.getElementById('startBtn').style.display = 'block';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('signLetter').textContent = '';
  document.getElementById('statsFPS').textContent = '--';
  document.getElementById('inferenceBadge').style.display = 'none';

  setStatus('', 'Camera stopped — Click Start to resume');
  showToast('Camera stopped');
}


// ═══════════════════════════════════════════════════════════════════
// FRAME CAPTURE
// Reads video frames → encodes as JPEG → sends to backend via WS
// ═══════════════════════════════════════════════════════════════════

// Off-screen canvas used for capturing frames (more efficient)
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

let lastFrameTime = 0;
const TARGET_FPS = 12; // FIX: 12fps to backend prevents WS queue buildup 0026 blinking
const FRAME_INTERVAL = 1000 / TARGET_FPS;

/**
 * Frame capture loop using requestAnimationFrame.
 * RAF syncs to display refresh rate and auto-pauses when tab is hidden.
 * We throttle to TARGET_FPS to limit bandwidth.
 */
function startFrameCapture(video) {
  function loop(timestamp) {
    if (!STATE.isRunning) return;

    STATE.animFrame = requestAnimationFrame(loop);

    // Throttle to TARGET_FPS
    if (timestamp - lastFrameTime < FRAME_INTERVAL) return;
    lastFrameTime = timestamp;

    // Skip if WebSocket isn't ready
    if (!STATE.ws || STATE.ws.readyState !== WebSocket.OPEN) return;

    // Set capture canvas size to match video
    const { videoWidth: w, videoHeight: h } = video;
    if (w === 0 || h === 0) return;

    captureCanvas.width = w;
    captureCanvas.height = h;

    // Draw current video frame to canvas
    captureCtx.drawImage(video, 0, 0, w, h);

    // Encode to JPEG (quality 0.7 = good balance of quality vs size)
    // Lower quality = smaller payload = faster round-trip
    const imageData = captureCanvas.toDataURL('image/jpeg', 0.5); // FIX: lower quality = smaller payload = less lag

    // Send to backend via WebSocket
    STATE.ws.send(JSON.stringify({
      type: 'frame',
      data: imageData
    }));
  }

  STATE.animFrame = requestAnimationFrame(loop);
}


// ═══════════════════════════════════════════════════════════════════
// DETECTION PROCESSING
// Handle incoming sign detections from backend
// ═══════════════════════════════════════════════════════════════════

/**
 * Process a sign detection from the backend.
 * Implements "hold-to-confirm": user must hold sign for holdDuration ms.
 * This prevents accidental/fleeting detections from being added.
 */
function processDetection(sign, confidence) {
  // Update the big sign display in the HUD
  const letterEl = document.getElementById('signLetter');
  const labelEl = document.getElementById('signLabel');

  letterEl.textContent = sign;
  letterEl.style.display = 'block';
  labelEl.textContent = 'HOLD TO CONFIRM';

  // If this is a new sign (different from what we were holding)
  if (sign !== STATE.lastSign) {
    STATE.lastSign = sign;
    STATE.holdStart = Date.now();

    // Clear any existing hold timer
    if (STATE.holdTimer) {
      clearTimeout(STATE.holdTimer);
      STATE.holdTimer = null;
    }

    // Start hold timer — confirm sign after holdDuration ms
    STATE.holdTimer = setTimeout(() => {
      confirmSign(sign, confidence);
      STATE.holdTimer = null;
    }, STATE.holdDuration);

    // Show hold progress bar
    document.getElementById('holdBarWrapper').style.display = 'block';
  }

  // Update hold progress bar in real-time
  if (STATE.holdStart) {
    const progress = Math.min(1, (Date.now() - STATE.holdStart) / STATE.holdDuration);
    document.getElementById('holdBarFill').style.width = (progress * 100) + '%';
  }
}

/**
 * Confirm a sign and add it to the translation.
 * Called after user holds a sign for holdDuration ms.
 */
function confirmSign(sign, confidence) {
  // Cooldown: hard block ALL detections for cooldownMs after each confirm
  const now = Date.now();
  if (now - STATE.lastAdded < STATE.cooldownMs) return;
  STATE.lastAdded = now;
  
  // Reset hold state so same sign can't re-trigger immediately
  STATE.lastSign  = null;
  STATE.holdStart = null;
  if (STATE.holdTimer) { clearTimeout(STATE.holdTimer); STATE.holdTimer = null; }

  // Add to translation
  STATE.translatedText += sign;
  STATE.detectionCount++;

  // Track sign frequency for analytics
  STATE.signHistory[sign] = (STATE.signHistory[sign] || 0) + 1;

  // Update UI
  updateTranslationDisplay();
  updateAnalytics();

  // Sound feedback (optional)
  if (document.getElementById('togSound')?.checked) playBeep();

  // Visual feedback on the sign display
  document.getElementById('signLabel').textContent = '✓ CONFIRMED';
  document.getElementById('holdBarFill').style.width = '100%';

  // Reset hold state
  STATE.lastSign = null;
  STATE.holdStart = null;

  setTimeout(() => {
    document.getElementById('holdBarWrapper').style.display = 'none';
    document.getElementById('holdBarFill').style.width = '0%';
    document.getElementById('signLabel').textContent = 'DETECTING';
  }, 300);
}

/**
 * Called when no hand is detected in frame.
 * Resets hold state and optionally adds a space.
 */
// Fade-out timer for canvas — prevents instant flicker when hand disappears briefly
let _canvasClearTimer = null;

function handleNoHand() {
  // Cancel any pending hold timer
  if (STATE.holdTimer) {
    clearTimeout(STATE.holdTimer);
    STATE.holdTimer = null;
  }

  // FIX: Don't instantly clear canvas — wait 400ms before clearing.
  // This prevents flickering when MediaPipe briefly loses tracking
  // between frames (very common when moving hand slightly).
  if (_canvasClearTimer) clearTimeout(_canvasClearTimer);
  _canvasClearTimer = setTimeout(() => {
    const canvas = document.getElementById('canvasEl');
    if (canvas) {
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }
    _canvasClearTimer = null;
  }, 400);

  // Hide hold progress
  document.getElementById('holdBarWrapper').style.display = 'none';
  document.getElementById('holdBarFill').style.width = '0%';
  document.getElementById('signLetter').textContent = '';
  document.getElementById('signLabel').textContent = 'SHOW HAND';

  STATE.lastSign  = null;
  STATE.holdStart = null;

  // Auto-space: add a space after 1.5s without a hand
  if (document.getElementById('togAutoSpace')?.checked) {
    if (STATE.noHandTimer) clearTimeout(STATE.noHandTimer);
    STATE.noHandTimer = setTimeout(() => {
      if (STATE.translatedText.length > 0 && !STATE.translatedText.endsWith(' ')) {
        addSpace();
      }
      STATE.noHandTimer = null;
    }, 1500);
  }
}


// ═══════════════════════════════════════════════════════════════════
// CANVAS RENDERING
// ═══════════════════════════════════════════════════════════════════

/**
 * Render annotated frame from backend onto the display canvas.
 * The backend (OpenCV) draws landmarks, bounding box, and sign label.
 */
function renderAnnotatedFrame(base64Jpeg) {
  const canvas = document.getElementById('canvasEl');
  const ctx = canvas.getContext('2d');
  const video = document.getElementById('videoEl');

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = 'data:image/jpeg;base64,' + base64Jpeg;
}

/**
 * Fallback: draw hand landmarks on canvas using data from backend.
 * Used when backend doesn't send annotated frames (e.g., lower bandwidth mode).
 */
function drawLandmarks(landmarks) {
  const canvas = document.getElementById('canvasEl');
  const ctx    = canvas.getContext('2d');
  const video  = document.getElementById('videoEl');

  if (!landmarks || !video.videoWidth) return;

  // Match canvas to video size
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  // FIX: Clear ONLY the canvas, not the video — video shows live through it
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const w = canvas.width;
  const h = canvas.height;

  // MediaPipe hand connections
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],           // Thumb
    [5,6],[6,7],[7,8],                 // Index finger
    [9,10],[10,11],[11,12],            // Middle finger
    [13,14],[14,15],[15,16],           // Ring finger
    [17,18],[18,19],[19,20],           // Pinky
    [0,5],[5,9],[9,13],[13,17],[0,17]  // Palm base
  ];

  // ── Draw connection lines (glowing teal) ──────────────────
  ctx.lineWidth   = 2.5;
  ctx.strokeStyle = 'rgba(0, 245, 196, 0.7)';
  ctx.shadowColor = 'rgba(0, 245, 196, 0.4)';
  ctx.shadowBlur  = 6;

  CONNECTIONS.forEach(([a, b]) => {
    const pa = landmarks[a], pb = landmarks[b];
    if (!pa || !pb) return;
    ctx.beginPath();
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
    ctx.stroke();
  });

  // ── Draw landmark dots ────────────────────────────────────
  ctx.shadowBlur = 0;
  const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

  landmarks.forEach((lm, i) => {
    const x      = lm.x * w;
    const y      = lm.y * h;
    const isTip  = FINGERTIPS.has(i);
    const radius = isTip ? 6 : 3.5;

    // Outer glow ring for fingertips
    if (isTip) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 245, 196, 0.15)';
      ctx.fill();
    }

    // Main dot
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isTip ? '#00f5c4' : 'rgba(0, 245, 196, 0.8)';
    ctx.fill();

    // White center for fingertips
    if (isTip) {
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
    }
  });

  // ── Bounding box around hand ──────────────────────────────
  const xs  = landmarks.map(p => p.x * w);
  const ys  = landmarks.map(p => p.y * h);
  const pad = 24;
  const x1  = Math.max(0, Math.min(...xs) - pad);
  const y1  = Math.max(0, Math.min(...ys) - pad);
  const x2  = Math.min(w, Math.max(...xs) + pad);
  const y2  = Math.min(h, Math.max(...ys) + pad);
  const bw  = x2 - x1;
  const bh  = y2 - y1;

  // Dashed bounding box
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = 'rgba(0, 245, 196, 0.5)';
  ctx.lineWidth   = 1.5;
  ctx.strokeRect(x1, y1, bw, bh);
  ctx.setLineDash([]);

  // Corner accents (solid bright corners)
  const cs = 16, ct = 2.5;
  ctx.strokeStyle = '#00f5c4';
  ctx.lineWidth   = ct;
  [ [x1,y1, 1,1], [x2,y1, -1,1], [x1,y2, 1,-1], [x2,y2, -1,-1] ]
    .forEach(([cx,cy,dx,dy]) => {
      ctx.beginPath();
      ctx.moveTo(cx + dx*cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy*cs);
      ctx.stroke();
    });
}


// ═══════════════════════════════════════════════════════════════════
// TRANSLATION TEXT MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

/**
 * Update the translation display with current STATE.translatedText.
 * The last character gets a special animation class.
 */
function updateTranslationDisplay() {
  const el = document.getElementById('translationText');
  const text = STATE.translatedText;

  // Build HTML with animated last character
  let html = '';
  for (let i = 0; i < text.length; i++) {
    const char = text[i] === ' ' ? '&nbsp;' : escapeHtml(text[i]);
    if (i === text.length - 1) {
      html += `<span class="char-new">${char}</span>`;
    } else {
      html += char;
    }
  }

  el.innerHTML = html;
  document.getElementById('charCount').textContent = text.length + ' chars';
  updateWordSuggestions();
  updateWordCount();
}

function addSpace() {
  if (!STATE.translatedText.endsWith(' ')) {
    STATE.translatedText += ' ';
    updateTranslationDisplay();
  }
}

function deleteLast() {
  if (STATE.translatedText.length > 0) {
    STATE.translatedText = STATE.translatedText.slice(0, -1);
    updateTranslationDisplay();
  }
}

function clearText() {
  if (!STATE.translatedText.trim()) return;
  saveToHistory(STATE.translatedText);
  STATE.translatedText = '';
  updateTranslationDisplay();
  showToast('✓ Cleared & saved to history');
}

function copyText() {
  if (!STATE.translatedText.trim()) return;
  navigator.clipboard.writeText(STATE.translatedText.trim())
    .then(() => showToast('📋 Copied to clipboard!'));
}

/**
 * Text-to-Speech using the Web Speech API.
 * Great for the demo — makes the project feel complete.
 */
function speakText() {
  const text = STATE.translatedText.trim();
  if (!text) return;

  // Cancel any ongoing speech
  speechSynthesis.cancel();

  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.9;
  utter.pitch = 1.0;
  utter.volume = 1.0;

  // Use a natural-sounding voice if available
  const voices = speechSynthesis.getVoices();
  const preferred = voices.find(v => v.lang === 'en-US' && v.localService);
  if (preferred) utter.voice = preferred;

  speechSynthesis.speak(utter);
  showToast('🔊 Speaking...');

  utter.onend = () => {
    document.getElementById('statsFPS'); // Just to do something on end
  };
}


// ═══════════════════════════════════════════════════════════════════
// WORD SUGGESTIONS
// Auto-complete based on current partial word being typed
// ═══════════════════════════════════════════════════════════════════

function updateWordSuggestions() {
  const container = document.getElementById('suggestions');
  const words = STATE.translatedText.toUpperCase().trim().split(/\s+/);
  const lastWord = words[words.length - 1] || '';

  if (lastWord.length < 2) {
    container.innerHTML = '';
    return;
  }

  // Find words that START WITH the current partial word
  const matches = WORD_BANK
    .filter(w => w.startsWith(lastWord) && w !== lastWord)
    .slice(0, 5);

  if (matches.length === 0) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = matches.map(word =>
    `<div class="suggestion-chip" onclick="applySuggestion('${word}')">${word}</div>`
  ).join('');
}

function applySuggestion(word) {
  const words = STATE.translatedText.trim().split(/\s+/);
  words[words.length - 1] = word;
  STATE.translatedText = words.join(' ') + ' ';
  updateTranslationDisplay();
  showToast(`✓ ${word}`);
}


// ═══════════════════════════════════════════════════════════════════
// HISTORY
// Stores past translated sentences for review
// ═══════════════════════════════════════════════════════════════════

function saveToHistory(text) {
  if (!text.trim()) return;

  const list = document.getElementById('historyList');
  const empty = list.querySelector('.history-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'history-item';
  item.textContent = text.trim();
  item.title = 'Click to restore';
  item.onclick = () => {
    STATE.translatedText = text.trim() + ' ';
    updateTranslationDisplay();
    showToast('✓ Restored');
  };

  list.insertBefore(item, list.firstChild);

  // Limit history to 15 items
  const items = list.querySelectorAll('.history-item');
  if (items.length > 15) items[items.length - 1].remove();
}

function clearHistory() {
  document.getElementById('historyList').innerHTML =
    '<div class="history-empty">Start translating to build history</div>';
}


// ═══════════════════════════════════════════════════════════════════
// ANALYTICS
// ═══════════════════════════════════════════════════════════════════

function updateAnalytics() {
  document.getElementById('statsDetections').textContent = STATE.detectionCount;

  // Top sign (most frequently detected)
  const topEntry = Object.entries(STATE.signHistory)
    .sort((a, b) => b[1] - a[1])[0];
  document.getElementById('aTopSign').textContent = topEntry ? topEntry[0] : '--';

  // Unique signs count
  document.getElementById('aUnique').textContent = Object.keys(STATE.signHistory).length;
}

function updateWordCount() {
  const words = STATE.translatedText.trim().split(/\s+/).filter(w => w);
  document.getElementById('statsWords').textContent = words.length;
}

function updateSessionTimer() {
  if (!STATE.startTime) return;
  const elapsed = Math.floor((Date.now() - STATE.startTime) / 1000);
  const m = Math.floor(elapsed / 60);
  const s = elapsed % 60;
  document.getElementById('statsSession').textContent =
    `${m}:${s.toString().padStart(2, '0')}`;
}


// ═══════════════════════════════════════════════════════════════════
// ASL REFERENCE GUIDE
// Populated from API data or built-in fallback
// ═══════════════════════════════════════════════════════════════════

// Reference data (loaded from API or fallback)
let REF_DATA = { alphabet: {}, numbers: {} };

/**
 * Load ASL reference data from the backend API.
 * This keeps the frontend in sync with the backend's ground truth.
 */
async function loadReferenceData() {
  try {
    const res = await fetch(`${API_BASE}/api/signs`);
    const data = await res.json();
    REF_DATA = data;
    renderReferenceGrid('alpha');
  } catch (e) {
    console.warn('Could not load reference data from API:', e);
    loadFallbackReference();
  }
}

/**
 * Fallback reference data used when backend is unavailable.
 * Contains descriptions for all 36 signs.
 */
function loadFallbackReference() {
  // Populated from built-in constant
  REF_DATA = FALLBACK_REF;
  renderReferenceGrid('alpha');
}

/**
 * Render the ASL reference grid for the given type ('alpha' or 'numbers').
 * Uses real sign images from ASL handshape image repositories.
 */
function renderReferenceGrid(type) {
  const isAlpha = type === 'alpha';
  const containerId = isAlpha ? 'refAlpha' : 'refNumbers';
  const container = document.getElementById(containerId);

  const data = isAlpha
    ? (REF_DATA.alphabet || FALLBACK_REF.alphabet)
    : (REF_DATA.numbers || FALLBACK_REF.numbers);

  container.innerHTML = Object.entries(data).map(([sign, info]) => {
    // Real ASL sign images from lifeprint.com (free ASL resource)
    // This is the largest free ASL image database available
    const imgUrl = getSignImageUrl(sign);
    const emoji = SIGN_EMOJI[sign] || '✋';
    const difficulty = info.difficulty || 'medium';

    return `
      <div class="ref-card" title="${info.description || ''}">
        <div class="ref-difficulty difficulty-${difficulty}">${difficulty.toUpperCase()}</div>
        <div class="ref-letter">${sign}</div>
        <div class="ref-img-wrapper">
          <img
            class="ref-img"
            src="${imgUrl}"
            alt="ASL sign for ${sign}"
            onerror="this.style.display='none'; this.nextElementSibling.style.display='flex'"
            loading="lazy"
          >
          <div class="ref-img-fallback" style="display:none">${emoji}</div>
        </div>
        <div class="ref-desc">${info.finger_config || info.description || ''}</div>
      </div>
    `;
  }).join('');
}

/**
 * Get URL for ASL sign image.
 * Primary source: lifeprint.com (Dr. Bill Vicars — largest free ASL resource)
 * These are the actual textbook-quality ASL images used in ASL education.
 */
function getSignImageUrl(sign) {
  const s = sign.toLowerCase();
  // lifeprint.com hosts high-quality ASL fingerspelling images
  return `https://www.lifeprint.com/asl101/fingerspelling/abc-gifs/${s}.gif`;
}

function switchRef(btn, type) {
  document.querySelectorAll('.ref-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');

  ['refAlpha', 'refNumbers', 'refTips'].forEach(id => {
    document.getElementById(id).style.display = 'none';
  });

  if (type === 'alpha') {
    document.getElementById('refAlpha').style.display = 'grid';
    renderReferenceGrid('alpha');
  } else if (type === 'numbers') {
    document.getElementById('refNumbers').style.display = 'grid';
    renderReferenceGrid('numbers');
  } else {
    document.getElementById('refTips').style.display = 'grid';
    renderTips();
  }
}

/**
 * Render the tips & tricks grid.
 */
function renderTips() {
  const tips = [
    { icon: '💡', title: 'Lighting is Key', body: 'Use a bright, even light source in front of you. Avoid backlighting — bright background washes out your hand.' },
    { icon: '📐', title: 'Distance Matters', body: 'Keep your hand 40–70cm from the camera. Too close crops landmarks; too far loses accuracy.' },
    { icon: '✋', title: 'Right Hand Preferred', body: 'ASL is typically signed with the dominant hand. Right hand generally gives better detection results.' },
    { icon: '🎯', title: 'Hold Still', body: 'Static signs need to be held for the configured duration (default 1.2s). Keep your hand steady until confirmed.' },
    { icon: '🌈', title: 'Background Contrast', body: 'Plain, neutral backgrounds (white wall, plain desk) improve detection accuracy significantly.' },
    { icon: '⚡', title: 'Adjust Hold Speed', body: 'Use the Hold Duration slider to tune confirmation speed — faster for practice, slower to avoid false detections.' },
    { icon: '🔤', title: 'Word Completion', body: 'Use the auto-suggest chips to complete words faster — tap a suggestion to replace the partial word.' },
    { icon: '🔊', title: 'Speak Output', body: 'After building a sentence, click Speak to have the browser read it aloud using text-to-speech.' },
  ];

  document.getElementById('refTips').innerHTML = tips.map(t => `
    <div class="tip-card">
      <div class="tip-icon">${t.icon}</div>
      <div class="tip-title">${t.title}</div>
      <div class="tip-body">${t.body}</div>
    </div>
  `).join('');
}


// ═══════════════════════════════════════════════════════════════════
// UI UTILITIES
// ═══════════════════════════════════════════════════════════════════

function setMode(mode) {
  STATE.currentMode = mode;
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
  const tab = document.getElementById(`tab-${mode}`);
  if (tab) tab.classList.add('active');
  showToast(`Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`);
}

function toggleSettings() {
  STATE.settingsOpen = !STATE.settingsOpen;
  document.getElementById('settingsPanel').style.display =
    STATE.settingsOpen ? 'block' : 'none';
}

function toggleFullscreen() {
  const wrapper = document.getElementById('videoWrapper');
  if (!document.fullscreenElement) {
    wrapper.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
}

function applyMirror() {
  const mirror = document.getElementById('togMirror')?.checked !== false;
  const video = document.getElementById('videoEl');
  const canvas = document.getElementById('canvasEl');
  const cls = 'mirrored';

  if (mirror) {
    video.classList.add(cls);
    canvas.classList.add(cls);
  } else {
    video.classList.remove(cls);
    canvas.classList.remove(cls);
  }
}

function setStatus(type, text) {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');

  dot.className = 'status-dot' + (type ? ` ${type}` : '');
  txt.textContent = text;
}

function showToast(msg, duration = 2500) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => t.classList.remove('show'), duration);
}

/**
 * Play a subtle confirmation beep using Web Audio API.
 * Much lighter than loading an audio file — pure JS.
 */
function playBeep() {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;   // A5 note
    osc.type = 'sine';
    gain.gain.setValueAtTime(0.25, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.12);
  } catch (e) {
    // AudioContext might be blocked before user gesture
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function generateSessionId() {
  // Crypto-random session ID (used as WebSocket path parameter)
  return 'sb_' + Date.now().toString(36) + '_' +
         Math.random().toString(36).substring(2, 9);
}


// ═══════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════════════════

document.addEventListener('keydown', (e) => {
  // Ctrl+Space: add space
  if (e.code === 'Space' && e.ctrlKey) { e.preventDefault(); addSpace(); }
  // Ctrl+Backspace: delete last character
  if (e.code === 'Backspace' && e.ctrlKey) { e.preventDefault(); deleteLast(); }
  // Ctrl+Shift+C: copy translation
  if (e.code === 'KeyC' && e.ctrlKey && e.shiftKey) { e.preventDefault(); copyText(); }
  // Ctrl+Shift+S: speak translation
  if (e.code === 'KeyS' && e.ctrlKey && e.shiftKey) { e.preventDefault(); speakText(); }
  // Escape: clear translation (with confirmation)
  if (e.code === 'Escape' && e.ctrlKey) { clearText(); }
});


// ═══════════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════════

function setupEventListeners() {
  // Hold duration slider
  const slider = document.getElementById('holdSlider');
  if (slider) {
    slider.addEventListener('input', (e) => {
      STATE.holdDuration = parseFloat(e.target.value) * 1000;
      document.getElementById('holdVal').textContent = e.target.value + 's';
    });
  }

  // Mirror toggle
  document.getElementById('togMirror')?.addEventListener('change', applyMirror);

  // Skeleton toggle — send to backend via WebSocket
  document.getElementById('togSkeleton')?.addEventListener('change', (e) => {
    if (STATE.ws?.readyState === WebSocket.OPEN) {
      STATE.ws.send(JSON.stringify({
        type: 'settings',
        data: { show_landmarks: e.target.checked }
      }));
    }
  });
}


// ═══════════════════════════════════════════════════════════════════
// FALLBACK REFERENCE DATA
// Used when backend API is unavailable
// ═══════════════════════════════════════════════════════════════════

const FALLBACK_REF = {
  alphabet: {
    'A': { finger_config: 'Fist, thumb beside index', difficulty: 'easy' },
    'B': { finger_config: 'Flat hand, thumb tucked', difficulty: 'easy' },
    'C': { finger_config: 'Curved C shape', difficulty: 'easy' },
    'D': { finger_config: 'Index up, others circle', difficulty: 'medium' },
    'E': { finger_config: 'All fingers bent, claw', difficulty: 'medium' },
    'F': { finger_config: 'Index+thumb circle, others up', difficulty: 'medium' },
    'G': { finger_config: 'Index points sideways', difficulty: 'medium' },
    'H': { finger_config: 'Index+middle horizontal', difficulty: 'medium' },
    'I': { finger_config: 'Only pinky up', difficulty: 'easy' },
    'J': { finger_config: 'Pinky up, trace J', difficulty: 'hard' },
    'K': { finger_config: 'Index+middle V, thumb between', difficulty: 'hard' },
    'L': { finger_config: 'Index up, thumb out — L shape', difficulty: 'easy' },
    'M': { finger_config: '3 fingers over thumb', difficulty: 'hard' },
    'N': { finger_config: '2 fingers over thumb', difficulty: 'hard' },
    'O': { finger_config: 'All fingers form O', difficulty: 'easy' },
    'P': { finger_config: 'K-shape pointing down', difficulty: 'hard' },
    'Q': { finger_config: 'G-shape pointing down', difficulty: 'hard' },
    'R': { finger_config: 'Crossed index + middle', difficulty: 'medium' },
    'S': { finger_config: 'Fist, thumb over fingers', difficulty: 'easy' },
    'T': { finger_config: 'Thumb between index+middle', difficulty: 'medium' },
    'U': { finger_config: 'Index+middle together up', difficulty: 'easy' },
    'V': { finger_config: 'Peace sign, fingers spread', difficulty: 'easy' },
    'W': { finger_config: '3 fingers spread up', difficulty: 'medium' },
    'X': { finger_config: 'Hooked index finger', difficulty: 'medium' },
    'Y': { finger_config: 'Thumb+pinky out (shaka)', difficulty: 'easy' },
    'Z': { finger_config: 'Index traces Z in air', difficulty: 'hard' },
  },
  numbers: {
    '0': { finger_config: 'Fingers pinch to thumb', difficulty: 'easy' },
    '1': { finger_config: 'Index finger up only', difficulty: 'easy' },
    '2': { finger_config: 'Index + middle up', difficulty: 'easy' },
    '3': { finger_config: 'Thumb + index + middle', difficulty: 'easy' },
    '4': { finger_config: '4 fingers, thumb tucked', difficulty: 'easy' },
    '5': { finger_config: 'All 5 fingers spread', difficulty: 'easy' },
    '6': { finger_config: 'Pinky+thumb touch, others up', difficulty: 'medium' },
    '7': { finger_config: 'Ring+thumb touch, others up', difficulty: 'medium' },
    '8': { finger_config: 'Middle+thumb touch, others up', difficulty: 'medium' },
    '9': { finger_config: 'Index+thumb circle (OK sign)', difficulty: 'easy' },
  }
};