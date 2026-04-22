# ✋ SignBridge — Real-Time ASL Sign Language Translator

> **Final Year B.Tech Project** | AI / Deep Learning / Computer Vision  
> Built with Python · FastAPI · Google MediaPipe · WebSocket · OpenCV

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-FF6F00?style=flat&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat&logo=opencv&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-Realtime-green?style=flat)

---

## 🌟 What It Does

SignBridge translates American Sign Language (ASL) hand gestures into text in **real-time** using your webcam. No pre-recorded videos, no cloud processing — everything runs locally at ~30 FPS.

### ✅ Capabilities
| Feature | Detail |
|---|---|
| **Alphabet** | All 26 ASL letters (A–Z) |
| **Numbers** | 0–9 |
| **Word Building** | Letters stack into words with auto-space |
| **Sentence Mode** | Build full sentences, copy or speak them |
| **Text-to-Speech** | Web Speech API reads output aloud |
| **Word Suggestions** | Auto-completes partial words |
| **Session Analytics** | FPS, confidence, signs/min, history |
| **Live API Docs** | Swagger UI at `/docs` (great for demos) |
| **WebSocket** | ~30 FPS real-time bidirectional stream |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                │
│  HTML + CSS + Vanilla JS                             │
│  • Camera access (getUserMedia API)                  │
│  • Frame capture (Canvas → base64 JPEG)              │
│  • WebSocket client                                  │
│  • UI updates & translation display                  │
└──────────────────┬───────────────────────────────────┘
                   │  WebSocket (ws://localhost:8000)
                   │  ~20 JPEG frames/second
                   ▼
┌──────────────────────────────────────────────────────┐
│                    BACKEND (Python)                  │
│  FastAPI + Uvicorn ASGI Server                       │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │         GestureEngine                        │    │
│  │  1. Decode base64 → OpenCV BGR frame         │    │
│  │  2. BGR → RGB conversion                     │    │
│  │  3. MediaPipe Hands inference                │    │
│  │     → 21 3D hand landmarks                  │    │
│  │  4. Geometric classifier                     │    │
│  │     → Finger states (up/down/bent)           │    │
│  │     → Normalized distance ratios             │    │
│  │     → Joint angle calculations               │    │
│  │  5. Temporal smoothing (5-frame window)      │    │
│  │  6. Draw annotations (OpenCV)                │    │
│  │  7. Encode annotated frame → base64          │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │         SessionManager                       │    │
│  │  • Tracks detections per session             │    │
│  │  • Calculates analytics (SPM, confidence)    │    │
│  │  • Stores history                            │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### Why These Technologies?

| Technology | Why Chosen |
|---|---|
| **FastAPI** | Async-native Python framework; auto-generates Swagger docs; built-in WebSocket; 2-3x faster than Flask; industry standard for ML APIs |
| **MediaPipe** | Google's pre-trained hand landmark model; 21 3D keypoints per hand; works on CPU; no training data needed; production-grade accuracy |
| **WebSocket** | ~10x lower latency than HTTP polling; essential for 30 FPS real-time video; bidirectional communication |
| **OpenCV** | Industry-standard computer vision library; GPU-optional image processing; draws annotations efficiently |
| **NumPy** | Efficient array math for image data; used for coordinate normalization in classifier |
| **Geometric Classifier** | Fully explainable (every decision traceable); no dataset needed; works on CPU; can be extended with ML later |

---

## 📁 Project Structure

```
signbridge/
├── backend/
│   ├── main.py              # FastAPI app, WebSocket endpoint, HTTP routes
│   ├── gesture_engine.py    # MediaPipe + ASL geometric classifier
│   ├── session_manager.py   # Session tracking & analytics
│   └── requirements.txt     # Python dependencies
│
├── frontend/
│   ├── templates/
│   │   └── index.html       # Main Jinja2 template
│   └── static/
│       ├── css/
│       │   └── style.css    # All styles (dark futuristic theme)
│       └── js/
│           └── app.js       # Frontend app logic (camera, WS, UI)
│
├── docs/
│   └── architecture.md      # Detailed architecture documentation
│
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Webcam / camera

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/signbridge.git
cd signbridge/backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
# OR
uvicorn main:app --reload --port 8000
```

### 3. Open in Browser

```
http://localhost:8000
```

### 4. Interactive API Docs (Swagger)

```
http://localhost:8000/docs
```

This is a great thing to show your professor — auto-generated, interactive API documentation!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve frontend |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/signs` | All ASL reference data |
| `GET` | `/api/signs/{sign}` | Details for specific sign |
| `POST` | `/api/detect/frame` | Single-frame detection (REST) |
| `GET` | `/api/session/{id}/stats` | Session analytics |
| `WS` | `/ws/stream/{session_id}` | Real-time video stream |

---

## 🧠 How the Classifier Works

The core ML pipeline uses **geometric analysis** on MediaPipe's 21 hand landmarks:

```
Landmark → Finger State → Distance Ratios → Angle Calc → Sign Label
```

### Finger State Detection
```python
# A finger is "up" if its TIP is above its PIP joint
# (Y increases downward in image coordinates)
index_up = lm[INDEX_TIP].y < lm[INDEX_PIP].y
```

### Scale-Invariant Distance Ratios
```python
# Normalize all distances by palm size
# This works for big hands AND small hands at any distance
palm_size = dist(lm[WRIST], lm[MIDDLE_MCP])
thumb_index_ratio = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm_size
```

### Temporal Smoothing (Anti-Flicker)
```python
# Store last 5 predictions and return majority vote
# [A, A, B, A, A] → A (prevents single-frame glitches)
buffer.append(raw_prediction)
return Counter(buffer).most_common(1)[0][0]
```

---

## 🌐 Deployment

### Option 1: Railway (Recommended — Free tier available)
```bash
railway login
railway init
railway up
```

### Option 2: Render.com
Add `render.yaml`:
```yaml
services:
  - type: web
    name: signbridge
    env: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### Option 3: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🎓 Academic Context

### Problem Statement
Deaf and hard-of-hearing individuals face significant communication barriers. Existing translation tools are either expensive, require specialized hardware, or have significant latency. This project demonstrates a software-only, real-time solution accessible via any modern browser.

### Technical Contributions
1. **Zero-training architecture**: Uses geometric feature extraction instead of a trained neural network, making the system deployable without any dataset collection or GPU training
2. **Scale-invariant features**: Normalizing distances by palm size makes detection robust to varying hand sizes and distances
3. **Temporal smoothing**: 5-frame voting window eliminates detection noise without adding perceptible latency
4. **Full-stack ML integration**: End-to-end pipeline from raw video to structured API responses with session analytics

### Potential Enhancements (Future Work)
- Train a small CNN/LSTM on top of MediaPipe features for motion signs (J, Z)
- Add two-hand gesture support (many ASL words use both hands)
- Integrate a language model for contextual word prediction
- Build a mobile app (React Native + Python backend)
- Add support for other sign languages (BSL, ISL, etc.)

---

## 🛠️ Tech Stack Summary

```
Backend          Frontend         ML/CV            Infrastructure
──────────       ────────         ─────            ──────────────
Python 3.10+     HTML5            MediaPipe Hands  Uvicorn (ASGI)
FastAPI          CSS3             OpenCV 4.9       WebSocket
Pydantic         Vanilla JS       NumPy            Jinja2 (templates)
```

---

*Built for Final Year B.Tech Project — Computer Science / AI Track*