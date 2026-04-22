"""
SignBridge — FastAPI Backend v5
Run: python main.py → open http://localhost:8000
"""
import asyncio, base64, json, logging, time, os
from datetime import datetime
import cv2, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gesture_engine import GestureEngine
from session_manager import SessionManager
from nlp_engine import get_nlp_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("signbridge")

app = FastAPI(title="SignBridge API v5", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
STATIC_DIR   = os.path.join(FRONTEND_DIR, "static")
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates     = Jinja2Templates(directory=TEMPLATE_DIR)
gesture_engine = GestureEngine()
session_manager = SessionManager()
nlp            = get_nlp_engine()

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "5.0.0", "timestamp": datetime.now().isoformat()}

@app.get("/api/signs")
async def get_signs():
    return {"alphabet": gesture_engine.get_alphabet_reference(), "numbers": gesture_engine.get_numbers_reference()}

@app.get("/api/signs/{sign}")
async def get_sign(sign: str):
    d = gesture_engine.get_sign_detail(sign.upper())
    if not d: raise HTTPException(404, f"Sign '{sign}' not found")
    return d

@app.get("/api/nlp/suggest")
async def nlp_suggest(text: str = "", partial: str = ""):
    """NLP suggestion endpoint — returns completions + next words."""
    return nlp.get_smart_suggestions(text, partial)

@app.get("/api/nlp/correct")
async def nlp_correct(sentence: str = ""):
    return {"original": sentence, "corrected": nlp.correct_sentence(sentence)}

@app.post("/api/nlp/word-confirmed")
async def nlp_word_confirmed(request: Request):
    body = await request.json()
    word = body.get("word", "")
    if word: nlp.record_word(word)
    return {"ok": True}

@app.get("/api/session/{session_id}/stats")
async def get_stats(session_id: str):
    s = session_manager.get_stats(session_id)
    if not s: raise HTTPException(404, "Session not found")
    return s

@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WS connected: {session_id}")
    session_manager.create_session(session_id)
    frame_count = 0; last_fps_time = time.time(); fps = 0.0

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "frame":
                img_bytes = base64.b64decode(msg["data"].split(",")[-1])
                frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue

                t0     = time.perf_counter()
                result = gesture_engine.process_frame(frame)
                inf_ms = round((time.perf_counter() - t0) * 1000, 1)

                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps = round(frame_count / (now - last_fps_time), 1)
                    frame_count = 0; last_fps_time = now

                if result.get("sign"):
                    session_manager.log_detection(session_id, result["sign"], result["confidence"])

                await websocket.send_text(json.dumps({
                    "type":          "detection",
                    "sign":          result.get("sign"),
                    "confidence":    round(result.get("confidence", 0.0), 3),
                    "hand_detected": result.get("hand_detected", False),
                    "landmarks":     result.get("landmarks_json"),
                    "fps":           fps,
                    "inference_ms":  inf_ms,
                    "session_stats": session_manager.get_quick_stats(session_id)
                }))

            elif msg.get("type") == "nlp_request":
                # Frontend asks for suggestions mid-typing
                suggestions = nlp.get_smart_suggestions(
                    msg.get("text", ""), msg.get("partial", "")
                )
                await websocket.send_text(json.dumps({"type": "nlp_response", **suggestions}))

            elif msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WS error: {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except: pass

@app.on_event("startup")
async def startup():
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    gesture_engine.process_frame(dummy)
    logger.info("✅ SignBridge v5 ready — http://localhost:8000")
    logger.info("📖 Docs: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown():
    gesture_engine.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")