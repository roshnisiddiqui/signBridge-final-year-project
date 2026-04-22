"""
╔══════════════════════════════════════════════════════════════════╗
║   session_manager.py — Session Tracking & Analytics             ║
╚══════════════════════════════════════════════════════════════════╝

WHY a SessionManager?
  - Tracks every detection per WebSocket connection
  - Enables session analytics (words per minute, accuracy, etc.)
  - Provides history for the frontend's translation log
  - In production: would persist to a database (PostgreSQL/Redis)
  
For the project: we use in-memory storage.
For production: swap the dict for a Redis client or SQLAlchemy session.
"""

import time
import logging
from datetime import datetime
from typing import Optional
from collections import Counter, deque

logger = logging.getLogger("signbridge.session")


class Session:
    """
    Represents one user session (one WebSocket connection lifecycle).
    
    Tracks:
    - All detected signs with timestamps
    - Confidence scores
    - Words formed
    - Session duration
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_active = time.time()

        # Raw detection log: list of {"sign": "A", "confidence": 0.9, "time": 1234}
        self.detections = []

        # Confidence scores for averaging
        self.confidence_scores = deque(maxlen=1000)  # Rolling window

        # Formed text (mirrors frontend's translation buffer)
        self.formed_text = ""

        # Stats cache
        self._words_cache = 0

        logger.info(f"Session created: {session_id}")

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return time.time() - self.created_at

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def unique_signs(self) -> list:
        return list(set(d["sign"] for d in self.detections))

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def signs_per_minute(self) -> float:
        """Detection rate — useful for performance metric on the frontend."""
        if self.duration < 1:
            return 0.0
        return (self.detection_count / self.duration) * 60

    def log_detection(self, sign: str, confidence: float):
        """Record a single confirmed detection."""
        self.last_active = time.time()
        self.detections.append({
            "sign": sign,
            "confidence": confidence,
            "timestamp": time.time(),
            "time_str": datetime.now().strftime("%H:%M:%S")
        })
        self.confidence_scores.append(confidence)
        logger.debug(f"[{self.session_id}] Detected: {sign} ({confidence:.2f})")

    def get_stats(self) -> dict:
        """Return full statistics for this session."""
        # Count sign frequencies
        sign_counts = Counter(d["sign"] for d in self.detections)

        # Calculate words (space-separated sequences in formed_text)
        words = [w for w in self.formed_text.strip().split() if w]

        return {
            "session_id": self.session_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "duration_seconds": round(self.duration, 1),
            "total_detections": self.detection_count,
            "unique_signs": self.unique_signs,
            "most_common_signs": sign_counts.most_common(5),
            "avg_confidence": round(self.avg_confidence, 3),
            "signs_per_minute": round(self.signs_per_minute, 1),
            "words_formed": len(words),
            "formed_text": self.formed_text,
            "recent_detections": self.detections[-20:],  # Last 20
        }

    def get_quick_stats(self) -> dict:
        """Lightweight stats sent with every WebSocket frame."""
        return {
            "detections": self.detection_count,
            "words": len([w for w in self.formed_text.strip().split() if w]),
            "avg_conf": round(self.avg_confidence, 2),
            "spm": round(self.signs_per_minute, 1),
        }


class SessionManager:
    """
    Manages all active sessions in memory.
    
    In production, this would interface with:
    - Redis for fast in-memory sessions
    - PostgreSQL for persistent history
    - JWT for authentication
    
    For this project: simple dict-based in-memory store.
    """

    def __init__(self):
        # session_id → Session object
        self._sessions: dict[str, Session] = {}

        # Max sessions to prevent memory leaks
        self._max_sessions = 100

        logger.info("SessionManager initialized")

    def create_session(self, session_id: str) -> Session:
        """
        Create a new session or reset an existing one.
        Called when a WebSocket connection is established.
        """
        # Clean up old sessions if at limit
        if len(self._sessions) >= self._max_sessions:
            self._evict_oldest()

        session = Session(session_id)
        self._sessions[session_id] = session
        return session

    def log_detection(self, session_id: str, sign: str, confidence: float):
        """Log a detection to the appropriate session."""
        session = self._sessions.get(session_id)
        if session:
            session.log_detection(sign, confidence)

    def update_text(self, session_id: str, text: str):
        """Update the formed text for a session (synced from frontend)."""
        session = self._sessions.get(session_id)
        if session:
            session.formed_text = text

    def get_stats(self, session_id: str) -> Optional[dict]:
        """Return full stats for a session."""
        session = self._sessions.get(session_id)
        return session.get_stats() if session else None

    def get_quick_stats(self, session_id: str) -> dict:
        """Return quick stats for real-time WebSocket responses."""
        session = self._sessions.get(session_id)
        return session.get_quick_stats() if session else {}

    def clear(self, session_id: str):
        """Remove a session."""
        self._sessions.pop(session_id, None)
        logger.info(f"Session cleared: {session_id}")

    def _evict_oldest(self):
        """Remove the oldest session to free memory."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda k: self._sessions[k].created_at)
        del self._sessions[oldest_id]
        logger.info(f"Evicted oldest session: {oldest_id}")

    def active_count(self) -> int:
        return len(self._sessions)