"""
╔══════════════════════════════════════════════════════════════════╗
║  gesture_engine.py  —  ASL Classifier v6 "Score & Vote"         ║
╚══════════════════════════════════════════════════════════════════╝

ROOT CAUSE OF ALL PREVIOUS FAILURES:
  Hard if/else thresholds fail because:
  - Different people's hands have different proportions
  - Camera distance changes all normalized distances
  - A sign that barely misses one threshold = detected as something else

NEW APPROACH — Candidate Scoring:
  Every sign defines a score() function.
  We compute scores for ALL signs simultaneously.
  The HIGHEST scoring sign wins.
  A sign must score > 0.60 to be reported (confidence gate).
  This means: even if our thresholds are slightly off, the RIGHT sign
  still wins because it scores HIGHER than wrong ones.

LANDMARK INDICES (MediaPipe 21-point model):
  0=WRIST
  1-4:  THUMB  (CMC, MCP, IP, TIP)
  5-8:  INDEX  (MCP, PIP, DIP, TIP)
  9-12: MIDDLE (MCP, PIP, DIP, TIP)
 13-16: RING   (MCP, PIP, DIP, TIP)
 17-20: PINKY  (MCP, PIP, DIP, TIP)
"""

import math, logging, time
from typing import Optional
from collections import Counter
import cv2, mediapipe as mp, numpy as np

logger = logging.getLogger("signbridge.engine")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Landmark index constants
W  = 0
T_CMC,T_MCP,T_IP,T_TIP   = 1,2,3,4
I_MCP,I_PIP,I_DIP,I_TIP   = 5,6,7,8
M_MCP,M_PIP,M_DIP,M_TIP   = 9,10,11,12
R_MCP,R_PIP,R_DIP,R_TIP   = 13,14,15,16
P_MCP,P_PIP,P_DIP,P_TIP   = 17,18,19,20

ASL_REFERENCE = {
    "alphabet": {
        "A":{"description":"Fist, thumb beside index","difficulty":"easy"},
        "B":{"description":"4 fingers flat together, thumb bent in","difficulty":"easy"},
        "C":{"description":"Curved C shape like holding a ball","difficulty":"easy"},
        "D":{"description":"Index up, others curl to thumb forming D","difficulty":"medium"},
        "E":{"description":"All fingers bent forward, nails face viewer","difficulty":"medium"},
        "F":{"description":"Index+thumb circle, middle+ring+pinky up","difficulty":"medium"},
        "G":{"description":"Index+thumb point sideways horizontally","difficulty":"medium"},
        "H":{"description":"Index+middle extended sideways together","difficulty":"medium"},
        "I":{"description":"Only pinky up, ALL others tightly curled","difficulty":"easy"},
        "J":{"description":"Pinky up, trace J in air (motion sign)","difficulty":"hard"},
        "K":{"description":"V shape with thumb up between fingers","difficulty":"hard"},
        "L":{"description":"Index up + thumb out = 90° L shape","difficulty":"easy"},
        "M":{"description":"3 fingers (index+mid+ring) over tucked thumb","difficulty":"hard"},
        "N":{"description":"2 fingers (index+mid) over tucked thumb","difficulty":"hard"},
        "O":{"description":"All fingertips meet thumb tip forming oval","difficulty":"easy"},
        "P":{"description":"K shape rotated pointing downward","difficulty":"hard"},
        "Q":{"description":"G shape rotated pointing downward","difficulty":"hard"},
        "R":{"description":"Index and middle crossed over each other","difficulty":"medium"},
        "S":{"description":"Fist, thumb rests OVER index+middle knuckles","difficulty":"easy"},
        "T":{"description":"Fist, thumb peeks between index and middle","difficulty":"medium"},
        "U":{"description":"Index+middle up and pressed together","difficulty":"easy"},
        "V":{"description":"Index+middle up and spread apart (peace)","difficulty":"easy"},
        "W":{"description":"Index+middle+ring all up and spread","difficulty":"medium"},
        "X":{"description":"Index hooked/bent like a hook, others fist","difficulty":"medium"},
        "Y":{"description":"Thumb out + pinky up (shaka/hang-loose)","difficulty":"easy"},
        "Z":{"description":"Index traces Z in air (motion sign)","difficulty":"hard"},
    },
    "numbers": {
        "0":{"description":"All fingertips pinch to thumb, tight O","difficulty":"easy"},
        "1":{"description":"Only index up, thumb folded","difficulty":"easy"},
        "2":{"description":"Index + middle up","difficulty":"easy"},
        "3":{"description":"Thumb + index + middle out","difficulty":"easy"},
        "4":{"description":"4 fingers up, thumb tucked in","difficulty":"easy"},
        "5":{"description":"All 5 fingers spread open","difficulty":"easy"},
        "6":{"description":"Pinky+thumb touch, other 3 up","difficulty":"medium"},
        "7":{"description":"Ring+thumb touch, other 3 up","difficulty":"medium"},
        "8":{"description":"Middle+thumb touch, other 3 up","difficulty":"medium"},
        "9":{"description":"Index+thumb circle (OK), others up","difficulty":"easy"},
    }
}


# ══════════════════════════════════════════════════════════════════
class Hand:
    """
    Wrapper around MediaPipe landmarks with computed features.
    Pre-computes everything once so classifiers run fast.
    """
    def __init__(self, lm):
        self.lm = lm

        # Palm scale — wrist to middle MCP
        self.palm = max(self._d(W, M_MCP), 0.001)

        # ── Finger UP: tip clearly above PIP ──────────────────────
        # Threshold 0.04*palm makes it robust vs tiny noise
        thr = 0.04
        self.i_up = (lm[I_PIP].y - lm[I_TIP].y) / self.palm > thr
        self.m_up = (lm[M_PIP].y - lm[M_TIP].y) / self.palm > thr
        self.r_up = (lm[R_PIP].y - lm[R_TIP].y) / self.palm > thr
        self.p_up = (lm[P_PIP].y - lm[P_TIP].y) / self.palm > thr

        # ── Finger PARTIALLY up (for curl detection) ───────────────
        thr2 = 0.01
        self.i_any = (lm[I_PIP].y - lm[I_TIP].y) / self.palm > thr2
        self.m_any = (lm[M_PIP].y - lm[M_TIP].y) / self.palm > thr2
        self.r_any = (lm[R_PIP].y - lm[R_TIP].y) / self.palm > thr2
        self.p_any = (lm[P_PIP].y - lm[P_TIP].y) / self.palm > thr2

        # ── Thumb extension (KEY feature) ─────────────────────────
        # Distance from thumb tip to index MCP, normalized by palm
        # I sign:  thumb pressed against palm   → ~0.15–0.30
        # Y sign:  thumb spread (shaka)         → ~0.45–0.70
        # L sign:  thumb out sideways            → ~0.50–0.70
        # A sign:  thumb beside index (partial) → ~0.25–0.40
        # S sign:  thumb over fingers (folded)  → ~0.10–0.25
        self.thumb_ext = self._d(T_TIP, I_MCP) / self.palm

        # ── Thumb direction (X axis comparison) ───────────────────
        # In mirrored camera, right-hand thumb points LEFT = smaller x
        # But this varies — use as secondary signal only
        self.thumb_left  = lm[T_TIP].x < lm[T_MCP].x - 0.01
        self.thumb_right = lm[T_TIP].x > lm[T_MCP].x + 0.01

        # ── Tip-to-tip distances (normalized) ─────────────────────
        self.ti = self._d(T_TIP, I_TIP) / self.palm   # thumb↔index tip
        self.tm = self._d(T_TIP, M_TIP) / self.palm   # thumb↔middle tip
        self.tr = self._d(T_TIP, R_TIP) / self.palm   # thumb↔ring tip
        self.tp = self._d(T_TIP, P_TIP) / self.palm   # thumb↔pinky tip
        self.im = self._d(I_TIP, M_TIP) / self.palm   # index↔middle tip
        self.mr = self._d(M_TIP, R_TIP) / self.palm   # middle↔ring tip

        # ── Curl scores ────────────────────────────────────────────
        # 0.0 = straight, 1.0 = fully curled
        self.ic = self._curl(I_MCP, I_PIP, I_TIP)
        self.mc = self._curl(M_MCP, M_PIP, M_TIP)
        self.rc = self._curl(R_MCP, R_PIP, R_TIP)
        self.pc = self._curl(P_MCP, P_PIP, P_TIP)

        # ── Gap for O detection ────────────────────────────────────
        # Distance from index MCP (base knuckle) to thumb tip
        # Large = thumb not covering palm = O/C gap visible
        self.gap = self._d(I_MCP, T_TIP) / self.palm

        # ── Finger angle (for G/H horizontal detection) ────────────
        self.index_angle = abs(math.atan2(
            lm[I_TIP].y - lm[I_MCP].y,
            lm[I_TIP].x - lm[I_MCP].x
        ))  # 0 = horizontal, π/2 = vertical

        # ── Up count ──────────────────────────────────────────────
        self.up4 = sum([self.i_up, self.m_up, self.r_up, self.p_up])

    def _d(self, a, b):
        lm = self.lm
        return math.sqrt((lm[a].x-lm[b].x)**2 + (lm[a].y-lm[b].y)**2)

    def _curl(self, mcp, pip, tip):
        lm = self.lm
        v1 = (lm[pip].x-lm[mcp].x, lm[pip].y-lm[mcp].y)
        v2 = (lm[tip].x-lm[pip].x, lm[tip].y-lm[pip].y)
        dot = v1[0]*v2[0]+v1[1]*v2[1]
        m1  = math.sqrt(v1[0]**2+v1[1]**2)
        m2  = math.sqrt(v2[0]**2+v2[1]**2)
        if m1*m2 < 0.0001: return 0.0
        cos = max(-1.0, min(1.0, dot/(m1*m2)))
        return 1.0 - (math.acos(cos)/math.pi)


# ══════════════════════════════════════════════════════════════════
# SIGN SCORE FUNCTIONS
# Each returns a float 0.0–1.0 representing how well the hand matches.
# Higher = better match. Above 0.60 = confident detection.
# ══════════════════════════════════════════════════════════════════

def score_A(h: Hand) -> float:
    """Fist, thumb BESIDE index (not over fingers)"""
    s = 0.0
    if not h.i_up: s += 0.25
    if not h.m_up: s += 0.25
    if not h.r_up: s += 0.20
    if not h.p_up: s += 0.15
    # Thumb beside = moderate extension, not too folded
    if 0.20 < h.thumb_ext < 0.50: s += 0.20
    # Thumb tip at roughly same height as index PIP (beside, not over)
    ty = h.lm[T_TIP].y; iy = h.lm[I_PIP].y
    if abs(ty - iy) < 0.10: s += 0.10
    return min(s, 1.0)

def score_B(h: Hand) -> float:
    """4 fingers flat and tight together, thumb bent across palm"""
    s = 0.0
    if h.i_up: s += 0.22
    if h.m_up: s += 0.22
    if h.r_up: s += 0.22
    if h.p_up: s += 0.20
    if h.thumb_ext < 0.30: s += 0.20  # Thumb folded in
    if h.im < 0.18: s += 0.10         # Fingers close together
    return min(s, 1.0)

def score_C(h: Hand) -> float:
    """Curved C shape — all fingers curved, thumb curves to meet them"""
    s = 0.0
    # All fingers partially bent but NOT fully down
    if 0.20 < h.ic < 0.65: s += 0.18
    if 0.20 < h.mc < 0.65: s += 0.18
    if 0.20 < h.rc < 0.65: s += 0.15
    # Thumb out making the C gap
    if 0.30 < h.ti < 0.70: s += 0.20
    if h.thumb_ext > 0.30: s += 0.15
    # Fingers NOT fully up
    if not h.i_up and not h.m_up: s += 0.14
    return min(s, 1.0)

def score_D(h: Hand) -> float:
    """Index up, middle/ring/pinky curl toward thumb forming circle"""
    s = 0.0
    if h.i_up:     s += 0.35
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.15
    if h.tm < 0.30: s += 0.20  # Middle tip near thumb = circle formed
    if not (h.thumb_ext > 0.45): s += 0.10  # Thumb not fully extended like L
    return min(s, 1.0)

def score_E(h: Hand) -> float:
    """All fingers bent at first knuckle, deeply curled, nails forward"""
    s = 0.0
    if h.ic > 0.45: s += 0.25
    if h.mc > 0.45: s += 0.25
    if h.rc > 0.40: s += 0.20
    if h.pc > 0.35: s += 0.15
    # Tips NOT near thumb (unlike O)
    if h.ti > 0.22: s += 0.10
    if not h.i_up and not h.m_up: s += 0.05
    return min(s, 1.0)

def score_F(h: Hand) -> float:
    """Index+thumb circle (OK sign), middle+ring+pinky pointing up"""
    s = 0.0
    if not h.i_up: s += 0.25   # Index curled down to meet thumb
    if h.m_up:     s += 0.25
    if h.r_up:     s += 0.25
    if h.p_up:     s += 0.15
    if h.ti < 0.25: s += 0.20  # Index tip near thumb = circle
    return min(s, 1.0)

def score_G(h: Hand) -> float:
    """Index + thumb both pointing SIDEWAYS (horizontal)"""
    s = 0.0
    if h.i_up: s += 0.15       # index "up" loosely (it's sideways)
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.20
    if not h.p_up: s += 0.15
    # KEY: index is pointing SIDEWAYS not upward
    if h.index_angle < 0.70: s += 0.30   # Nearly horizontal
    if h.thumb_ext > 0.30: s += 0.10
    return min(s, 1.0)

def score_H(h: Hand) -> float:
    """Index + middle extended SIDEWAYS together (horizontal)"""
    s = 0.0
    if h.i_up: s += 0.20
    if h.m_up: s += 0.20
    if not h.r_up: s += 0.20
    if not h.p_up: s += 0.15
    # KEY difference from U/V: fingers point sideways
    mid_angle = abs(math.atan2(
        h.lm[M_TIP].y - h.lm[M_MCP].y,
        h.lm[M_TIP].x - h.lm[M_MCP].x
    ))
    if h.index_angle < 0.65: s += 0.15
    if mid_angle < 0.65:     s += 0.15
    if h.im < 0.20: s += 0.10  # Together (not spread like V)
    return min(s, 1.0)

def score_I(h: Hand) -> float:
    """ONLY pinky up. Thumb TIGHTLY pressed against palm (not extended at all)"""
    s = 0.0
    if h.p_up:     s += 0.35   # Pinky up = required
    if not h.i_up: s += 0.22
    if not h.m_up: s += 0.18
    if not h.r_up: s += 0.15
    # CRITICAL: thumb must be clearly NOT extended
    # This is what distinguishes I from Y
    if h.thumb_ext < 0.32: s += 0.15
    if h.thumb_ext < 0.25: s += 0.10  # Extra bonus for very folded thumb
    return min(s, 1.0)

def score_K(h: Hand) -> float:
    """V shape (index+middle up) with thumb pointing UP between them"""
    s = 0.0
    if h.i_up: s += 0.25
    if h.m_up: s += 0.25
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.15
    # Thumb tip is ABOVE the MCP knuckles (pointing up)
    if h.lm[T_TIP].y < h.lm[I_MCP].y: s += 0.20
    if h.thumb_ext > 0.35: s += 0.10
    return min(s, 1.0)

def score_L(h: Hand) -> float:
    """Index pointing UP + thumb pointing SIDEWAYS = 90° L"""
    s = 0.0
    if h.i_up:     s += 0.35   # Index must be up
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.15
    # Thumb must be clearly extended (the horizontal bar of L)
    if h.thumb_ext > 0.40: s += 0.20
    if h.thumb_ext > 0.50: s += 0.10  # Extra for very extended
    return min(s, 1.0)

def score_M(h: Hand) -> float:
    """Index + middle + ring drape OVER tucked thumb (3 fingers cover thumb)"""
    s = 0.0
    # All 4 fingers down
    if not h.i_up: s += 0.20
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.10
    # 3 fingertips BELOW thumb tip (draped over it)
    lm = h.lm
    if lm[I_TIP].y > lm[T_TIP].y: s += 0.12
    if lm[M_TIP].y > lm[T_TIP].y: s += 0.12
    if lm[R_TIP].y > lm[T_TIP].y: s += 0.12
    if h.ic > 0.50 and h.mc > 0.50 and h.rc > 0.45: s += 0.10
    return min(s, 1.0)

def score_N(h: Hand) -> float:
    """Index + middle drape over thumb (only 2 fingers, ring is NOT draped)"""
    s = 0.0
    if not h.i_up: s += 0.20
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.10
    if not h.p_up: s += 0.10
    lm = h.lm
    if lm[I_TIP].y > lm[T_TIP].y: s += 0.15
    if lm[M_TIP].y > lm[T_TIP].y: s += 0.15
    # Ring tip NOT below thumb (key diff from M)
    if lm[R_TIP].y < lm[T_TIP].y: s += 0.12
    if h.ic > 0.50 and h.mc > 0.50: s += 0.08
    return min(s, 1.0)

def score_O(h: Hand) -> float:
    """All fingertips meet thumb tip forming visible oval/circle"""
    s = 0.0
    # All tips close to thumb
    if h.ti < 0.30: s += 0.25
    if h.tm < 0.35: s += 0.20
    if h.tr < 0.45: s += 0.15
    # Visible gap (not flat fist) — index MCP to thumb tip
    if h.gap > 0.25: s += 0.20
    # Fingers not fully up (they curve toward thumb)
    if not h.i_up and not h.m_up: s += 0.15
    if h.ti < 0.22: s += 0.10  # Bonus for very tight circle
    return min(s, 1.0)

def score_R(h: Hand) -> float:
    """Index and middle CROSSED over each other"""
    s = 0.0
    if h.i_up: s += 0.30
    if h.m_up: s += 0.30
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.10
    # KEY: fingers very close or overlapping
    if h.im < 0.10: s += 0.20   # Nearly touching = crossed
    if h.thumb_ext < 0.35: s += 0.05
    return min(s, 1.0)

def score_S(h: Hand) -> float:
    """Fist, thumb rests OVER (on top of) index+middle knuckles"""
    s = 0.0
    if not h.i_up: s += 0.20
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.18
    if not h.p_up: s += 0.15
    # Thumb covers fingers = tip is over MCP area
    ty = h.lm[T_TIP].y; iy = h.lm[I_MCP].y
    if ty > iy - 0.05: s += 0.15  # Thumb at or below index MCP height
    if h.thumb_ext < 0.30: s += 0.12
    return min(s, 1.0)

def score_T(h: Hand) -> float:
    """Fist, thumb tip peeks between index and middle fingers"""
    s = 0.0
    if not h.i_up: s += 0.20
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.12
    # Thumb tip x is between index MCP and middle MCP
    tx = h.lm[T_TIP].x
    lo = min(h.lm[I_MCP].x, h.lm[M_MCP].x)
    hi = max(h.lm[I_MCP].x, h.lm[M_MCP].x)
    margin = 0.02
    if (lo - margin) < tx < (hi + margin): s += 0.25
    if h.thumb_ext < 0.35: s += 0.08
    return min(s, 1.0)

def score_U(h: Hand) -> float:
    """Index + middle up and PRESSED TOGETHER (close gap)"""
    s = 0.0
    if h.i_up:     s += 0.30
    if h.m_up:     s += 0.30
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.12
    # KEY: fingers close together
    if h.im < 0.20: s += 0.18
    if h.thumb_ext < 0.35: s += 0.05
    return min(s, 1.0)

def score_V(h: Hand) -> float:
    """Index + middle up and SPREAD APART (peace/victory sign)"""
    s = 0.0
    if h.i_up:     s += 0.28
    if h.m_up:     s += 0.28
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.12
    # KEY: clear gap between fingers
    if h.im > 0.18: s += 0.20
    if h.im > 0.25: s += 0.08  # Extra for very spread
    if h.thumb_ext < 0.35: s += 0.05
    return min(s, 1.0)

def score_W(h: Hand) -> float:
    """Index + middle + ring all up and spread like a fan"""
    s = 0.0
    if h.i_up:     s += 0.25
    if h.m_up:     s += 0.25
    if h.r_up:     s += 0.25
    if not h.p_up: s += 0.15
    if h.thumb_ext < 0.35: s += 0.10
    return min(s, 1.0)

def score_X(h: Hand) -> float:
    """ONLY index extended but bent/hooked like a fishhook"""
    s = 0.0
    # Index partially extended (not fully up, not fully down)
    if h.i_any and not h.i_up: s += 0.30   # Partially up = bent
    if 0.25 < h.ic < 0.75:    s += 0.25   # Curl in hook range
    if not h.m_up: s += 0.15
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.10
    if h.thumb_ext < 0.40: s += 0.05
    return min(s, 1.0)

def score_Y(h: Hand) -> float:
    """Thumb OUT + ONLY pinky up (Hawaiian shaka / hang-loose)"""
    s = 0.0
    if h.p_up:     s += 0.30   # Pinky up
    if not h.i_up: s += 0.18
    if not h.m_up: s += 0.15
    if not h.r_up: s += 0.12
    # CRITICAL: thumb must be CLEARLY EXTENDED
    if h.thumb_ext > 0.40: s += 0.25
    if h.thumb_ext > 0.50: s += 0.10  # Extra bonus
    return min(s, 1.0)

def score_1(h: Hand) -> float:
    """Only index up, thumb folded"""
    s = 0.0
    if h.i_up:     s += 0.40
    if not h.m_up: s += 0.20
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.12
    if h.thumb_ext < 0.38: s += 0.13
    return min(s, 1.0)

def score_2(h: Hand) -> float:
    """Index + middle up, thumb not extended"""
    s = 0.0
    if h.i_up:     s += 0.30
    if h.m_up:     s += 0.30
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.12
    if h.thumb_ext < 0.38: s += 0.13
    return min(s, 1.0)

def score_3(h: Hand) -> float:
    """Thumb + index + middle extended"""
    s = 0.0
    if h.i_up: s += 0.28
    if h.m_up: s += 0.28
    if not h.r_up: s += 0.15
    if not h.p_up: s += 0.12
    if h.thumb_ext > 0.38: s += 0.22
    return min(s, 1.0)

def score_4(h: Hand) -> float:
    """4 fingers up, thumb tucked in"""
    s = 0.0
    if h.i_up: s += 0.22
    if h.m_up: s += 0.22
    if h.r_up: s += 0.22
    if h.p_up: s += 0.20
    if h.thumb_ext < 0.32: s += 0.18
    if h.im > 0.10: s += 0.05   # Some spread (vs B which is tight)
    return min(s, 1.0)

def score_5(h: Hand) -> float:
    """All 5 fingers spread open"""
    s = 0.0
    if h.i_up: s += 0.20
    if h.m_up: s += 0.20
    if h.r_up: s += 0.20
    if h.p_up: s += 0.18
    if h.thumb_ext > 0.40: s += 0.22
    return min(s, 1.0)

def score_6(h: Hand) -> float:
    """Pinky+thumb touch, index+middle+ring up"""
    s = 0.0
    if h.i_up: s += 0.22
    if h.m_up: s += 0.22
    if h.r_up: s += 0.22
    if not h.p_up: s += 0.10
    if h.tp < 0.25: s += 0.28   # Pinky tip near thumb tip
    return min(s, 1.0)

def score_7(h: Hand) -> float:
    """Ring+thumb touch, index+middle+pinky up"""
    s = 0.0
    if h.i_up: s += 0.22
    if h.m_up: s += 0.22
    if not h.r_up: s += 0.10
    if h.p_up: s += 0.22
    if h.tr < 0.25: s += 0.28   # Ring tip near thumb tip
    return min(s, 1.0)

def score_8(h: Hand) -> float:
    """Middle+thumb touch, index+ring+pinky up"""
    s = 0.0
    if h.i_up: s += 0.20
    if not h.m_up: s += 0.15
    if h.r_up: s += 0.20
    if h.p_up: s += 0.18
    if h.tm < 0.25: s += 0.30   # Middle tip near thumb tip
    return min(s, 1.0)

def score_0(h: Hand) -> float:
    """ALL fingertips pinch tightly to thumb (tight O)"""
    s = 0.0
    if h.ti < 0.22: s += 0.28
    if h.tm < 0.28: s += 0.22
    if h.tr < 0.36: s += 0.15
    if not h.i_up and not h.m_up: s += 0.20
    if h.ti < 0.16: s += 0.15   # Very tight = bonus
    return min(s, 1.0)

def score_9(h: Hand) -> float:
    """Index+thumb circle (OK), middle+ring+pinky up"""
    s = 0.0
    if not h.i_up: s += 0.20
    if h.m_up:     s += 0.22
    if h.r_up:     s += 0.22
    if h.p_up:     s += 0.18
    if h.ti < 0.30: s += 0.25   # Index near thumb = circle
    return min(s, 1.0)


# ── Master scorer map ──────────────────────────────────────────────
SCORERS = {
    'A':score_A, 'B':score_B, 'C':score_C, 'D':score_D, 'E':score_E,
    'F':score_F, 'G':score_G, 'H':score_H, 'I':score_I,
    'K':score_K, 'L':score_L, 'M':score_M, 'N':score_N, 'O':score_O,
    'R':score_R, 'S':score_S, 'T':score_T, 'U':score_U, 'V':score_V,
    'W':score_W, 'X':score_X, 'Y':score_Y,
    '0':score_0, '1':score_1, '2':score_2, '3':score_3, '4':score_4,
    '5':score_5, '6':score_6, '7':score_7, '8':score_8, '9':score_9,
}

# Signs that require BOTH index+middle up but differ only in:
# - spread (V vs U) and horizontal angle (H)
# These get extra disambiguation after main scoring
DISAMBIGUATION_GROUPS = [
    ['V', 'U', 'R', 'H', '2'],      # Index+middle signs
    ['I', 'Y'],                       # Pinky-up signs
    ['L', 'D', '1', 'G'],            # Index-up signs
    ['A', 'S', 'T', 'M', 'N', 'E'], # Closed fist signs
    ['O', 'C', '0'],                  # Circular signs
    ['B', '4', '5', 'W'],            # Multi-finger signs
]

MIN_CONFIDENCE = 0.58   # Must beat this threshold to report


# ══════════════════════════════════════════════════════════════════
class GestureEngine:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
        )
        self._smooth_buf  = []
        self._smooth_n    = 4
        self._last_sign   = None
        self._last_time   = 0.0
        self._settings    = {"show_landmarks": True}
        logger.info("GestureEngine v6 (Score & Vote) initialized")

    def is_ready(self):       return self.hands is not None
    def update_settings(self, s): self._settings.update(s)

    def process_frame(self, frame: np.ndarray) -> dict:
        if frame is None: return self._empty()
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res  = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_hand_landmarks:
            self._smooth_buf.clear()
            return self._empty(frame.copy())

        hand_lm = res.multi_hand_landmarks[0]
        lm      = hand_lm.landmark
        hand    = Hand(lm)

        # Score all signs simultaneously
        scores = {sign: fn(hand) for sign, fn in SCORERS.items()}

        # Pick highest scoring sign
        best_sign = max(scores, key=scores.get)
        best_score = scores[best_sign]

        # Only report if above confidence threshold
        raw_sign   = best_sign if best_score >= MIN_CONFIDENCE else None
        confidence = best_score

        # Rate limiting (backend spam protection)
        now = time.time() * 1000
        if raw_sign:
            gap = 2200 if raw_sign == self._last_sign else 700
            if now - self._last_time < gap:
                raw_sign = None

        smoothed = self._smooth(raw_sign)
        if smoothed:
            self._last_sign = smoothed
            self._last_time = now

        lm_json = [{"x":p.x,"y":p.y,"z":p.z} for p in lm]
        ann     = self._draw(frame.copy(), hand_lm, lm, smoothed, confidence, w, h)

        return {
            "sign":           smoothed,
            "confidence":     round(confidence, 3),
            "hand_detected":  True,
            "landmarks_json": lm_json,
            "annotated_frame":ann,
            "debug_scores":   {k:round(v,2) for k,v in sorted(scores.items(),key=lambda x:-x[1])[:5]}
        }

    def _smooth(self, sign):
        self._smooth_buf.append(sign)
        if len(self._smooth_buf) > self._smooth_n:
            self._smooth_buf.pop(0)
        valid = [s for s in self._smooth_buf if s]
        if not valid: return None
        top, cnt = Counter(valid).most_common(1)[0]
        if cnt >= max(2, int(self._smooth_n * 0.60)):
            return top
        return None

    def _draw(self, frame, hand_lm, lm, sign, conf, w, h):
        if self._settings.get("show_landmarks", True):
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,245,196), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0,180,140), thickness=2),
            )
        if sign:
            pts = [(lm[i].x*w, lm[i].y*h) for i in range(21)]
            xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x1 = int(max(0, min(xs)-22)); y1 = int(max(0, min(ys)-22))
            x2 = int(min(w, max(xs)+22)); y2 = int(min(h, max(ys)+22))
            col = (0,245,196)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(cx,cy),(cx+dx*14,cy),col,3)
                cv2.line(frame,(cx,cy),(cx,cy+dy*14),col,3)
            cv2.rectangle(frame,(x1,max(0,y1-38)),(x1+52,y1),(10,10,30),-1)
            cv2.putText(frame,f" {sign}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,1.3,col,3,cv2.LINE_AA)
            cv2.putText(frame,f"{int(conf*100)}%",(x2-42,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(160,160,160),1,cv2.LINE_AA)
        return frame

    def _empty(self, frame=None):
        return {"sign":None,"confidence":0.0,"hand_detected":False,
                "landmarks_json":None,"annotated_frame":frame,"debug_scores":{}}

    def get_alphabet_reference(self): return ASL_REFERENCE["alphabet"]
    def get_numbers_reference(self):  return ASL_REFERENCE["numbers"]
    def get_sign_detail(self, sign):
        return {**ASL_REFERENCE["alphabet"],**ASL_REFERENCE["numbers"]}.get(sign)
    def cleanup(self):
        if self.hands: self.hands.close()