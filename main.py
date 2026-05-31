import cv2
import numpy as np
import time
import sys
import threading
import os
import warnings
import logging

# ── Suppress ALL model loading noise ──────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Redirect stderr temporarily to suppress InsightFace print() spam
import io
_real_stderr = sys.stderr
_real_stdout = sys.stdout

def _silence():
    sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()

def _unsilence():
    sys.stderr = _real_stderr
    sys.stdout = _real_stdout

_silence()
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_OK = True
except Exception:
    INSIGHTFACE_OK = False
_unsilence()

_silence()
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception:
    DEEPFACE_OK = False
_unsilence()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_insightface_model():
    _silence()
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "genderage"],
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    _unsilence()
    return app

face_app = None


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_skin_type(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return "Normal"
    hsv    = cv2.cvtColor(cv2.resize(face_bgr, (100, 100)), cv2.COLOR_BGR2HSV)
    mean_v = np.mean(hsv[:, :, 2])
    mean_s = np.mean(hsv[:, :, 1])
    if   mean_v > 170 and mean_s > 80:          return "Oily"
    elif mean_v < 110 and mean_s < 60:          return "Dry"
    elif 110 <= mean_v <= 150 and mean_s > 70:  return "Combination"
    else:                                        return "Normal"


def detect_emotion(face_bgr):
    if not DEEPFACE_OK or face_bgr is None or face_bgr.size == 0:
        return "Neutral", 0.0
    try:
        _silence()
        r = DeepFace.analyze(face_bgr, actions=["emotion"],
                             enforce_detection=False, silent=True)
        _unsilence()
        r   = r[0] if isinstance(r, list) else r
        dom = r["dominant_emotion"].capitalize()
        conf= round(r["emotion"][r["dominant_emotion"]], 1)
        return dom, conf
    except Exception:
        _unsilence()
        return "Neutral", 0.0


def analyze_behavior(emotion, age):
    e = emotion.lower()
    if e == "angry":                         return "Stressed"
    if e == "sad":                           return "Low Mood"
    if e in ["fear", "disgust"]:             return "Anxious"
    if e == "surprise":                      return "Alert"
    if e == "happy":                         return "Confident"
    if e == "neutral" and age and age > 50:  return "Tired"
    return "Calm"


def get_tips(age, gender, emotion, skin, behavior):
    """Returns 4 personalised tips — one per category."""
    e = emotion.lower(); b = behavior.lower()
    s = skin.lower();    g = (gender or "").lower()
    age = age if isinstance(age, int) else 22

    # Tip 1 – Emotion / Behavior
    if e == "happy" or b == "confident":
        t1 = "Keep that positive energy — stay consistent with your sleep schedule."
    elif e == "sad" or b == "low mood":
        t1 = "Take a short walk outside, talk to someone you trust, aim for 7-8 hrs sleep."
    elif e == "angry" or b == "stressed":
        t1 = "Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s) to reduce stress instantly."
    elif e in ["fear","disgust"] or b == "anxious":
        t1 = "5-min mindfulness: close eyes, focus on breathing, slowly release tension."
    elif b == "tired":
        t1 = "20-20-20 rule: every 20 min, look 20 ft away for 20 sec to rest your eyes."
    else:
        t1 = "30 min of movement daily significantly boosts mood, focus, and energy levels."

    # Tip 2 – Age group
    if age < 18:
        t2 = "8-9 hrs sleep is critical for brain development at your age — avoid late nights."
    elif age <= 25:
        t2 = "Build habits now: regular meals, limit junk food, and drink 2 litres of water daily."
    elif age <= 35:
        t2 = "Add strength or cardio 3x/week — metabolism starts slowing slightly in this decade."
    elif age <= 50:
        t2 = "Schedule annual checkups: blood pressure, cholesterol, and Vitamin D levels."
    else:
        t2 = "Low-sodium diet, daily 30-min walk, and regular doctor visits are essential now."

    # Tip 3 – Skin type
    if s == "oily":
        t3 = "Oily skin: cleanse twice daily with a gentle foaming wash; drink 2-3L water."
    elif s == "dry":
        t3 = "Dry skin: apply fragrance-free moisturizer within 3 min of washing your face."
    elif s == "combination":
        t3 = "Combination skin: lightweight gel on T-zone, cream moisturizer on dry cheeks."
    else:
        t3 = "Normal skin: apply SPF 30+ sunscreen daily — even when sitting indoors."

    # Tip 4 – Gender + general wellness
    if g == "female":
        t4 = ("Iron and calcium are key — include leafy greens and dairy in daily meals."
              if age < 30 else
              "Prioritise bone health: 1000mg calcium + Vitamin D daily; yoga helps posture.")
    elif g == "male":
        t4 = ("Aim for 0.8g protein per kg of body weight to support healthy muscle growth."
              if age < 30 else
              "Heart health matters after 30: limit saturated fats and monitor blood pressure.")
    else:
        t4 = "Drink water first thing each morning — it kickstarts metabolism and aids digestion."

    return [t1, t2, t3, t4]


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL PRINT  — clears screen, shows ONE clean block
# ══════════════════════════════════════════════════════════════════════════════

CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"
RED="\033[91m";  BOLD="\033[1m";   RESET="\033[0m"; MAGENTA="\033[95m"

def print_result(age, gender, gender_conf, emotion, emotion_conf,
                 skin, behavior, tips):
    ec = {"happy":GREEN,"sad":CYAN,"angry":RED,"neutral":YELLOW,
          "surprise":MAGENTA,"fear":RED,"disgust":RED}.get(emotion.lower(), YELLOW)

    print("\033[2J\033[H", end="")   # clear terminal
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"  {BOLD}FACE ANALYSIS  —  FINAL RESULT{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")
    print(f"  {'Age':<14}: {BOLD}{age}{RESET}")
    print(f"  {'Gender':<14}: {BOLD}{gender}{RESET}  ({gender_conf:.0f}% confident)")
    print(f"  {'Emotion':<14}: {ec}{BOLD}{emotion}{RESET}  ({emotion_conf:.0f}%)")
    print(f"  {'Skin Type':<14}: {BOLD}{skin}{RESET}")
    print(f"  {'Behavior':<14}: {BOLD}{behavior}{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")
    print(f"  {BOLD}{GREEN}💡 Personalised Tips:{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")
    for i, tip in enumerate(tips, 1):
        print(f"  {BOLD}{YELLOW}{i}.{RESET} {tip}")
    print(f"{CYAN}{'═'*60}{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM OVERLAY  — minimal: just face box + label
# ══════════════════════════════════════════════════════════════════════════════

ECOL = {"happy":(0,220,100),"sad":(200,100,0),"angry":(0,50,220),
        "neutral":(180,180,50),"surprise":(220,150,0),
        "fear":(0,80,200),"disgust":(50,0,180)}

def draw_overlay(frame, x, y, w, h, age, gender, emotion):
    col = ECOL.get(emotion.lower(), (100,220,100))
    cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
    lbl = f"{gender}  |  {age}y  |  {emotion}"
    (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y-lh-8), (x+lw+6, y), col, -1)
    cv2.putText(frame, lbl, (x+3, y-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class Smoother:
    def __init__(self, n=6): self.n=n; self.h=[]
    def update(self, v): self.h.append(v); self.h=self.h[-self.n:]
    def get(self): return max(set(self.h),key=self.h.count) if self.h else None


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

result = {"age":"?","gender":"?","gender_conf":0.0,"emotion":"Neutral",
          "emotion_conf":0.0,"skin":"Normal","behavior":"Calm","tips":[]}
r_lock = threading.Lock()

latest_frame = None; latest_crop = None
f_lock = threading.Lock()

age_s=Smoother(6); gender_s=Smoother(8); emotion_s=Smoother(5)

_stable_key = ""; _stable_count = 0; STABLE_NEEDED = 5


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND THREAD
# ══════════════════════════════════════════════════════════════════════════════

def worker():
    global _stable_key, _stable_count

    while True:
        time.sleep(0.35)

        with f_lock:
            full = latest_frame.copy() if latest_frame is not None else None
            crop = latest_crop.copy()  if latest_crop  is not None else None

        if full is None:
            continue

        try:
            age_v=None; gender_v="?"; gconf=0.0

            if INSIGHTFACE_OK and face_app:
                _silence()
                faces = face_app.get(full)
                _unsilence()
                if faces:
                    f       = max(faces, key=lambda x: x.det_score)
                    age_v   = int(f.age)
                    gender_v= "Male" if int(f.gender)==1 else "Female"
                    gconf   = float(f.det_score)*100
                    age_s.update(age_v);     age_v   = age_s.get()
                    gender_s.update(gender_v); gender_v= gender_s.get()

            emotion_v, econf = detect_emotion(crop)
            emotion_s.update(emotion_v); emotion_v = emotion_s.get() or emotion_v

            skin_v    = detect_skin_type(crop)
            age_int   = age_v if isinstance(age_v, int) else 22
            behav_v   = analyze_behavior(emotion_v, age_int)
            tips_v    = get_tips(age_int, gender_v, emotion_v, skin_v, behav_v)

            with r_lock:
                result.update({"age":age_v,"gender":gender_v,"gender_conf":gconf,
                               "emotion":emotion_v,"emotion_conf":econf,
                               "skin":skin_v,"behavior":behav_v,"tips":tips_v})

            # Print only when stable for STABLE_NEEDED consecutive readings
            key = f"{age_v}|{gender_v}|{emotion_v}|{skin_v}|{behav_v}"
            if key == _stable_key:
                _stable_count += 1
                if _stable_count == STABLE_NEEDED:
                    print_result(age_v, gender_v, gconf, emotion_v, econf,
                                 skin_v, behav_v, tips_v)
            else:
                _stable_key   = key
                _stable_count = 0

        except Exception:
            _unsilence()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global face_app, latest_frame, latest_crop

    if INSIGHTFACE_OK:
        face_app = load_insightface_model()

    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    threading.Thread(target=worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = haar.detectMultiScale(gray, 1.1, 5, minSize=(80,80))

        best=None; best_a=0
        for (fx,fy,fw,fh) in rects:
            if fw*fh > best_a: best_a=fw*fh; best=(fx,fy,fw,fh)

        with r_lock:
            res = result.copy()

        if best:
            fx,fy,fw,fh = best
            crop = frame[fy:fy+fh, fx:fx+fw]
            lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l,a,b= cv2.split(lab)
            l    = cv2.createCLAHE(2.0,(8,8)).apply(l)
            norm = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

            with f_lock:
                latest_frame = frame.copy()
                latest_crop  = norm

            frame = draw_overlay(frame, fx,fy,fw,fh,
                                 res["age"], res["gender"], res["emotion"])
        else:
            cv2.putText(frame,"No face — move closer",(20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,80,255),2,cv2.LINE_AA)

        cv2.imshow("Face Analysis  |  Q = Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()