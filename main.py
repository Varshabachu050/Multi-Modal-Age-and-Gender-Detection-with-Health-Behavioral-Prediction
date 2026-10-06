import os

# Keep CPU thread pools small (must be set BEFORE importing numpy / cv2 / onnxruntime)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import shutil
import threading
import traceback
import urllib.request
import zipfile

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB max upload

MAX_IMAGE_SIDE = 640      # bigger frames are shrunk before analysis
DET_SIZE = (320, 320)     # face detector input size (lower = less RAM)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# MODEL FILES (downloaded once if they are not in ./models)
# =========================================================

BUFFALO_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
)
BUFFALO_NEEDED = ["det_500m.onnx", "genderage.onnx"]

EMOTION_FILE = "emotion-ferplus-8.onnx"
EMOTION_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)
# FER+ output order
EMOTION_LABELS = [
    "Neutral", "Happy", "Surprise", "Sad",
    "Angry", "Disgust", "Fear", "Contempt",
]


def ensure_buffalo():
    """Extract ONLY the 2 needed files from the buffalo_s pack (saves RAM)."""
    if all(os.path.exists(os.path.join(MODEL_DIR, n)) for n in BUFFALO_NEEDED):
        return
    zip_path = os.path.join(MODEL_DIR, "buffalo_s.zip")
    print("Downloading face model pack (one time)...", flush=True)
    urllib.request.urlretrieve(BUFFALO_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name in BUFFALO_NEEDED:
                with z.open(member) as src, open(
                    os.path.join(MODEL_DIR, name), "wb"
                ) as out:
                    shutil.copyfileobj(src, out)
    os.remove(zip_path)


def ensure_emotion():
    """Download the small emotion model. Returns True if it is available."""
    path = os.path.join(MODEL_DIR, EMOTION_FILE)
    # a real model is ~34 MB; a tiny file means a broken download / LFS pointer
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return True
    try:
        print("Downloading emotion model (one time)...", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(EMOTION_URL, tmp)
        if os.path.getsize(tmp) < 1_000_000:
            os.remove(tmp)
            raise RuntimeError("downloaded file too small")
        os.replace(tmp, path)
        return True
    except Exception as error:
        print("Emotion model unavailable:", error, flush=True)
        return False


print("Loading models...", flush=True)
ensure_buffalo()
EMOTION_OK = ensure_emotion()

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.enable_cpu_mem_arena = False
sess_options.enable_mem_pattern = False
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
PROVIDERS = ["CPUExecutionProvider"]

detector = model_zoo.get_model(
    os.path.join(MODEL_DIR, "det_500m.onnx"),
    providers=PROVIDERS, sess_options=sess_options,
)
detector.prepare(ctx_id=-1, input_size=DET_SIZE, det_thresh=0.5)

genderage = model_zoo.get_model(
    os.path.join(MODEL_DIR, "genderage.onnx"),
    providers=PROVIDERS, sess_options=sess_options,
)
genderage.prepare(ctx_id=-1)

emotion_session = None
emotion_input = None
if EMOTION_OK:
    try:
        emotion_session = ort.InferenceSession(
            os.path.join(MODEL_DIR, EMOTION_FILE),
            sess_options=sess_options, providers=PROVIDERS,
        )
        emotion_input = emotion_session.get_inputs()[0].name
    except Exception as error:
        print("Could not load emotion model:", error, flush=True)
        emotion_session = None

print("Models loaded. Emotion model:", emotion_session is not None, flush=True)

model_lock = threading.Lock()

# =========================================================
# ANALYSIS HELPERS
# =========================================================


def detect_emotion(face_bgr):
    """Emotion from a face crop using the FER+ ONNX model."""
    if emotion_session is None or face_bgr is None or face_bgr.size == 0:
        return "Neutral", 0.0
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32).reshape(1, 1, 64, 64)  # raw 0-255 values
        scores = emotion_session.run(None, {emotion_input: x})[0][0]
        e = np.exp(scores - np.max(scores))
        probs = e / e.sum()
        i = int(np.argmax(probs))
        return EMOTION_LABELS[i], round(float(probs[i]) * 100, 1)
    except Exception as error:
        print("Emotion error:", error, flush=True)
        return "Neutral", 0.0


def detect_skin_type(face_bgr):
    """Rough skin-type guess from HSV brightness/saturation of the face centre.
    Very lighting-dependent, so treat it as an estimate only."""
    if face_bgr is None or face_bgr.size == 0:
        return "Normal"
    h, w = face_bgr.shape[:2]
    centre = face_bgr[int(h * 0.25):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    if centre.size == 0:
        centre = face_bgr
    hsv = cv2.cvtColor(cv2.resize(centre, (100, 100)), cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    if mean_v > 170 and mean_s > 80:
        return "Oily"
    if mean_v < 110 and mean_s < 60:
        return "Dry"
    if 110 <= mean_v <= 150 and mean_s > 70:
        return "Combination"
    return "Normal"


def analyze_behavior(emotion, age):
    e = emotion.lower()
    if e == "angry":
        return "Stressed"
    if e == "sad":
        return "Low Mood"
    if e in ("fear", "disgust"):
        return "Anxious"
    if e == "surprise":
        return "Alert"
    if e == "happy":
        return "Confident"
    if e == "neutral" and age and age > 50:
        return "Tired"
    return "Calm"


def get_tips(age, gender, emotion, skin, behavior):
    e, b, s, g = emotion.lower(), behavior.lower(), skin.lower(), (gender or "").lower()
    age = age if isinstance(age, int) else 22

    if e == "happy" or b == "confident":
        t1 = "Keep that positive energy: stay consistent with your sleep schedule."
    elif e == "sad" or b == "low mood":
        t1 = "Take a short walk outside, talk to someone you trust, aim for 7-8 hrs sleep."
    elif e == "angry" or b == "stressed":
        t1 = "Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s) to reduce stress."
    elif e in ("fear", "disgust") or b == "anxious":
        t1 = "5-min mindfulness: close your eyes, focus on breathing, slowly release tension."
    elif b == "tired":
        t1 = "20-20-20 rule: every 20 min, look 20 ft away for 20 sec to rest your eyes."
    else:
        t1 = "30 min of movement daily boosts mood, focus, and energy."

    if age < 18:
        t2 = "8-9 hrs of sleep matters a lot at your age. Avoid late nights."
    elif age <= 25:
        t2 = "Build habits now: regular meals, limit junk food, drink about 2 litres of water daily."
    elif age <= 35:
        t2 = "Add strength or cardio 3 times a week to stay active."
    elif age <= 50:
        t2 = "Schedule annual checkups: blood pressure, cholesterol, and Vitamin D levels."
    else:
        t2 = "A low-sodium diet, a daily 30-min walk, and regular doctor visits are important now."

    if s == "oily":
        t3 = "Oily skin: cleanse twice daily with a gentle foaming wash and stay hydrated."
    elif s == "dry":
        t3 = "Dry skin: apply a fragrance-free moisturiser within 3 min of washing your face."
    elif s == "combination":
        t3 = "Combination skin: lightweight gel on the T-zone, cream moisturiser on dry cheeks."
    else:
        t3 = "Normal skin: apply SPF 30+ sunscreen daily, even when indoors."

    if g == "female":
        t4 = (
            "Include iron- and calcium-rich foods such as leafy greens and dairy."
            if age < 30 else
            "Look after bone health: calcium, Vitamin D, and regular exercise such as yoga."
        )
    elif g == "male":
        t4 = (
            "Eat enough protein and stay active to support healthy muscle growth."
            if age < 30 else
            "Heart health matters: limit saturated fats and monitor blood pressure."
        )
    else:
        t4 = "Drink water first thing each morning. It aids digestion."

    return [t1, t2, t3, t4]


# =========================================================
# CORE PIPELINE
# =========================================================


def run_analysis(frame):
    """Returns (result_dict, None) or (None, 'no_face')."""
    h, w = frame.shape[:2]

    with model_lock:
        bboxes, kpss = detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or bboxes.shape[0] == 0:
            return None, "no_face"

        # use the largest face
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas))
        face = Face(
            bbox=bboxes[i, 0:4],
            kps=kpss[i] if kpss is not None else None,
            det_score=bboxes[i, 4],
        )
        genderage.get(frame, face)  # fills face.gender and face.age

    x1, y1, x2, y2 = face.bbox.astype(int)
    pad_x, pad_y = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2]

    # light-normalise the crop (CLAHE) so skin/emotion are less lighting-sensitive
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    crop_norm = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    age = int(round(float(face.age)))
    gender = "Male" if int(face.gender) == 1 else "Female"  # 0=Female, 1=Male

    with model_lock:
        emotion, emotion_conf = detect_emotion(crop_norm)

    skin = detect_skin_type(crop_norm)
    behavior = analyze_behavior(emotion, age)

    return {
        "age": age,
        "gender": gender,
        "emotion": emotion,
        "emotion_conf": emotion_conf,
        "skin": skin,
        "behavior": behavior,
        "tips": get_tips(age, gender, emotion, skin, behavior),
        # face box as fractions of the image, so the page can draw it
        "bbox": [
            max(0.0, x1 / w), max(0.0, y1 / h),
            min(1.0, x2 / w), min(1.0, y2 / h),
        ],
    }, None


# =========================================================
# ROUTES
# =========================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image too large (max 4 MB)."}), 413


@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    """Accepts a JPEG as multipart field 'frame' or as the raw request body."""
    frame = None
    try:
        if request.content_type and request.content_type.startswith("multipart"):
            if "frame" not in request.files:
                return jsonify({"error": "No 'frame' field in form data."}), 400
            raw = request.files["frame"].read()
        else:
            raw = request.get_data()

        if not raw:
            return jsonify({"error": "Empty request."}), 400

        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        del raw
        if frame is None:
            return jsonify({"error": "Cannot decode image."}), 400

        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > MAX_IMAGE_SIDE:
            s = MAX_IMAGE_SIDE / longest
            frame = cv2.resize(
                frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
            )

        result, err = run_analysis(frame)
        if err == "no_face":
            return jsonify({"error": "no_face"}), 200  # soft error, page handles it
        return jsonify(result), 200

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": "Analysis failed: " + str(ex)}), 500

    finally:
        del frame
        gc.collect()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "emotion_model": emotion_session is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
