"""
=========================================================
   03 - Real-Time Drowsiness Detection + Voice Alert
   (CNN Model + EAR Eye-Closure Detection combined)
=========================================================
Had script kayjam3 bin JOUJ signals bach ykun ahsen w asra3:

    1) CNN Model (best_model.keras) -> kayshouf l wjeh kaml w kayqol
       "Drowsy" / "Non Drowsy" (général, b base 3la dataset)

    2) EAR (Eye Aspect Ratio) -> kayhseb b formule riyadia direct mn
       landmarks dyal l3inin (368/362...) wach l3inin msdodin wla la.
       Hadi asra3 w adaq bzaf l "sad 3inih" specifiquement.

    -> Ila wa7ed mnhom 9al "khatar" (CNN=Drowsy WLA EAR=3inin msdodin),
       kaytfa3el SAWT TTS bel darija + sfara + (optional) confirmation
       b sawt (Whisper, chof 04_voice_confirmation.py).

Installation:
    pip install opencv-python mediapipe tensorflow gTTS playsound==1.2.2 numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
from playsound import playsound
import threading
import time
import json
import os

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Voice confirmation (Hugging Face Whisper) - optional, kaykhdam ghir
# ila l fichier 04_voice_confirmation.py kayn f nfs l folder
try:
    from importlib import import_module
    voice_confirm_module = import_module("04_voice_confirmation")
    HAS_VOICE_CONFIRM = True
except Exception:
    HAS_VOICE_CONFIRM = False
    print("⚠️ Voice confirmation machi disponible (chof 04_voice_confirmation.py)")

# ---------- CONFIG ----------
MODEL_PATH = "benchmark_results/best_model.keras"
CLASS_INDICES_PATH = "benchmark_results/class_indices.json"
IMG_SIZE = (64, 64)

# EAR thresholds (eye closure - direct detection)
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 12        # ~ 1 - 1.5 secondes b3d li tsdo l3inin

# CNN thresholds (general drowsiness)
DROWSY_CONSEC_FRAMES = 9

ALERT_COOLDOWN_SEC = 5         # bach ma ykrarch sawt kola frame

# MediaPipe Face Mesh eye landmarks indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ---------- LOAD MODEL ----------
print("Tahmil l model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
print("Classes:", idx_to_class)

# ---------- TTS SETUP (gTTS - Darija, b cache) ----------
ALERT_SOUNDS_DIR = "alert_sounds"
os.makedirs(ALERT_SOUNDS_DIR, exist_ok=True)

ALERT_MESSAGES = {
    "eyes": ("فيق! عينيك مسدودين، وقف الطوموبيل", "alert_eyes.mp3"),
    "drowsy": ("نتا تعبان وساق، خاصك توقف وترتاح", "alert_drowsy.mp3"),
}

ALERT_PATHS = {}
for key, (text, filename) in ALERT_MESSAGES.items():
    path = os.path.join(ALERT_SOUNDS_DIR, filename)
    ALERT_PATHS[key] = path
    if not os.path.exists(path):
        print(f"Tahmil sawt '{key}' mn internet (mra wahda ghir)...")
        tts = gTTS(text=text, lang="ar", slow=False)
        tts.save(path)
        print(f"[OK] Sawt msajel f: {path}")

is_speaking = False


def play_siren():
    """Sfara dyal ndar - kat3awd bin jouj frequencies bach tkhal3, instant (bla internet)"""
    if HAS_WINSOUND:
        for _ in range(3):
            winsound.Beep(1500, 150)  # frequency Hz, duration ms
            winsound.Beep(1000, 150)
    else:
        print("\a\a\a")  # fallback bsit l Linux/Mac


def speak_alert(alert_key):
    """Sfara dyal ndar (instant) + sawt b kelmat + (optional) confirmation"""
    global is_speaking
    if is_speaking:
        return
    is_speaking = True
    try:
        play_siren()                       # 1) sfara fawran - bach yfi9 bzrba
        playsound(ALERT_PATHS[alert_key])  # 2) mn b3d l message b kelmat

        # 3) (optional) nstanaw confirmation b sawt - Whisper
        if HAS_VOICE_CONFIRM:
            confirmed, text = voice_confirm_module.listen_and_confirm()
            if not confirmed:
                print("🚨 ESCALATION: ma jawabch! Khatar akbar.")
                play_siren()
                play_siren()  # sfara a9wa - tkrarat
                # Hna momkin tzid: simulation dyal SMS/Email l jiha dyal twari2
                # (mathlan b Twilio API, wla bAch tbAt notification b n8n)

    except Exception as e:
        print(f"Khta2 f sawt: {e}")
    finally:
        is_speaking = False


# ---------- MEDIAPIPE FACE MESH (wjeh + 3inin f nfs lwa9t) ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = coords
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_face_bbox(landmarks, w, h, margin=0.15):
    """Kayhseb bounding box dyal wjeh kaml mn ga3 les landmarks (l CNN)"""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * margin))
    y1 = max(0, int(y1 - box_h * margin))
    x2 = min(w, int(x2 + box_w * margin))
    y2 = min(h, int(y2 + box_h * margin))
    return x1, y1, x2, y2


def preprocess_face(face_img):
    face_img = cv2.resize(face_img, IMG_SIZE)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khta2: ma9drtch n7el l camera.")
        return

    eye_counter = 0
    drowsy_counter = 0
    last_alert_time = 0
    print("Camera 7ayda... b Q bach tsd.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "Ma l9itch wjeh"
        status_color = (0, 165, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ---------- 1) EAR - eye closure direct ----------
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            eyes_closed = ear < EAR_THRESHOLD
            if eyes_closed:
                eye_counter += 1
            else:
                eye_counter = 0

            # ---------- 2) CNN - drowsiness général ----------
            x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
            face_crop = frame[y1:y2, x1:x2]

            prediction_label = "N/A"
            confidence = 0.0
            if face_crop.size > 0:
                input_tensor = preprocess_face(face_crop)
                pred = model.predict(input_tensor, verbose=0)[0][0]
                pred_class_idx = int(pred > 0.5)
                prediction_label = idx_to_class[pred_class_idx]
                confidence = pred if pred_class_idx == 1 else 1 - pred

            is_drowsy_cnn = prediction_label.lower() == "drowsy"
            if is_drowsy_cnn:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            # ---------- 3) DECISION (eyes OR drowsy) ----------
            current_time = time.time()
            can_alert = (current_time - last_alert_time) > ALERT_COOLDOWN_SEC

            if eye_counter >= EAR_CONSEC_FRAMES:
                status_text = f"FI9! 3INIK MSDODIN (EAR={ear:.2f})"
                status_color = (0, 0, 255)
                if can_alert:
                    last_alert_time = current_time
                    threading.Thread(target=speak_alert, args=("eyes",), daemon=True).start()

            elif drowsy_counter >= DROWSY_CONSEC_FRAMES:
                status_text = f"NTABEH! T3BAN - Drowsy ({confidence*100:.0f}%)"
                status_color = (0, 0, 255)
                if can_alert:
                    last_alert_time = current_time
                    threading.Thread(target=speak_alert, args=("drowsy",), daemon=True).start()

            else:
                status_text = f"Mzyan, Mrkez | EAR={ear:.2f} | {prediction_label} ({confidence*100:.0f}%)"
                status_color = (0, 255, 0)

            # Draw face box + eye landmarks
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        # ---- Status banner ----
        cv2.rectangle(frame, (0, 0), (w, 40), status_color, -1)
        cv2.putText(frame, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Drowsiness Detection - Real Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()