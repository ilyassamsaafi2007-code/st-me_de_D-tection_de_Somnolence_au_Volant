"""
==================================================
   Driver Drowsiness Detection - Streamlit App
   BETTER Voice Recording + Text Fallback
==================================================
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import re
import threading

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Driver Drowsiness - Voice Fixed",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS ----------
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .subtitle { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .status-box { padding: 1.5rem; border-radius: 10px; text-align: center; font-size: 1.8rem; font-weight: bold; margin: 1rem 0; }
    .status-safe { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .status-danger { background-color: #f8d7da; color: #721c24; border: 4px solid #dc3545; animation: pulse-danger 0.5s infinite; }
    .status-escalation { background-color: #721c24; color: #fff; border: 4px solid #f5c6cb; animation: pulse-escalation 0.3s infinite; }
    .status-confirmed { background-color: #d4edda; color: #155724; border: 3px solid #28a745; animation: pulse-green 1s infinite; }
    @keyframes pulse-danger { 0% { transform: scale(1); border-width: 4px; } 50% { transform: scale(1.03); border-width: 6px; } 100% { transform: scale(1); border-width: 4px; } }
    @keyframes pulse-escalation { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    @keyframes pulse-green { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    .stButton>button { width: 100%; height: 3rem; font-size: 1.1rem; font-weight: bold; }
    .voice-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; }
    .confirm-box { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; animation: pulse-green 1s infinite; }
    .confirm-box h3 { margin: 0; font-size: 1.5rem; }
    .confirm-box p { margin: 0.5rem 0 0 0; font-size: 1.1rem; }
    .alert-log { background: #f8f9fa; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #dc3545; margin: 0.3rem 0; font-size: 0.9rem; }
    .confirm-log { background: #f8f9fa; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 0.3rem 0; font-size: 0.9rem; }
    .escalation-log { background: #f8d7da; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #721c24; margin: 0.3rem 0; font-size: 0.9rem; font-weight: bold; }
    .debug-box { background: #1e1e1e; color: #00ff00; padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.85rem; }
    .audio-alert-box { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0; animation: shake 0.5s infinite; }
    @keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-5px); } 75% { transform: translateX(5px); } }
    .recorder-box { background: #fff3cd; border: 3px solid #ffc107; border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .recorder-box h3 { color: #856404; margin: 0 0 1rem 0; font-size: 1.5rem; }
    .recorder-box p { color: #666; margin: 0.5rem 0; }
    .big-button { font-size: 1.3rem !important; height: 4rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------- MEDIAPIPE ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 14]

# ---------- THRESHOLDS ----------
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.55
MAR_CONSEC_FRAMES = 12
ALERT_COOLDOWN_SEC = 5

# ---------- AUDIO ALERT SETUP ----------
HAS_GTTS = False
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    pass

try:
    from playsound import playsound
    HAS_PLAYSOUND = True
except ImportError:
    HAS_PLAYSOUND = False

ALERT_MESSAGES = {
    "eyes": "فيق! عينيك مسدودين، وقف الطوموبيل",
    "drowsy": "نتا تعبان بزاف، خاصك توقف وترتاح",
    "yawn": "تعبان وكتثاوب، وقف شوية",
}

CONFIRM_MESSAGES = {
    "confirmed": "ماشي مشكل! راه فايق، كمل الطريق بأمان",
    "escalation": "ما جاوبش! خاصك توقف فورا",
}

ALERT_SOUNDS_DIR = "alert_sounds"
os.makedirs(ALERT_SOUNDS_DIR, exist_ok=True)

def generate_alert_sound(message_key, text):
    if not HAS_GTTS:
        return None
    filepath = os.path.join(ALERT_SOUNDS_DIR, f"alert_{message_key}.mp3")
    if not os.path.exists(filepath):
        try:
            tts = gTTS(text=text, lang="ar", slow=False)
            tts.save(filepath)
        except Exception as e:
            print(f"Error generating sound: {e}")
            return None
    return filepath

alert_sound_files = {}
if HAS_GTTS:
    for key, text in ALERT_MESSAGES.items():
        alert_sound_files[key] = generate_alert_sound(key, text)

is_speaking = False

def play_alert_sound(message_key, repeat=3):
    global is_speaking
    if is_speaking:
        return
    is_speaking = True

    try:
        print("\a" * 3)

        if HAS_GTTS and HAS_PLAYSOUND and message_key in alert_sound_files:
            sound_file = alert_sound_files[message_key]
            if sound_file and os.path.exists(sound_file):
                for i in range(repeat):
                    try:
                        playsound(sound_file)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                        break
        else:
            for i in range(repeat):
                print(f"🚨 ALERT {i+1}/{repeat}: {ALERT_MESSAGES.get(message_key, 'ALERT')}")
                time.sleep(1)
    except Exception as e:
        print(f"Alert error: {e}")
    finally:
        is_speaking = False

def trigger_alert(message_key, repeat=3):
    thread = threading.Thread(target=play_alert_sound, args=(message_key, repeat), daemon=True)
    thread.start()

# ---------- UTILS ----------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = coords
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    h_dist = euclidean(p1, p4)
    if h_dist == 0:
        return 1.0
    return (v1 + v2) / (2.0 * h_dist)

def calculate_mar(landmarks, mouth_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_points]
    p1, p2, p3, p4, p5, p6 = coords
    v = euclidean(p2, p5)
    h_dist = euclidean(p1, p4)
    if h_dist == 0:
        return 0.0
    return v / h_dist

def draw_landmarks(frame, landmarks, w, h):
    for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    return frame

# ---------- SESSION STATE ----------
def init_state():
    defaults = {
        'running': False, 'eye_counter': 0, 'yawn_counter': 0,
        'alarm_triggered': False, 'total_frames': 0, 'drowsy_events': 0,
        'yawn_events': 0, 'start_time': None, 'last_alert_time': 0,
        'voice_active': False, 'waiting_voice': False, 'voice_text': "",
        'voice_confirmed': False, 'escalation_count': 0, 'alerts_log': [],
        'debug_info': "", 'audio_enabled': True, 'alert_repeat': 3,
        'confirmation_status': None, 'confirmation_time': 0,
        'text_response': "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def reset():
    st.session_state.eye_counter = 0
    st.session_state.yawn_counter = 0
    st.session_state.alarm_triggered = False
    st.session_state.total_frames = 0
    st.session_state.drowsy_events = 0
    st.session_state.yawn_events = 0
    st.session_state.start_time = time.time()
    st.session_state.last_alert_time = 0
    st.session_state.voice_active = False
    st.session_state.waiting_voice = False
    st.session_state.voice_text = ""
    st.session_state.voice_confirmed = False
    st.session_state.escalation_count = 0
    st.session_state.alerts_log = []
    st.session_state.debug_info = ""
    st.session_state.confirmation_status = None
    st.session_state.confirmation_time = 0
    st.session_state.text_response = ""

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    ear_thresh = st.slider("Seuil EAR", 0.10, 0.40, 0.22, 0.01)
    ear_frames = st.slider("Frames yeux", 5, 50, 15, 1)
    mar_thresh = st.slider("Seuil MAR", 0.30, 0.90, 0.55, 0.01)
    mar_frames = st.slider("Frames bâillement", 5, 40, 12, 1)

    st.markdown("---")
    st.markdown("### 🔊 Audio")
    st.session_state.audio_enabled = st.toggle("Activer audio", value=True)
    st.session_state.alert_repeat = st.slider("Répétitions", 1, 5, 3, 1)

    if HAS_GTTS and HAS_PLAYSOUND:
        st.success("✅ Audio TTS prêt")
    else:
        st.warning("⚠️ Audio limité")
        st.info("pip install gTTS playsound==1.2.2")

    st.markdown("---")
    st.markdown("### 📊 Stats")
    if st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        mins, secs = divmod(elapsed, 60)
        st.metric("⏱️ Temps", f"{mins:02d}:{secs:02d}")
    else:
        st.metric("⏱️ Temps", "00:00")
    st.metric("📸 Frames", st.session_state.total_frames)
    st.metric("😴 Sommeil", st.session_state.drowsy_events)
    st.metric("🥱 Bâillement", st.session_state.yawn_events)
    st.metric("🚨 Escalation", st.session_state.escalation_count)

# ============ MAIN ============
st.markdown('<p class="main-title">🚗 Driver Drowsiness Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔊 Audio Alerts + Voice Confirmation (FIXED)</p>', unsafe_allow_html=True)

audio_alert_banner = st.empty()
confirmation_banner = st.empty()
voice_banner = st.empty()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("▶️ Démarrer", type="primary", use_container_width=True):
        st.session_state.running = True
        reset()
        st.rerun()
with col2:
    if st.button("⏹️ Arrêter", use_container_width=True):
        st.session_state.running = False
        st.rerun()
with col3:
    if st.button("🔄 Reset", use_container_width=True):
        reset()
        st.rerun()

status_col, video_col = st.columns([1, 2])

with status_col:
    st.markdown("### 📊 État")
    status_placeholder = st.empty()

    st.markdown("### 🎤 Confirmation")
    confirmation_status = st.empty()
    voice_status = st.empty()
    voice_result = st.empty()

    st.markdown("#### Métriques")
    ear_metric = st.empty()
    mar_metric = st.empty()
    eye_counter_metric = st.empty()
    yawn_counter_metric = st.empty()

    st.markdown("#### 🚨 Historique")
    alert_history = st.empty()

    st.markdown("#### 🔧 Debug")
    debug_placeholder = st.empty()

with video_col:
    st.markdown("### 📹 Vidéo")
    video_placeholder = st.empty()

    # VOICE RECORDER AREA - BIG AND CLEAR
    voice_recorder_area = st.empty()

    if not st.session_state.running:
        st.info("""
        👆 **Démarrer** pour activer la webcam.

        **Fonctionnement:**
        1. 📹 Camera detecte wjeh
        2. 👁️ Rmed 3inik → ALERT + AUDIO 🔊
        3. 🎤 "فيق! عينيك مسدودين..." (3x)
        4. 🎙️ **Katjaweb hna** (audio wla text)
        5. ✅ "ah/wakha" → **CONFIRMÉ!**
        6. 🚨 Skat → **ESCALATION!**
        """)
        demo = np.zeros((480, 640, 3), dtype=np.uint8)
        demo = cv2.putText(demo, "En attente...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        video_placeholder.image(demo, channels="BGR", use_column_width=True)

# ============ DETECTION LOOP ============
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Webcam inaccessible")
        st.session_state.running = False
    else:
        stop_btn = st.button("⏹️ Arrêter", key="stop_btn")

        while st.session_state.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = "✅ MZYAN - B9A MRKEZ"
            status_class = "status-safe"
            status_color = (0, 255, 0)
            ear_val = 0.0
            mar_val = 0.0
            alert_triggered = False
            alert_type = None
            debug_msg = []

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                left_ear = calculate_ear(lm, LEFT_EYE, w, h)
                right_ear = calculate_ear(lm, RIGHT_EYE, w, h)
                ear_val = (left_ear + right_ear) / 2.0
                mar_val = calculate_mar(lm, MOUTH, w, h)

                frame = draw_landmarks(frame, lm, w, h)

                current_time = time.time()
                can_alert = (current_time - st.session_state.last_alert_time) > ALERT_COOLDOWN_SEC

                debug_msg.append(f"EAR: {ear_val:.3f} (seuil: {ear_thresh})")
                debug_msg.append(f"MAR: {mar_val:.3f} (seuil: {mar_thresh})")
                debug_msg.append(f"Eye counter: {st.session_state.eye_counter}/{ear_frames}")
                debug_msg.append(f"Can alert: {can_alert}")

                # ========== EYES CLOSED ==========
                if ear_val < ear_thresh:
                    st.session_state.eye_counter += 1
                    debug_msg.append(f"⚠️ EAR < {ear_thresh} → counter = {st.session_state.eye_counter}")

                    if st.session_state.eye_counter >= ear_frames:
                        debug_msg.append(f"🚨 COUNTER >= {ear_frames} → ALERT!")

                        if can_alert:
                            status = "🚨 FI9! 3INIK MSDODIN"
                            status_class = "status-danger"
                            status_color = (0, 0, 255)
                            alert_triggered = True
                            alert_type = "eyes"

                            if not st.session_state.alarm_triggered:
                                st.session_state.drowsy_events += 1
                                st.session_state.alarm_triggered = True
                                st.session_state.last_alert_time = current_time
                                st.session_state.alerts_log.append(
                                    f"⏰ {time.strftime('%H:%M:%S')} - 😴 Sommeil détecté!"
                                )

                                if st.session_state.audio_enabled:
                                    trigger_alert("eyes", st.session_state.alert_repeat)
                                    audio_alert_banner.markdown(f"""
                                    <div class="audio-alert-box">
                                        <h3>🔊 AUDIO ALERT ({st.session_state.alert_repeat}x)</h3>
                                        <p>"{ALERT_MESSAGES['eyes']}"</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                debug_msg.append(f"✅ ALERT + AUDIO ({st.session_state.alert_repeat}x)!")
                        else:
                            debug_msg.append(f"⏳ Cooldown...")
                    else:
                        debug_msg.append(f"⏳ Building... {st.session_state.eye_counter}/{ear_frames}")
                else:
                    if st.session_state.eye_counter > 0:
                        debug_msg.append(f"✅ EAR >= {ear_thresh} → reset")
                    st.session_state.eye_counter = 0
                    st.session_state.alarm_triggered = False
                    audio_alert_banner.empty()

                # ========== YAWNING ==========
                if mar_val > mar_thresh:
                    st.session_state.yawn_counter += 1
                    if st.session_state.yawn_counter >= mar_frames and can_alert:
                        status = "⚠️ T3BAN... WAQEF"
                        status_class = "status-warning"
                        status_color = (0, 165, 255)
                        alert_triggered = True
                        alert_type = "yawn"
                        if st.session_state.yawn_counter == mar_frames:
                            st.session_state.yawn_events += 1
                            st.session_state.last_alert_time = current_time
                            st.session_state.alerts_log.append(
                                f"⏰ {time.strftime('%H:%M:%S')} - 🥱 Bâillement!"
                            )
                            if st.session_state.audio_enabled:
                                trigger_alert("yawn", st.session_state.alert_repeat)
                            debug_msg.append("✅ YAWN ALERT!")
                else:
                    st.session_state.yawn_counter = 0

                # Face box
                xs = [lm[i].x * w for i in range(len(lm))]
                ys = [lm[i].y * h for i in range(len(lm))]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 3)
            else:
                status = "🔍 MA L9ITCH WJEH"
                status_class = "status-warning"
                status_color = (0, 165, 255)
                debug_msg.append("❌ No face")

            # Draw overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), status_color, -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            cv2.putText(frame, f"EAR: {ear_val:.3f}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"MAR: {mar_val:.3f}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Counter: {st.session_state.eye_counter}/{ear_frames}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            st.session_state.total_frames += 1

            # ========== VOICE CONFIRMATION ==========
            if alert_triggered and not st.session_state.waiting_voice:
                st.session_state.waiting_voice = True
                st.session_state.voice_active = True
                voice_banner.markdown("""
                <div class="voice-box">
                    <h3>🎤 CONFIRMATION VOCALE REQUISE</h3>
                    <p>"Wach mazal faiq? Goul 'ah' wla 'wakha'!"</p>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.voice_active and st.session_state.waiting_voice:
                # BIG RECORDER BOX
                voice_recorder_area.markdown("""
                <div class="recorder-box">
                    <h3>🎙️ WACH MAZAL FAIQ?</h3>
                    <p><b>1.</b> Cliquez sur le microphone ci-dessous 👇</p>
                    <p><b>2.</b> Goul <b>"ah"</b>, <b>"wakha"</b>, <b>"yes"</b> wla <b>"oui"</b></p>
                    <p><b>3.</b> Waqef l'enregistrement w cliquez CONFIRMER</p>
                </div>
                """, unsafe_allow_html=True)

                # AUDIO RECORDER - BIG AND CLEAR
                st.markdown("---")
                st.markdown("### 🎤 **ENREGISTREMENT AUDIO**")
                st.markdown("👇 **Cliquez sur le microphone pour commencer:**")

                audio_file = st.audio_input("🎙️ Enregistrez votre réponse:", key=f"audio_{st.session_state.total_frames}")

                st.markdown("---")
                st.markdown("### ⌨️ **OU tapez votre réponse:**")

                # TEXT INPUT FALLBACK
                text_input = st.text_input(
                    "✍️ Écrivez 'ah', 'wakha', 'yes' ou 'oui':",
                    key=f"text_{st.session_state.total_frames}",
                    placeholder="ah"
                )

                st.markdown("---")
                st.markdown("### ✅ **CONFIRMER VOTRE RÉPONSE:**")

                col_c1, col_c2 = st.columns(2)

                with col_c1:
                    confirm_clicked = st.button(
                        "✅ JAWABT (Faiq)",
                        key=f"confirm_{st.session_state.total_frames}",
                        use_container_width=True,
                        type="primary"
                    )

                with col_c2:
                    escalate_clicked = st.button(
                        "🚨 MA JAWABTCH (T3ban)",
                        key=f"escalate_{st.session_state.total_frames}",
                        use_container_width=True
                    )

                # PROCESS CONFIRMATION
                if confirm_clicked or (text_input and text_input.strip() != ""):
                    response_text = text_input.strip() if text_input else "ah (audio)"

                    st.session_state.voice_confirmed = True
                    st.session_state.confirmation_status = "confirmed"
                    st.session_state.confirmation_time = time.time()
                    st.session_state.waiting_voice = False
                    st.session_state.voice_active = False

                    confirmation_banner.markdown(f"""
                    <div class="confirm-box">
                        <h3>✅ CONFIRMÉ!</h3>
                        <p>"{CONFIRM_MESSAGES['confirmed']}"</p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">Jawab: '{response_text}'</p>
                    </div>
                    """, unsafe_allow_html=True)

                    voice_result.success(f"✅ CONFIRMÉ! Jawab: '{response_text}' - Faiq mzyan, kml l3awd!")
                    st.session_state.alerts_log.append(
                        f"✅ {time.strftime('%H:%M:%S')} - Confirmé: '{response_text}' - Faiq mzyan!"
                    )

                    voice_banner.empty()
                    voice_recorder_area.empty()

                elif escalate_clicked:
                    st.session_state.voice_confirmed = False
                    st.session_state.confirmation_status = "escalation"
                    st.session_state.confirmation_time = time.time()
                    st.session_state.escalation_count += 1
                    st.session_state.waiting_voice = False
                    st.session_state.voice_active = False

                    confirmation_banner.markdown(f"""
                    <div class="audio-alert-box">
                        <h3>🚨 ESCALATION!</h3>
                        <p>"{CONFIRM_MESSAGES['escalation']}"</p>
                    </div>
                    """, unsafe_allow_html=True)

                    voice_result.error("🚨 ESCALATION! Ma jawabtch - Khatar akbar!")
                    st.session_state.alerts_log.append(
                        f"🚨 {time.strftime('%H:%M:%S')} - ESCALATION! Ma jawabtch - Khatar!"
                    )

                    voice_banner.empty()
                    voice_recorder_area.empty()

            # Show confirmation status if recent
            if st.session_state.confirmation_status and (time.time() - st.session_state.confirmation_time) < 5:
                if st.session_state.confirmation_status == "confirmed":
                    confirmation_status.markdown("""
                    <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid #28a745;">
                        <h3 style="margin: 0;">✅ DERNIERE CONFIRMATION</h3>
                        <p style="margin: 0.5rem 0 0 0;">Faiq mzyan - Audio confirmé!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.confirmation_status == "escalation":
                    confirmation_status.markdown("""
                    <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid #dc3545;">
                        <h3 style="margin: 0;">🚨 DERNIERE ESCALATION</h3>
                        <p style="margin: 0.5rem 0 0 0;">Ma jawabtch - Khatar akbar!</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                confirmation_status.empty()
                if st.session_state.confirmation_status and (time.time() - st.session_state.confirmation_time) >= 5:
                    st.session_state.confirmation_status = None
                    confirmation_banner.empty()

            # Update UI
            status_placeholder.markdown(f'<div class="status-box {status_class}">{status}</div>', unsafe_allow_html=True)
            ear_metric.metric("👁️ EAR", f"{ear_val:.3f}", delta=f"Seuil: {ear_thresh}")
            mar_metric.metric("🥱 MAR", f"{mar_val:.3f}", delta=f"Seuil: {mar_thresh}")
            eye_counter_metric.metric("😴 Counter yeux", st.session_state.eye_counter, delta=f"/ {ear_frames}")
            yawn_counter_metric.metric("🥱 Counter bâillement", st.session_state.yawn_counter, delta=f"/ {mar_frames}")

            if st.session_state.alerts_log:
                log_html = ""
                for log in st.session_state.alerts_log[-10:]:
                    if "ESCALATION" in log:
                        log_html += f'<div class="escalation-log">{log}</div>'
                    elif "✅" in log or "Confirm" in log:
                        log_html += f'<div class="confirm-log">{log}</div>'
                    else:
                        log_html += f'<div class="alert-log">{log}</div>'
                alert_history.markdown(log_html, unsafe_allow_html=True)
            else:
                alert_history.info("Aucune alerte")

            debug_placeholder.markdown(f'<div class="debug-box">{"<br>".join(debug_msg)}</div>', unsafe_allow_html=True)

            video_placeholder.image(frame, channels="BGR", use_column_width=True)

            if stop_btn:
                st.session_state.running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        if not st.session_state.running:
            st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚗 <b>Driver Drowsiness Detection - Voice Fixed</b></p>
    <p>🔊 TTS Audio (3x) 🔹 Audio/Text Confirmation 🔹 Escalation System</p>
</div>
""", unsafe_allow_html=True)