"""
=========================================================
   04 - Voice Confirmation System (Hugging Face Whisper)
=========================================================
Had module kayzid "dakika dyal ta2akod" 3la 3ad signal dyal drowsiness:

    1) Ila tla3 alert (3inik msdodin / t3ban)
    2) Système kaysolek: "Wach mazal fai9?"
    3) Microphone kaysajel jawabek (3 secondes)
    4) Whisper (model mn Hugging Face) kay7awl sawtek l text
    5) Ila l9a kelma dyal confirmation (ah, wakha, na3am...) -> mzyan
       Ila ma sma3ch walo / jawab machi wadih -> ESCALATION (khatar akbar)

Installation:
    pip install transformers torch sounddevice scipy numpy

Note: Whisper-tiny mste3mal hna (bdal base) hit khafif w asra3 3la CPU.
Awal run ghadi y7amel l model mn internet (~150MB), mn b3d kayban cached.
"""

import sounddevice as sd
import numpy as np
from transformers import pipeline

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
RECORD_SECONDS = 3

import re

# Kelmat dyal confirmation (3arabia + darija + transliteration + variants)
CONFIRM_KEYWORDS = [
    "ah", "aywa", "wakha", "iyeh", "na3am", "fai9", "labas", "ok", "yes",
    "نعم", "واخا", "وخا", "فايق", "اه", "ايه", "لاباس", "واخ", "نعام",
]


def is_confirmed(text):
    """Kay check wach l text fih chi kelma dyal confirmation"""
    # nzilo 3alamat tar9im (؟!.,) w nrj3o letters bla flouss zayda
    cleaned = re.sub(r"[؟?!.,]", "", text).strip().lower()
    return any(keyword in cleaned for keyword in CONFIRM_KEYWORDS)

print("Tahmil Whisper model mn Hugging Face (mra wahda ghir)...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",   # khafif, khdam mzyan f CPU
)
print("[OK] Whisper jahz.")


def record_audio(duration=RECORD_SECONDS, fs=SAMPLE_RATE):
    """Kaysajel sawt mn microphone l 'duration' secondes"""
    print(f"🎤 Goul chi haja... ({duration}s)")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return recording.flatten(), fs


def transcribe_audio(audio, fs):
    """Kay7awl l audio l text b Whisper"""
    print("⏳ Kan7allel sawt b Whisper... (momkin yakhod chi 10-30 secondes f CPU)")
    result = asr_pipeline(
        {"array": audio, "sampling_rate": fs},
        generate_kwargs={"language": "arabic", "max_new_tokens": 40},
    )
    print("✅ Tahlil kml.")
    return result["text"].strip()


def listen_and_confirm():
    """
    Function principale: kaysajel + kay7awl + kay check.
    Kayrj3: (confirmed: bool, text: str)
    """
    try:
        audio, fs = record_audio()
        text = transcribe_audio(audio, fs)
        print(f"📝 Smait: '{text}'")

        confirmed = is_confirmed(text)
        if confirmed:
            print("✅ Confirmed - sa2i9 mzyan, fai9")
        else:
            print("⚠️ Ma fhamtch jawab wadih wla skat - ESCALATION!")
        return confirmed, text

    except Exception as e:
        print(f"Khta2 f tasjil/ta7lil sawt: {e}")
        return False, ""


# ---------- TEST MANUEL ----------
if __name__ == "__main__":
    print("=" * 60)
    print("TEST - Voice Confirmation System")
    print("=" * 60)
    print("Tjarib: ghadi nsajlo sawtek 3 secondes, goul 'ah' wla 'wakha'")
    input("\nDos Enter mnin tkoun mjahd...")

    confirmed, text = listen_and_confirm()

    print("\n" + "=" * 60)
    print(f"Resultat: {'✅ CONFIRMED' if confirmed else '⚠️ NOT CONFIRMED'}")
    print(f"Text msmou3: '{text}'")
    print("=" * 60)