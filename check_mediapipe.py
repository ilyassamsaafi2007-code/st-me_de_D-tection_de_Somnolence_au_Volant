"""
Diagnostic script to check MediaPipe installation
Run this first to see what's installed!
"""
import sys
print("Python:", sys.version)
print("-" * 50)

try:
    import mediapipe as mp
    print("✅ MediaPipe version:", mp.__version__)
    print("MediaPipe location:", mp.__file__)

    # Check available attributes
    print("\nAvailable attributes in mediapipe:")
    attrs = [a for a in dir(mp) if not a.startswith('_')]
    for a in attrs:
        print(f"  - {a}")

    # Try importing solutions
    print("\nTrying mp.solutions.face_mesh...")
    try:
        mp_face_mesh = mp.solutions.face_mesh
        print("✅ mp.solutions.face_mesh OK")
    except AttributeError as e:
        print(f"❌ Error: {e}")

    # Try importing drawing_utils
    print("\nTrying mp.solutions.drawing_utils...")
    try:
        mp_drawing = mp.solutions.drawing_utils
        print("✅ mp.solutions.drawing_utils OK")
    except AttributeError as e:
        print(f"❌ Error: {e}")

except ImportError as e:
    print(f"❌ MediaPipe not installed: {e}")
    print("Install with: pip install mediapipe==0.10.9")

print("\n" + "=" * 50)
print("Checking other packages...")

try:
    import streamlit
    print("✅ Streamlit:", streamlit.__version__)
except:
    print("❌ Streamlit not installed")

try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except:
    print("❌ OpenCV not installed")

try:
    import numpy
    print("✅ NumPy:", numpy.__version__)
except:
    print("❌ NumPy not installed")
