import sys
import traceback

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Trying to import mediapipe...")
try:
    import mediapipe as mp

    print("mediapipe.__version__ =", getattr(mp, "__version__", "unknown"))
    print("mediapipe.__file__ =", getattr(mp, "__file__", "unknown"))
    print("Import OK")
except Exception:
    print("Import FAILED:")
    traceback.print_exc()
