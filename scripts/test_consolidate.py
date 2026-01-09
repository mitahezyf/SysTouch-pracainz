# test konsolidacji
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.consolidate_collected import consolidate_sessions

print("=== Test konsolidacji ===")
consolidate_sessions()
print("=== Koniec ===")
