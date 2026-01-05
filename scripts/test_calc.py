"""Test manual calculations."""

import sys

sys.path.insert(0, ".")

from app.gesture_engine.actions.volume_helpers import quantize_pct

# test problematycznych wartosci
test_cases = [
    (67.6, 5),
    (50.0, 100),
    (12.99, 5),
]

print("Testing quantize_pct:")
for val, step in test_cases:
    result = quantize_pct(val, step)
    int_val = int(val)
    division = int_val / step
    rounded = round(division)
    expected = int(rounded * step)
    print(f"quantize_pct({val}, {step}) = {result}")
    print(f"  int({val}) = {int_val}")
    print(f"  {int_val}/{step} = {division}")
    print(f"  round({division}) = {rounded}")
    print(f"  {rounded}*{step} = {expected}")
    print()
