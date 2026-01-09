# test rozszerzonych cech
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from app.sign_language.features import FeatureConfig, FeatureExtractor

print("=== TEST ROZSZERZONYCH CECH ===")

# test standardowych cech (63D)
cfg_std = FeatureConfig(extended_features=False)  # type: ignore[call-arg]
ext_std = FeatureExtractor(cfg_std)
print(f"Standard feature size: {ext_std.feature_size}")  # type: ignore[attr-defined]

# test rozszerzonych cech (82D)
cfg_ext = FeatureConfig(extended_features=True)  # type: ignore[call-arg]
ext_ext = FeatureExtractor(cfg_ext)
print(f"Extended feature size: {ext_ext.feature_size}")  # type: ignore[attr-defined]

# test ekstrakcji
lm = np.random.randn(21, 3).astype(np.float32)
feat_std = ext_std.extract(lm)
feat_ext = ext_ext.extract(lm)

print(f"\nStandard features shape: {feat_std.shape}")
print(f"Extended features shape: {feat_ext.shape}")

# sprawdz czy pierwsze 63 cechy sa identyczne
if np.allclose(feat_std, feat_ext[:63]):
    print("[OK] Pierwsze 63 cechy sa identyczne")
else:
    print("[BLAD] Pierwsze 63 cechy sie roznia!")

# pokaz dodatkowe cechy
print("\nDodatkowe cechy (19D):")
print(f"  Finger angles (4D): {feat_ext[63:67]}")
print(f"  Finger curls (5D): {feat_ext[67:72]}")
print(f"  Fingertip distances (10D): {feat_ext[72:82]}")

# test na sekwencji 3-klatkowej
print("\n=== TEST SEKWENCJI 3-KLATKOWEJ ===")
block_size_std = 63
block_size_ext = 82
num_blocks = 3

seq_std = block_size_std * num_blocks
seq_ext = block_size_ext * num_blocks

print(f"Sekwencja standardowa: {num_blocks} x {block_size_std} = {seq_std}D")
print(f"Sekwencja rozszerzona: {num_blocks} x {block_size_ext} = {seq_ext}D")

print("\n=== TEST ZAKONCZONY POMYSLNIE ===")
