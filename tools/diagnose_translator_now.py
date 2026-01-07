# diagnostyka problemu translatora PJM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np

print("=" * 60)
print("DIAGNOSTYKA TRANSLATORA PJM")
print("=" * 60)

# 1. Sprawdz modele
print("\n[1] PLIKI MODELI:")
models_dir = Path("app/sign_language/models")
for f in models_dir.glob("*"):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name}: {size_kb:.1f} KB")

# 2. Sprawdz metadane
print("\n[2] METADANE MODELI:")
meta_path = models_dir / "model_meta.json"
if meta_path.exists():
    with open(meta_path) as meta_file:
        meta = json.load(meta_file)
    print(
        f"  model_meta.json: input_size={meta.get('input_size')}, hidden={meta.get('hidden_size')}"
    )

meta_ext_path = models_dir / "model_meta_extended.json"
if meta_ext_path.exists():
    with open(meta_ext_path) as meta_ext_file:
        meta_ext = json.load(meta_ext_file)
    print(
        f"  model_meta_extended.json: input_size={meta_ext.get('input_size')}, acc={meta_ext.get('test_acc'):.2%}"
    )

# 3. Sprawdz klasy
print("\n[3] KLASY W MODELACH:")
classes_path = models_dir / "classes.npy"
if classes_path.exists():
    classes = np.load(classes_path)
    print(f"  classes.npy: {len(classes)} klas")
    print(f"  Pierwsze 10: {list(classes[:10])}")

# 4. Sprawdz domyslne sciezki w translator.py
print("\n[4] DOMYSLNE SCIEZKI W TRANSLATOR:")
from app.sign_language.translator import (
    _DEFAULT_META_PATH,
    _DEFAULT_MODEL_PATH,
    BLOCK_SIZE,
    NUM_BLOCKS,
    SEQUENCE_INPUT_SIZE,
)

print(f"  _DEFAULT_MODEL_PATH: {_DEFAULT_MODEL_PATH}")
print(f"  _DEFAULT_META_PATH: {_DEFAULT_META_PATH}")
print(f"  BLOCK_SIZE: {BLOCK_SIZE}")
print(f"  NUM_BLOCKS: {NUM_BLOCKS}")
print(f"  SEQUENCE_INPUT_SIZE: {SEQUENCE_INPUT_SIZE}")

# 5. Sprawdz FeatureConfig
print("\n[5] FEATURE CONFIG:")
from app.sign_language.features import FeatureConfig

cfg = FeatureConfig()
print(f"  mirror_left: {cfg.mirror_left}")
print(f"  scale_by_mcp: {cfg.scale_by_mcp}")
print(f"  Czy ma extended_features: {hasattr(cfg, 'extended_features')}")

# 6. Sprawdz rozmiar cech generowanych przez FeatureExtractor
print("\n[6] TEST EKSTRAKCJI CECH:")
from app.sign_language.features import FeatureExtractor

extractor = FeatureExtractor()
dummy_landmarks = np.random.rand(21, 3).astype(np.float32)
features = extractor.extract(dummy_landmarks)
print(f"  Rozmiar cech z 21 landmarkow: {len(features)} (oczekiwano 63 lub 82)")

# 7. Test translatora
print("\n[7] TEST TRANSLATORA:")
try:
    from app.sign_language.translator import SignTranslator

    translator = SignTranslator()
    print(f"  model_input_size: {translator.model_input_size}")
    print(f"  classes: {len(translator.classes)}")
    print(f"  buffer_size: {translator.buffer_size}")
    print(f"  confidence_entry: {translator.confidence_entry}")

    # sprawdz wagi modelu
    w0 = translator.model.network[0].weight
    print(f"  Wagi warstwy 0: shape={w0.shape}")
except Exception as e:
    print(f"  BLAD: {e}")

# 8. Sprawdz przetworzone dane
print("\n[8] PRZETWORZONE DANE:")
processed_dir = Path("app/sign_language/data/processed")
for split in ["train", "val", "test"]:
    npz_path = processed_dir / f"{split}.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        print(f"  {split}.npz: X={data['X'].shape}")

processed_ext_dir = Path("app/sign_language/data/processed_extended")
for split in ["train", "val", "test"]:
    npz_path = processed_ext_dir / f"{split}.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        print(f"  {split}.npz (extended): X={data['X'].shape}")

print("\n" + "=" * 60)
print("WNIOSKI:")
print("=" * 60)
print(
    """
1. Jesli BLOCK_SIZE=63 a model ma input_size=189, uzywamy STAREGO modelu
2. Extended model wymaga BLOCK_SIZE=82 i input_size=246
3. Jesli FeatureExtractor generuje 63 cechy a model oczekuje 246 - NIESPOJNOSC!
4. Rozwiazanie: albo zmienic na extended model, albo przetrenowac stary
"""
)
