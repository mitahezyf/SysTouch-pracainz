# sprawdz klasy w datasecie
import pandas as pd

df = pd.read_csv("app/sign_language/data/raw/PJM-vectors.csv")
labels = sorted(df["sign_label"].unique())

print("=== KLASY W DATASECIE ===")
for i, label in enumerate(labels):
    count = len(df[df["sign_label"] == label])
    print(f"{i+1:2}. {label:5} - {count} probek")

print()
print(f"Lacznie klas: {len(labels)}")
print(f"Lacznie probek: {len(df)}")

# sprawdz czy A, B, C sa obecne
abc = ["A", "B", "C"]
for letter in abc:
    if letter in labels:
        print(f"[OK] {letter} jest w datasecie")
    else:
        print(f"[BRAK] {letter} NIE MA w datasecie!")
