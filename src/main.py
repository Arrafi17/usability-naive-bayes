import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "ulasan_usability.csv")
OUT = os.path.join(BASE, "hasil")
os.makedirs(OUT, exist_ok=True)

STOPWORDS = {"dan", "yang", "di", "ke", "dari", "untuk", "pada", "dengan", "ini", "itu", "saya", "nya", "tapi", "namun", "atau", "pun", "walau"}
URUTAN = ["rendah", "sedang", "tinggi"]


def bersihkan(teks):
    teks = teks.lower()
    teks = re.sub(r"[^a-z\s]", " ", teks)
    kata = [k for k in teks.split() if k not in STOPWORDS]
    return " ".join(kata)


df = pd.read_csv(DATA)
df["bersih"] = df["ulasan"].apply(bersihkan)

x_train, x_test, y_train, y_test = train_test_split(
    df["bersih"], df["tingkat_usability"], test_size=0.2, random_state=42, stratify=df["tingkat_usability"]
)

vektor = TfidfVectorizer(ngram_range=(1, 2))
x_train_v = vektor.fit_transform(x_train)
x_test_v = vektor.transform(x_test)

model = MultinomialNB(alpha=1.0)
model.fit(x_train_v, y_train)
prediksi = model.predict(x_test_v)

print("Jumlah data latih :", len(x_train))
print("Jumlah data uji   :", len(x_test))
print("Akurasi           : {:.2%}".format(accuracy_score(y_test, prediksi)))
print()
print(classification_report(y_test, prediksi, labels=URUTAN, zero_division=0))

fig, ax = plt.subplots(figsize=(5, 4))
df["tingkat_usability"].value_counts().reindex(URUTAN).plot(kind="bar", color=["#e76f51", "#f4a261", "#2a9d8f"], ax=ax)
ax.set_title("Distribusi Kelas Usability")
ax.set_xlabel("Tingkat usability")
ax.set_ylabel("Jumlah ulasan")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "distribusi_kelas.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, prediksi, labels=URUTAN, cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix Naive Bayes")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
plt.close(fig)

contoh = [
    "menu mudah ditemukan dan tampilan rapi, saya puas memakainya",
    "aplikasi kadang lambat tapi masih bisa dipakai dengan baik",
    "tampilan berantakan, saya sering salah menekan tombol dan kecewa",
]
hasil = model.predict(vektor.transform([bersihkan(c) for c in contoh]))
print("Contoh prediksi ulasan baru:")
for teks, label in zip(contoh, hasil):
    print(f"- {teks}\n  -> {label}")
