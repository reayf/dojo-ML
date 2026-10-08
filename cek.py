"""
cek.py — asisten latihan Dojo Anti-Ngangong.

Dipakai di notebook latihan:
    from cek import cek, hint, jawaban, progres, skor

    cek('1.2', ukuran)      -> periksa jawaban latihan 1.2
    hint('1.2')             -> petunjuk bertahap (panggil lagi untuk petunjuk berikutnya)
    jawaban('1.2')          -> kunci jawaban (setelah lihat, KETIK ULANG sendiri)
    progres()               -> latihan mana yang sudah lulus
    skor('mock1')           -> nilai file submission seperti leaderboard Kaggle

Begitu di-import, error di notebook juga otomatis diberi penjelasan bahasa Indonesia.
File ini tidak perlu diubah.
"""
import os
import re
import json
import random
import difflib
import datetime

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_BASE, "data")
_PROGRES = os.path.join(_BASE, ".progres.json")
_PUJIAN = ["Mantap!", "Gas terus!", "Nah, gitu!", "Rapi!", "Aman!", "Keren!", "Lanjut!"]


# ============================================================ utilitas

def _csv(*bagian):
    return pd.read_csv(os.path.join(_DATA, *bagian))


_CACHE = {}


def _data(nama):
    """Salinan segar data asli (tidak terpengaruh perubahan di notebook)."""
    if nama not in _CACHE:
        path = {"titanic": ("titanic", "train.csv"), "titanic_test": ("titanic", "test.csv"),
                "penguins": ("penguins", "penguins.csv")}[nama]
        _CACHE[nama] = _csv(*path)
    return _CACHE[nama].copy()


def _ns():
    """Variabel-variabel di notebook yang sedang berjalan."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip.user_ns if ip is not None else {}
    except Exception:
        return {}


def _isnan(v):
    try:
        return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


def _sama_nilai(x, y, tol=1e-6):
    x = np.asarray(x, dtype=object)
    y = np.asarray(y, dtype=object)
    if x.shape != y.shape:
        return False
    for u, v in zip(x.ravel(), y.ravel()):
        if _isnan(u) and _isnan(v):
            continue
        if _isnan(u) or _isnan(v):
            return False
        try:
            fu, fv = float(u), float(v)
            if abs(fu - fv) > tol * max(1.0, abs(fv)):
                return False
        except (TypeError, ValueError):
            if str(u) != str(v):
                return False
    return True


def _idx(s):
    return [str(i) for i in s.index]


def _sama_series(a, b, urut=False):
    """a = jawaban siswa, b = yang diharapkan. urut=True -> urutan index harus sama."""
    if not isinstance(a, pd.Series) or len(a) != len(b):
        return False
    if not urut:
        try:
            a, b = a.sort_index(), b.sort_index()
        except TypeError:
            pass
    return _idx(a) == _idx(b) and _sama_nilai(a.to_numpy(), b.to_numpy())


def _sama_df(a, b, cek_kolom=True):
    if not isinstance(a, pd.DataFrame) or a.shape != b.shape:
        return False
    if cek_kolom and list(map(str, a.columns)) != list(map(str, b.columns)):
        return False
    return _sama_nilai(a.to_numpy(dtype=object), b.to_numpy(dtype=object))


def _baca_progres():
    try:
        with open(_PROGRES, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"lulus": {}, "skor": {}}


def _tulis_progres(p):
    try:
        with open(_PROGRES, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=1)
    except Exception:
        pass


def _catat_lulus(no):
    p = _baca_progres()
    p.setdefault("lulus", {})[no] = datetime.datetime.now().isoformat(timespec="seconds")
    _tulis_progres(p)


# ============================================================ bank soal

SOAL = {}


def _soal(no, judul, hints, jawaban_kode):
    def daftar(fungsi):
        SOAL[no] = dict(judul=judul, hints=hints, jawaban=jawaban_kode.strip("\n"), cek=fungsi)
        return fungsi
    return daftar


class Gagal(Exception):
    """Dipakai fungsi pemeriksa untuk memberi pesan salah yang spesifik."""


def _pastikan(kondisi, pesan):
    if not kondisi:
        raise Gagal(pesan)


# ---------------------------------------------------------------- Notebook 01

@_soal("1.1", "8 baris pertama", [
    "Fungsi untuk melihat baris-baris PALING ATAS artinya 'kepala' dalam bahasa Inggris.",
    "Isi titik-titiknya dengan: head",
], "hasil = train.head(8)")
def _(hasil):
    t = _data("titanic")
    _pastikan(isinstance(hasil, pd.DataFrame), "Hasilnya harus tabel (DataFrame). Pakai train.head(8).")
    _pastikan(len(hasil) == 8, f"Jumlah barisnya {len(hasil)}, harusnya 8.")
    _pastikan(list(hasil["PassengerId"]) == list(t.head(8)["PassengerId"]),
              "Barisnya bukan 8 baris PERTAMA. tail() itu dari bawah, head() dari atas.")


@_soal("1.2", "ukuran data", [
    "Ukuran tabel = (jumlah baris, jumlah kolom). Namanya 'bentuk' dalam bahasa Inggris.",
    "Isi dengan: shape  (tanpa kurung!)",
], "ukuran = train.shape")
def _(ukuran):
    _pastikan(not callable(ukuran), "shape itu atribut, bukan fungsi. Tulis train.shape tanpa kurung.")
    _pastikan(isinstance(ukuran, tuple), "Hasilnya harus tuple (baris, kolom).")
    t = _ns().get("train")
    boleh = [(712, 12)] + ([t.shape] if isinstance(t, pd.DataFrame) else [])
    _pastikan(ukuran in boleh, f"Dapat {ukuran}. Harusnya (712, 12) = (baris, kolom).")


@_soal("1.3", "jumlah kolom", [
    "shape itu (baris, kolom). Kolom ada di posisi kedua. Python mulai menghitung dari 0.",
    "train.shape[1]  atau  len(train.columns)",
], "n_kolom = train.shape[1]")
def _(n_kolom):
    t = _ns().get("train")
    boleh = [12] + ([t.shape[1]] if isinstance(t, pd.DataFrame) else [])
    _pastikan(n_kolom != 712, "712 itu jumlah BARIS. Kolom ada di shape[1].")
    _pastikan(n_kolom in boleh, f"Dapat {n_kolom}. Harusnya 12.")


@_soal("1.4", "ambil kolom Fare", [
    "Ambil satu kolom: df['nama_kolom'] (nama kolom pakai tanda kutip).",
    "train['Fare']",
], "harga = train['Fare']")
def _(harga):
    _pastikan(isinstance(harga, pd.Series), "Satu kolom pakai satu kurung siku: train['Fare'].")
    _pastikan(harga.name == "Fare", f"Kamu mengambil kolom '{harga.name}', bukan 'Fare'.")


@_soal("1.5", "ambil 3 kolom", [
    "Beberapa kolom = DUA kurung siku: df[['a', 'b', 'c']]. Kurung luar untuk memilih, kurung dalam untuk daftar.",
    "train[['Name', 'Pclass', 'Survived']]",
], "tiga_kolom = train[['Name', 'Pclass', 'Survived']]")
def _(tiga_kolom):
    _pastikan(isinstance(tiga_kolom, pd.DataFrame), "Hasilnya harus DataFrame. Pakai dua kurung siku [[ ]].")
    _pastikan(set(tiga_kolom.columns) == {"Name", "Pclass", "Survived"},
              f"Kolommu: {list(tiga_kolom.columns)}. Harusnya Name, Pclass, Survived.")
    _pastikan(len(tiga_kolom) == 712, "Jumlah barisnya harus tetap 712 (semua baris).")


@_soal("1.6", "rata-rata Fare", [
    "Rata-rata dalam bahasa Inggris = mean.",
    "train['Fare'].mean()",
], "rata_fare = train['Fare'].mean()")
def _(rata_fare):
    f = _data("titanic")["Fare"]
    _pastikan(not callable(rata_fare), "Jangan lupa kurung: .mean()")
    _pastikan(not np.isclose(rata_fare, f.median()), "Itu MEDIAN (nilai tengah). Yang diminta rata-rata: mean().")
    _pastikan(np.isclose(rata_fare, f.mean()), f"Dapat {rata_fare}, harusnya sekitar {f.mean():.2f}.")


@_soal("1.7", "jumlah penumpang per kelas", [
    "Fungsi untuk MENGHITUNG berapa kali tiap nilai muncul. Namanya value_...",
    "train['Pclass'].value_counts()",
], "per_kelas = train['Pclass'].value_counts()")
def _(per_kelas):
    vc = _data("titanic")["Pclass"].value_counts()
    _pastikan(isinstance(per_kelas, pd.Series), "Hasilnya harus Series dari value_counts().")
    _pastikan(per_kelas.sum() > 1.5, "Itu proporsi (normalize=True). Yang diminta JUMLAH, tanpa normalize.")
    _pastikan(_sama_series(per_kelas, vc), "Angkanya belum cocok. Pastikan kolomnya Pclass.")


@_soal("1.8", "proporsi selamat", [
    "value_counts punya pilihan untuk mengubah jumlah jadi proporsi (0 sampai 1).",
    "value_counts(normalize=True)",
], "proporsi_selamat = train['Survived'].value_counts(normalize=True)")
def _(proporsi_selamat):
    vc = _data("titanic")["Survived"].value_counts(normalize=True)
    _pastikan(isinstance(proporsi_selamat, pd.Series), "Hasilnya harus Series.")
    _pastikan(abs(proporsi_selamat.sum() - 1) < 1e-6, "Totalnya harus 1 (proporsi). Isi dengan normalize=True.")
    _pastikan(_sama_series(proporsi_selamat, vc), "Angkanya belum cocok.")


@_soal("1.9", "filter Fare > 100", [
    "Pola filter: df[ kondisi ]. Kondisinya: train['Fare'] > 100",
    "train[train['Fare'] > 100]",
], "mahal = train[train['Fare'] > 100]")
def _(mahal):
    t = _data("titanic")
    _pastikan(not isinstance(mahal, pd.Series), "Itu masih kondisi True/False. Bungkus dengan train[ ... ].")
    _pastikan(isinstance(mahal, pd.DataFrame), "Hasilnya harus DataFrame.")
    exp = set(t.loc[t["Fare"] > 100, "PassengerId"])
    got = set(mahal["PassengerId"])
    _pastikan(got == exp, f"Dapat {len(got)} baris, harusnya {len(exp)} (Fare > 100, bukan >=).")


@_soal("1.10", "laki-laki yang selamat", [
    "Dua kondisi: (train['Sex'] == 'male') dan (train['Survived'] == 1). Gabungkan dengan &.",
    "Ingat: sama dengan itu ==, bukan =. Teks pakai kutip: 'male'.",
], "pria_selamat = train[(train['Sex'] == 'male') & (train['Survived'] == 1)]")
def _(pria_selamat):
    t = _data("titanic")
    _pastikan(isinstance(pria_selamat, pd.DataFrame), "Hasilnya harus DataFrame.")
    exp = set(t.loc[(t["Sex"] == "male") & (t["Survived"] == 1), "PassengerId"])
    got = set(pria_selamat["PassengerId"])
    _pastikan(got == exp, f"Dapat {len(got)} baris, harusnya {len(exp)}.")


@_soal("1.11", "5 penumpang termuda", [
    "sort_values mengurutkan dari kecil ke besar (default). Isi dengan nama kolom umur.",
    "train.sort_values('Age').head(5)",
], "termuda = train.sort_values('Age').head(5)")
def _(termuda):
    t = _data("titanic")
    _pastikan(isinstance(termuda, pd.DataFrame) and len(termuda) == 5, "Hasilnya harus DataFrame 5 baris.")
    exp = list(t.sort_values("Age")["Age"].head(5))
    _pastikan(_sama_nilai(list(termuda["Age"]), exp), f"Umurnya {list(termuda['Age'])}, harusnya {exp}.")


@_soal("1.12", "peluang selamat per kelas", [
    "Pola: df.groupby(KELOMPOK)[KOLOM].mean(). Kelompoknya Pclass, kolomnya Survived.",
    "train.groupby('Pclass')['Survived'].mean()",
], "selamat_per_kelas = train.groupby('Pclass')['Survived'].mean()")
def _(selamat_per_kelas):
    exp = _data("titanic").groupby("Pclass")["Survived"].mean()
    _pastikan(_sama_series(selamat_per_kelas, exp), "Belum cocok. Kelompok = 'Pclass', kolom yang dirata-rata = 'Survived'.")


@_soal("1.13", "kolom IsAlone", [
    "Kondisinya: FamilySize sama dengan 1. Pakai ==.",
    "(train['FamilySize'] == 1).astype(int)",
], "train['IsAlone'] = (train['FamilySize'] == 1).astype(int)")
def _(train):
    t = _data("titanic")
    _pastikan("IsAlone" in train.columns, "Kolom IsAlone belum ada.")
    exp = ((t["SibSp"] + t["Parch"]) == 0).astype(int)
    _pastikan(_sama_nilai(train["IsAlone"].to_numpy(), exp.to_numpy()), "Nilai IsAlone belum benar (1 kalau FamilySize == 1).")


@_soal("1.14", "rata-rata Fare per Embarked, urut terbesar", [
    "Gabungan dua jurus: groupby(...)[...].mean() lalu .sort_values(ascending=False).",
    "train.groupby('Embarked')['Fare'].mean().sort_values(...)",
], "jawab = train.groupby('Embarked')['Fare'].mean().sort_values(ascending=False)")
def _(jawab):
    exp = _data("titanic").groupby("Embarked")["Fare"].mean().sort_values(ascending=False)
    _pastikan(isinstance(jawab, pd.Series), "Hasilnya harus Series.")
    _pastikan(_sama_series(jawab, exp), "Angkanya belum cocok (kelompok = Embarked, kolom = Fare, fungsi = mean).")
    _pastikan(_sama_series(jawab, exp, urut=True), "Angkanya benar, tapi urutannya harus dari TERBESAR (ascending=False).")


@_soal("1.15", "jumlah perempuan kelas 1", [
    "Filter dulu (dua kondisi), lalu hitung barisnya dengan len(...).",
    "len(train[(train['Sex'] == 'female') & (train['Pclass'] == 1)])",
], "n_wanita_k1 = len(train[(train['Sex'] == 'female') & (train['Pclass'] == 1)])")
def _(n_wanita_k1):
    t = _data("titanic")
    exp = int(((t["Sex"] == "female") & (t["Pclass"] == 1)).sum())
    _pastikan(not isinstance(n_wanita_k1, pd.DataFrame), "Itu tabelnya. Yang diminta JUMLAH baris: len(...).")
    _pastikan(n_wanita_k1 == exp, f"Dapat {n_wanita_k1}, harusnya {exp}.")


# ---------------------------------------------------------------- Notebook 02 (EDA)

@_soal("2.1", "daftar kolom numerik", [
    "select_dtypes(include=...) memilih kolom berdasarkan tipe. Untuk semua jenis angka pakai kata 'number'.",
    "include='number'",
], "kolom_angka = train.select_dtypes(include='number').columns.tolist()")
def _(kolom_angka):
    exp = ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    _pastikan(set(kolom_angka) == set(exp), f"Dapat {list(kolom_angka)}. Harusnya {exp}.")


@_soal("2.2", "daftar kolom non-numerik", [
    "Kebalikan include adalah exclude.",
    "exclude='number'",
], "kolom_teks = train.select_dtypes(exclude='number').columns.tolist()")
def _(kolom_teks):
    exp = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
    _pastikan(set(kolom_teks) == set(exp), f"Dapat {list(kolom_teks)}. Harusnya {exp}.")


@_soal("2.3", "jumlah missing per kolom", [
    "Dua langkah: tandai yang kosong (isnull) lalu jumlahkan (sum).",
    "train.isnull().sum()",
], "jumlah_kosong = train.isnull().sum()")
def _(jumlah_kosong):
    exp = _data("titanic").isnull().sum()
    _pastikan(isinstance(jumlah_kosong, pd.Series), "Hasilnya harus Series.")
    _pastikan(_sama_series(jumlah_kosong, exp), "Belum cocok. Urutannya: .isnull() dulu, baru .sum().")


@_soal("2.4", "persen missing per kolom", [
    "Rata-rata dari True/False = proporsi True. Jadi pakai mean(), lalu kali 100.",
    "train.isnull().mean() * 100",
], "persen_kosong = train.isnull().mean() * 100")
def _(persen_kosong):
    exp = _data("titanic").isnull().mean() * 100
    _pastikan(isinstance(persen_kosong, pd.Series), "Hasilnya harus Series.")
    _pastikan(persen_kosong.max() > 1.5, "Angkanya masih 0–1. Kalikan 100 supaya jadi persen.")
    _pastikan(_sama_series(persen_kosong, exp), "Belum cocok. Pakai .mean() (bukan .sum()), lalu * 100.")


@_soal("2.5", "hanya kolom yang ada missing", [
    "Filter Series sama seperti filter DataFrame: s[kondisi]. Kondisinya: jumlah_kosong > 0",
    "jumlah_kosong[jumlah_kosong > 0]",
], "ada_kosong = jumlah_kosong[jumlah_kosong > 0]")
def _(ada_kosong):
    _pastikan(isinstance(ada_kosong, pd.Series), "Hasilnya harus Series.")
    _pastikan(set(ada_kosong.index) == {"Age", "Cabin", "Embarked"},
              f"Dapat {list(ada_kosong.index)}. Harusnya Age, Cabin, Embarked.")


@_soal("2.6", "jumlah baris duplikat", [
    "Fungsi yang menandai baris kembar: duplicated()",
    "train.duplicated().sum()",
], "n_duplikat = train.duplicated().sum()")
def _(n_duplikat):
    _pastikan(not isinstance(n_duplikat, (pd.Series, pd.DataFrame)), "Jumlahkan dulu dengan .sum().")
    _pastikan(int(n_duplikat) == 0, f"Dapat {n_duplikat}. Data Titanic ini tidak punya duplikat (0).")


@_soal("2.7", "akurasi tebak mayoritas", [
    "Proporsi kelas terbesar = nilai maksimum dari value_counts(normalize=True).",
    ".max()",
], "baseline = train['Survived'].value_counts(normalize=True).max()")
def _(baseline):
    exp = _data("titanic")["Survived"].value_counts(normalize=True).max()
    _pastikan(np.isclose(baseline, exp), f"Dapat {baseline}, harusnya sekitar {exp:.3f}.")


@_soal("2.8", "histogram Fare", [
    "Parameter x diisi NAMA KOLOM dalam kutip.",
    "x='Fare'",
], "ax = sns.histplot(data=train, x='Fare')")
def _(ax):
    _pastikan(hasattr(ax, "get_xlabel"), "Simpan hasil sns.histplot(...) ke variabel ax.")
    _pastikan(ax.get_xlabel() == "Fare", f"Sumbu x-nya '{ax.get_xlabel()}', harusnya 'Fare'.")
    _pastikan(len(ax.patches) > 0, "Tidak ada batang histogram. Pakai sns.histplot.")


@_soal("2.9", "jumlah outlier Age (IQR)", [
    "Q1 = kuartil 25% -> 0.25. Q3 = kuartil 75% -> 0.75.",
    "quantile(0.25) dan quantile(0.75)",
], """Q1 = train['Age'].quantile(0.25)
Q3 = train['Age'].quantile(0.75)
IQR = Q3 - Q1
bawah = Q1 - 1.5 * IQR
atas = Q3 + 1.5 * IQR
n_outlier_age = ((train['Age'] < bawah) | (train['Age'] > atas)).sum()""")
def _(n_outlier_age):
    a = _data("titanic")["Age"]
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    exp = int(((a < q1 - 1.5 * iqr) | (a > q3 + 1.5 * iqr)).sum())
    _pastikan(int(n_outlier_age) == exp, f"Dapat {n_outlier_age}, harusnya {exp}. Cek angka kuartilnya (0.25 & 0.75).")


@_soal("2.10", "jumlah nilai unik kolom teks", [
    "Fungsi menghitung banyaknya nilai unik: nunique()",
    "train[cat_cols].nunique()",
], "unik_teks = train[cat_cols].nunique()")
def _(unik_teks):
    t = _data("titanic")
    exp = t[["Name", "Sex", "Ticket", "Cabin", "Embarked"]].nunique()
    _pastikan(isinstance(unik_teks, pd.Series), "Hasilnya harus Series. Pastikan sel contoh cat_cols sudah dijalankan.")
    _pastikan(_sama_series(unik_teks, exp), "Belum cocok. Pakai nunique() (bukan unique()).")


@_soal("2.11", "peluang selamat per pelabuhan", [
    "Kelompoknya kolom Embarked; fungsinya rata-rata.",
    "train.groupby('Embarked')['Survived'].mean()",
], "selamat_per_pelabuhan = train.groupby('Embarked')['Survived'].mean()")
def _(selamat_per_pelabuhan):
    exp = _data("titanic").groupby("Embarked")["Survived"].mean()
    _pastikan(_sama_series(selamat_per_pelabuhan, exp), "Belum cocok. groupby('Embarked'), lalu ['Survived'].mean().")


@_soal("2.12", "rata-rata Fare per status selamat", [
    "Kolom yang dirata-rata: Fare.",
    "train.groupby('Survived')['Fare'].mean()",
], "fare_per_target = train.groupby('Survived')['Fare'].mean()")
def _(fare_per_target):
    exp = _data("titanic").groupby("Survived")["Fare"].mean()
    _pastikan(_sama_series(fare_per_target, exp), "Belum cocok. Kolom di dalam [ ] harus 'Fare'.")


@_soal("2.13", "countplot Sex per Survived", [
    "x = kolom yang dihitung (Sex). hue = kolom pewarna (Survived).",
    "x='Sex', hue='Survived'",
], "ax = sns.countplot(data=train, x='Sex', hue='Survived')")
def _(ax):
    _pastikan(hasattr(ax, "get_xlabel"), "Simpan hasil sns.countplot(...) ke variabel ax.")
    _pastikan(ax.get_xlabel() == "Sex", f"Sumbu x-nya '{ax.get_xlabel()}', harusnya 'Sex'.")
    leg = ax.get_legend()
    _pastikan(leg is not None and leg.get_title().get_text() == "Survived", "Warna (hue) harus 'Survived'.")


@_soal("2.14", "korelasi dengan target", [
    "numeric_only=True supaya kolom teks diabaikan. Urut terbesar -> ascending=False.",
    "train.corr(numeric_only=True)['Survived'].sort_values(ascending=False)",
], "korelasi_target = train.corr(numeric_only=True)['Survived'].sort_values(ascending=False)")
def _(korelasi_target):
    exp = _data("titanic").corr(numeric_only=True)["Survived"].sort_values(ascending=False)
    _pastikan(isinstance(korelasi_target, pd.Series), "Hasilnya harus Series.")
    _pastikan(_sama_series(korelasi_target, exp), "Angkanya belum cocok.")
    _pastikan(_sama_series(korelasi_target, exp, urut=True), "Angkanya benar, urutannya harus dari terbesar (ascending=False).")


@_soal("2.15", "kolom paling banyak kosong", [
    "Hitung persen kosong tiap kolom (isnull().mean()), lalu cari NAMA kolom dengan nilai terbesar.",
    ".idxmax() memberi index (nama kolom) dari nilai terbesar.",
], "kolom_paling_kosong = train.isnull().mean().idxmax()")
def _(kolom_paling_kosong):
    _pastikan(isinstance(kolom_paling_kosong, str), "Jawabannya NAMA kolom (teks). Pakai .idxmax(), bukan .max().")
    _pastikan(kolom_paling_kosong == "Cabin", f"Dapat '{kolom_paling_kosong}'. Coba cek lagi persen missing-nya.")


@_soal("2.16", "peluang selamat per Sex dan Pclass", [
    "groupby bisa dua kolom sekaligus: groupby(['Sex', 'Pclass']).",
    "train.groupby(['Sex', 'Pclass'])['Survived'].mean()",
], "selamat_sex_kelas = train.groupby(['Sex', 'Pclass'])['Survived'].mean()")
def _(selamat_sex_kelas):
    exp = _data("titanic").groupby(["Sex", "Pclass"])["Survived"].mean()
    _pastikan(isinstance(selamat_sex_kelas, pd.Series), "Hasilnya harus Series (pakai ['Survived'] sebelum .mean()).")
    _pastikan(len(selamat_sex_kelas) == 6, f"Harusnya 6 kombinasi (2 jenis kelamin x 3 kelas), dapat {len(selamat_sex_kelas)}.")
    _pastikan(_sama_series(selamat_sex_kelas, exp), "Angkanya belum cocok. Urutan groupby: ['Sex', 'Pclass'].")


# ---------------------------------------------------------------- Notebook 02b (Penguins, uji kertas kosong)

@_soal("P.0", "load data penguins", [
    "pd.read_csv('path/file.csv'). File-nya ada di data/penguins/penguins.csv",
    "df = pd.read_csv('data/penguins/penguins.csv')",
], "df = pd.read_csv('data/penguins/penguins.csv')")
def _(df):
    _pastikan(isinstance(df, pd.DataFrame), "df harus DataFrame hasil pd.read_csv(...).")
    _pastikan(df.shape == (344, 7), f"Ukurannya {df.shape}, harusnya (344, 7). Cek path file-nya.")


@_soal("P.1", "ukuran data penguins", ["shape (tanpa kurung)", "df.shape"], "ukuran = df.shape")
def _(ukuran):
    _pastikan(ukuran == (344, 7), f"Dapat {ukuran}, harusnya (344, 7).")


@_soal("P.2", "missing per kolom penguins", ["isnull lalu sum", "df.isnull().sum()"], "kosong = df.isnull().sum()")
def _(kosong):
    _pastikan(_sama_series(kosong, _data("penguins").isnull().sum()), "Belum cocok: df.isnull().sum()")


@_soal("P.3", "proporsi spesies", ["value_counts dengan normalize", "df['species'].value_counts(normalize=True)"],
       "proporsi_spesies = df['species'].value_counts(normalize=True)")
def _(proporsi_spesies):
    exp = _data("penguins")["species"].value_counts(normalize=True)
    _pastikan(_sama_series(proporsi_spesies, exp), "Belum cocok. Kolom 'species', normalize=True.")


@_soal("P.4", "rata-rata massa per spesies", ["groupby species, kolom body_mass_g, mean",
                                              "df.groupby('species')['body_mass_g'].mean()"],
       "massa_per_spesies = df.groupby('species')['body_mass_g'].mean()")
def _(massa_per_spesies):
    exp = _data("penguins").groupby("species")["body_mass_g"].mean()
    _pastikan(_sama_series(massa_per_spesies, exp), "Belum cocok.")


@_soal("P.5", "kolom numerik penguins", ["select_dtypes(include='number')", ".columns.tolist()"],
       "kolom_angka = df.select_dtypes(include='number').columns.tolist()")
def _(kolom_angka):
    exp = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _pastikan(list(kolom_angka) == exp, f"Dapat {list(kolom_angka)}. Harusnya {exp}.")


@_soal("P.6", "korelasi kolom angka penguins", ["corr dengan numeric_only=True", "df.corr(numeric_only=True)"],
       "kor = df.corr(numeric_only=True)")
def _(kor):
    exp = _data("penguins").corr(numeric_only=True)
    _pastikan(_sama_df(kor, exp), "Belum cocok. Harusnya tabel 4x4: df.corr(numeric_only=True).")


# ---------------------------------------------------------------- Notebook 03 (Preprocessing)

def _tt():
    t = _data("titanic")
    s = _data("titanic_test")
    return t, s


@_soal("3.1", "kolom IsAlone di X dan X_test", [
    "Sendirian = ukuran keluarga 1 orang.",
    "(X['FamilySize'] == 1).astype(int)",
], """X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
X_test['IsAlone'] = (X_test['FamilySize'] == 1).astype(int)""")
def _(X, X_test):
    t, s = _tt()
    for nama, df, asli in (("X", X, t), ("X_test", X_test, s)):
        _pastikan("IsAlone" in df.columns, f"{nama} belum punya kolom IsAlone.")
        exp = ((asli["SibSp"] + asli["Parch"]) == 0).astype(int).to_numpy()
        _pastikan(_sama_nilai(df["IsAlone"].to_numpy(), exp), f"Isi IsAlone di {nama} belum benar.")


@_soal("3.2", "kolom HasCabin", [
    "Kebalikan isnull() adalah notnull(): True kalau ADA isinya.",
    "X['Cabin'].notnull().astype(int)",
], """X['HasCabin'] = X['Cabin'].notnull().astype(int)
X_test['HasCabin'] = X_test['Cabin'].notnull().astype(int)""")
def _(X, X_test):
    t, s = _tt()
    for nama, df, asli in (("X", X, t), ("X_test", X_test, s)):
        _pastikan("HasCabin" in df.columns, f"{nama} belum punya kolom HasCabin.")
        exp = asli["Cabin"].notnull().astype(int).to_numpy()
        _pastikan(_sama_nilai(df["HasCabin"].to_numpy(), exp), f"Isi HasCabin di {nama} belum benar (1 = ada kabin).")


@_soal("3.3", "buang kolom", [
    "Kamu sudah punya list 'buang'. Masukkan list itu ke drop(columns=...).",
    "X.drop(columns=buang)",
], """buang = ['PassengerId', 'Name', 'Ticket', 'Cabin']
X = X.drop(columns=buang)
X_test = X_test.drop(columns=buang)""")
def _(X, X_test):
    for nama, df in (("X", X), ("X_test", X_test)):
        sisa = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
        _pastikan(not sisa, f"{nama} masih punya kolom {sisa}. Jangan lupa hasil drop disimpan: X = X.drop(...)")
    _pastikan("Survived" not in X.columns, "X tidak boleh berisi kolom target Survived.")


@_soal("3.4", "isi Fare kosong dengan median TRAIN", [
    "Hitung median dari X (train): X['Fare'].median(). Lalu fillna(median_fare) untuk X dan X_test.",
    "fillna(median_fare) — pakai nilai yang SAMA untuk train dan test.",
], """median_fare = X['Fare'].median()
X['Fare'] = X['Fare'].fillna(median_fare)
X_test['Fare'] = X_test['Fare'].fillna(median_fare)""")
def _(X, X_test):
    t, s = _tt()
    _pastikan(X_test["Fare"].isnull().sum() == 0, "X_test['Fare'] masih ada yang kosong. Simpan hasilnya: X_test['Fare'] = X_test['Fare'].fillna(...)")
    baris = s.index[s["Fare"].isnull()][0]
    isi = X_test["Fare"].iloc[baris]
    if np.isclose(isi, s["Fare"].median()) and not np.isclose(isi, t["Fare"].median()):
        raise Gagal("Kamu mengisi pakai median TEST. Aturannya: hitung dari TRAIN (X), pakai untuk keduanya.")
    _pastikan(np.isclose(isi, t["Fare"].median()), f"Nilai pengisinya {isi}. Harusnya median Fare dari train ({t['Fare'].median()}).")


@_soal("3.5", "isi Embarked kosong dengan modus", [
    "mode() menghasilkan Series (bisa lebih dari satu modus). Ambil yang pertama: [0].",
    "X['Embarked'].mode()[0]",
], """modus_embarked = X['Embarked'].mode()[0]
X['Embarked'] = X['Embarked'].fillna(modus_embarked)
X_test['Embarked'] = X_test['Embarked'].fillna(modus_embarked)""")
def _(X, X_test):
    t, s = _tt()
    for nama, df in (("X", X), ("X_test", X_test)):
        _pastikan(df["Embarked"].isnull().sum() == 0, f"{nama}['Embarked'] masih ada yang kosong.")
    baris = t.index[t["Embarked"].isnull()]
    _pastikan(all(X["Embarked"].iloc[i] == "S" for i in baris), "Nilai pengisinya harus modus train, yaitu 'S'.")


@_soal("3.6", "ubah Sex jadi angka", [
    "Kamus map: {'male': 0, 'female': 1}.",
    "Kalau hasilnya NaN semua: sel ini kejalan 2x (0/1 tidak ada di kamus). Jalankan ulang dari sel paling atas.",
], """X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X_test['Sex'] = X_test['Sex'].map({'male': 0, 'female': 1})""")
def _(X, X_test):
    for nama, df in (("X", X), ("X_test", X_test)):
        _pastikan(df["Sex"].isnull().sum() == 0,
                  f"{nama}['Sex'] berisi NaN. Biasanya karena sel map dijalankan 2 kali. Jalankan ulang notebook dari atas (Run All).")
        _pastikan(set(pd.unique(df["Sex"])) <= {0, 1}, f"{nama}['Sex'] harus berisi 0 dan 1.")
    t, s = _tt()
    male_x = X["Sex"].to_numpy()[(t["Sex"] == "male").to_numpy()][0]
    male_t = X_test["Sex"].to_numpy()[(s["Sex"] == "male").to_numpy()][0]
    _pastikan(male_x == male_t, "Kamus map untuk X dan X_test harus SAMA.")


@_soal("3.7", "one-hot Embarked", [
    "columns diisi list nama kolom yang mau di-one-hot: ['Embarked'].",
    "pd.get_dummies(X, columns=['Embarked'], dtype=int)",
], """X = pd.get_dummies(X, columns=['Embarked'], dtype=int)
X_test = pd.get_dummies(X_test, columns=['Embarked'], dtype=int)""")
def _(X, X_test):
    for nama, df in (("X", X), ("X_test", X_test)):
        _pastikan("Embarked" not in df.columns, f"{nama} masih punya kolom Embarked asli. Simpan hasil get_dummies ke {nama}.")
        _pastikan(any(str(c).startswith("Embarked_") for c in df.columns), f"{nama} belum punya kolom Embarked_C/Q/S.")


@_soal("3.8", "samakan kolom train & test", [
    "Kolom yang tidak ada di test diisi 0.",
    "fill_value=0",
], "X_test = X_test.reindex(columns=X.columns, fill_value=0)")
def _(X, X_test):
    _pastikan(list(X_test.columns) == list(X.columns), "Urutan/daftar kolom X_test belum sama dengan X.")
    _pastikan(X_test.isnull().sum().sum() == 0 or X.isnull().sum().sum() > 0,
              "Muncul NaN baru di X_test. Pakai fill_value=0.")


@_soal("3.9", "standardisasi (fit di train, transform di test)", [
    "Train: fit_transform (belajar rata-rata & std, lalu ubah). Test: transform saja (pakai hasil belajar dari train).",
    "scaler.fit_transform(...) lalu scaler.transform(...)",
], """from sklearn.preprocessing import StandardScaler
kolom_skala = ['Age', 'Fare', 'FamilySize']
scaler = StandardScaler()
X[kolom_skala] = scaler.fit_transform(X[kolom_skala])
X_test[kolom_skala] = scaler.transform(X_test[kolom_skala])""")
def _(X, X_test):
    kol = ["Age", "Fare", "FamilySize"]
    _pastikan(np.allclose(X[kol].mean().to_numpy(), 0, atol=1e-6), "Rata-rata kolom X setelah scaling harus ~0. Pakai fit_transform di X.")
    _pastikan(not np.allclose(X_test[kol].mean().to_numpy(), 0, atol=1e-9),
              "X_test ikut di-fit ulang. Untuk test pakai scaler.transform(...), BUKAN fit_transform.")
    _pastikan(X_test[kol].abs().max().max() < 50, "Nilai X_test aneh. Pastikan X_test[kolom_skala] = scaler.transform(X_test[kolom_skala]).")


@_soal("3.10", "pemeriksaan akhir", [
    "Missing total: X.isnull().sum().sum() + X_test.isnull().sum().sum().  Kolom teks: X.select_dtypes(exclude='number').columns.tolist().",
    "Kolom sama: list(X.columns) == list(X_test.columns)",
], """sisa_kosong = X.isnull().sum().sum() + X_test.isnull().sum().sum()
kolom_teks = X.select_dtypes(exclude='number').columns.tolist()
kolom_sama = list(X.columns) == list(X_test.columns)""")
def _(sisa_kosong, kolom_teks, kolom_sama):
    ns = _ns()
    X, Xt = ns.get("X"), ns.get("X_test")
    if isinstance(X, pd.DataFrame) and isinstance(Xt, pd.DataFrame):
        _pastikan(int(sisa_kosong) == int(X.isnull().sum().sum() + Xt.isnull().sum().sum()), "Hitungan sisa_kosong belum benar.")
        _pastikan(list(kolom_teks) == X.select_dtypes(exclude="number").columns.tolist(), "Isi kolom_teks belum benar.")
        _pastikan(bool(kolom_sama) == (list(X.columns) == list(Xt.columns)), "Isi kolom_sama belum benar.")
    _pastikan(int(sisa_kosong) == 0, f"Masih ada {sisa_kosong} nilai kosong. Periksa lagi langkah isi missing.")
    _pastikan(len(kolom_teks) == 0, f"Masih ada kolom teks: {kolom_teks}. Encode dulu.")
    _pastikan(bool(kolom_sama), "Kolom X dan X_test belum sama. Pakai reindex.")


# ---------------------------------------------------------------- Notebook 04 (Model & Submission)

@_soal("4.1", "bagi data train/validasi", [
    "20% untuk validasi -> 0.2. stratify=y supaya proporsi selamat di train & val sama.",
    "test_size=0.2, stratify=y",
], """from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)""")
def _(X_train, X_val, y_train, y_val):
    _pastikan(len(X_train) == len(y_train) and len(X_val) == len(y_val),
              "Pasangannya ketuker. Urutan hasil: X_train, X_val, y_train, y_val.")
    _pastikan(len(X_val) == 143, f"X_val berisi {len(X_val)} baris. Dengan test_size=0.2 harusnya 143.")
    _pastikan(list(X_train.index) == list(y_train.index), "Index X_train dan y_train tidak sejajar.")
    _pastikan(abs(y_val.mean() - y_train.mean()) < 0.01, "Proporsi target train & val beda jauh. Isi stratify=y.")


@_soal("4.2", "akurasi baseline di validasi", [
    "Tebakan mayoritas sudah ada di variabel tebakan_mayoritas. Bandingkan y_val dengan tebakan itu.",
    "(y_val == tebakan_mayoritas).mean()",
], """tebakan_mayoritas = y_train.mode()[0]
baseline_acc = (y_val == tebakan_mayoritas).mean()""")
def _(baseline_acc):
    ns = _ns()
    yv, yt = ns.get("y_val"), ns.get("y_train")
    if isinstance(yv, pd.Series) and isinstance(yt, pd.Series):
        _pastikan(np.isclose(baseline_acc, (yv == yt.mode()[0]).mean()), "Belum cocok: (y_val == tebakan_mayoritas).mean()")
    _pastikan(0.55 < float(baseline_acc) < 0.7, f"Nilai {baseline_acc} kurang masuk akal (harusnya ~0.62).")


@_soal("4.3", "Logistic Regression", [
    "Tiga langkah wajib: model.fit(X_train, y_train) -> model.predict(X_val) -> accuracy_score(y_val, pred).",
    "fit, predict, accuracy_score(y_val, pred_lr)",
], """from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_val)
acc_lr = accuracy_score(y_val, pred_lr)""")
def _(acc_lr):
    ns = _ns()
    m, Xv, yv = ns.get("model_lr"), ns.get("X_val"), ns.get("y_val")
    _pastikan(m is not None, "Variabel model_lr belum ada.")
    try:
        exp = (m.predict(Xv) == np.asarray(yv)).mean()
    except Exception:
        raise Gagal("model_lr belum di-fit. Jalankan model_lr.fit(X_train, y_train).")
    _pastikan(np.isclose(acc_lr, exp), "acc_lr belum cocok. Urutan: accuracy_score(y_val, pred_lr) — jawaban asli dulu, baru tebakan.")


@_soal("4.4", "Decision Tree dari ingatan", [
    "Sama persis polanya dengan Random Forest: buat model -> fit -> predict -> accuracy_score.",
    "model_dt = DecisionTreeClassifier(random_state=42)",
], """from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_val)
acc_dt = accuracy_score(y_val, pred_dt)""")
def _(acc_dt):
    ns = _ns()
    m, Xv, yv = ns.get("model_dt"), ns.get("X_val"), ns.get("y_val")
    _pastikan(m is not None, "Simpan modelnya di variabel model_dt.")
    try:
        exp = (m.predict(Xv) == np.asarray(yv)).mean()
    except Exception:
        raise Gagal("model_dt belum di-fit.")
    _pastikan(np.isclose(acc_dt, exp), "acc_dt belum cocok dengan model_dt.")


@_soal("4.5", "F1-score", [
    "Formatnya sama dengan accuracy_score: (jawaban asli, tebakan).",
    "f1_score(y_val, pred_val)",
], """from sklearn.metrics import f1_score
f1_rf = f1_score(y_val, pred_val)""")
def _(f1_rf):
    from sklearn.metrics import f1_score
    ns = _ns()
    yv, pv = ns.get("y_val"), ns.get("pred_val")
    _pastikan(yv is not None and pv is not None, "Jalankan dulu sel contoh Random Forest (yang membuat pred_val).")
    _pastikan(np.isclose(f1_rf, f1_score(yv, pv)), "Belum cocok: f1_score(y_val, pred_val).")


@_soal("4.6", "latih ulang di SEMUA data & prediksi test", [
    "Setelah validasi, model final dilatih pakai X dan y (semua data train), lalu predict X_test.",
    "model_final.fit(X, y)  lalu  model_final.predict(X_test)",
], """model_final = RandomForestClassifier(n_estimators=200, random_state=42)
model_final.fit(X, y)
pred_test = model_final.predict(X_test)""")
def _(pred_test):
    _pastikan(len(pred_test) == 179, f"pred_test berisi {len(pred_test)} nilai. Harusnya 179 (jumlah baris test). Predict X_test, bukan X_val.")
    _pastikan(set(np.unique(pred_test)) <= {0, 1}, "Isi pred_test harus 0/1.")
    ns = _ns()
    m, X = ns.get("model_final"), ns.get("X")
    try:
        n_latih = m.estimators_[0].tree_.weighted_n_node_samples[0]
    except Exception:
        n_latih = None
    if n_latih is not None and isinstance(X, pd.DataFrame):
        _pastikan(round(n_latih) == len(X),
                  f"model_final dilatih dengan {round(n_latih)} baris. Harusnya pakai SEMUA data train: model_final.fit(X, y).")


@_soal("4.7", "file submission", [
    "Kolom: PassengerId dari test_ids, Survived dari pred_test. Saat simpan: index=False.",
    "pd.DataFrame({'PassengerId': test_ids, 'Survived': pred_test}) ... to_csv('submission.csv', index=False)",
], """submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': pred_test})
submission.to_csv('submission.csv', index=False)""")
def _(submission):
    s = _data("titanic_test")
    _pastikan(list(submission.columns) == ["PassengerId", "Survived"], f"Kolomnya {list(submission.columns)}. Harusnya ['PassengerId', 'Survived'].")
    _pastikan(len(submission) == 179, f"Barisnya {len(submission)}, harusnya 179.")
    _pastikan(list(submission["PassengerId"]) == list(s["PassengerId"]), "PassengerId harus dari data TEST (test_ids).")
    _pastikan(os.path.exists("submission.csv"), "File submission.csv belum tersimpan. Jalankan to_csv(...).")
    kol = list(pd.read_csv("submission.csv").columns)
    _pastikan("Unnamed: 0" not in kol, "Di file ada kolom 'Unnamed: 0' -> kamu lupa index=False.")


@_soal("4.8", "pipeline anti-ribet", [
    "Angka -> 'median'. Teks -> 'most_frequent'. OneHotEncoder -> handle_unknown='ignore'.",
    "strategy='median', strategy='most_frequent', handle_unknown='ignore'",
], """preprocess = ColumnTransformer([
    ('num', Pipeline([('isi_kosong', SimpleImputer(strategy='median')),
                      ('skala', StandardScaler())]), num_cols),
    ('kat', Pipeline([('isi_kosong', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])
pipe = Pipeline([('prep', preprocess),
                 ('model', RandomForestClassifier(n_estimators=200, random_state=42))])
pipe.fit(X_raw, y_raw)""")
def _(pipe):
    try:
        prep = pipe.named_steps["prep"]
        t = {nama: obj for nama, obj, _ in prep.transformers}
        s_num = t["num"].named_steps["isi_kosong"].strategy
        s_kat = t["kat"].named_steps["isi_kosong"].strategy
        hu = t["kat"].named_steps["onehot"].handle_unknown
    except Exception:
        raise Gagal("Struktur pipeline berubah. Jangan ganti nama langkah ('prep', 'num', 'kat', 'isi_kosong', 'onehot').")
    _pastikan(s_num in ("median", "mean"), f"Untuk angka pakai 'median' (atau 'mean'), bukan '{s_num}'.")
    _pastikan(s_kat in ("most_frequent", "constant"), f"Untuk teks pakai 'most_frequent', bukan '{s_kat}'.")
    _pastikan(hu in ("ignore", "infrequent_if_exist"), "handle_unknown harus 'ignore' supaya kategori baru di test tidak bikin error.")
    _pastikan(hasattr(pipe, "classes_"), "Pipeline belum di-fit. Jalankan pipe.fit(X_raw, y_raw).")


@_soal("4.9", "RMSE dan MAE", [
    "Formatnya (jawaban asli, tebakan) = (yf_val, pred_fare). RMSE = akar dari MSE -> np.sqrt(...).",
    "np.sqrt(mean_squared_error(yf_val, pred_fare))  dan  mean_absolute_error(yf_val, pred_fare)",
], """rmse = np.sqrt(mean_squared_error(yf_val, pred_fare))
mae = mean_absolute_error(yf_val, pred_fare)""")
def _(rmse, mae):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    ns = _ns()
    yv, pv = ns.get("yf_val"), ns.get("pred_fare")
    _pastikan(yv is not None and pv is not None, "Jalankan dulu sel contoh regresi (yang membuat yf_val dan pred_fare).")
    mse = mean_squared_error(yv, pv)
    _pastikan(not np.isclose(rmse, mse), "Itu masih MSE. RMSE = np.sqrt(MSE).")
    _pastikan(np.isclose(rmse, np.sqrt(mse)), "RMSE belum cocok.")
    _pastikan(np.isclose(mae, mean_absolute_error(yv, pv)), "MAE belum cocok.")


@_soal("4.10", "cross-validation 5 lipatan", [
    "5 lipatan -> cv=5. Rata-rata array -> .mean().",
    "cv=5  lalu  skor_cv.mean()",
], """from sklearn.model_selection import cross_val_score
skor_cv = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5, scoring='accuracy')
rata_cv = skor_cv.mean()""")
def _(rata_cv):
    ns = _ns()
    s = ns.get("skor_cv")
    _pastikan(s is not None and len(s) == 5, "skor_cv harus berisi 5 nilai (cv=5).")
    _pastikan(np.isclose(rata_cv, np.mean(s)), "rata_cv = skor_cv.mean()")


# ============================================================ perintah untuk siswa

def cek(no, *jawaban_siswa):
    """Periksa jawaban latihan. Contoh: cek('1.2', ukuran)"""
    no = str(no)
    if no not in SOAL:
        print(f"Nomor latihan '{no}' tidak dikenal.")
        return
    s = SOAL[no]
    if any(j is Ellipsis for j in jawaban_siswa):
        print(f"[{no}] Kamu belum menulis jawaban (masih ada ...). Ganti ... dengan kodemu, lalu jalankan lagi.")
        return
    try:
        s["cek"](*jawaban_siswa)
    except Gagal as g:
        print(f"❌ [{no}] Belum tepat: {g}\n   Butuh petunjuk? Jalankan: hint('{no}')")
        return
    except Exception as e:  # jawaban bertipe aneh, dsb.
        print(f"❌ [{no}] Belum tepat. Jawabanmu belum berbentuk yang diminta ({type(e).__name__}: {e}).\n"
              f"   Butuh petunjuk? Jalankan: hint('{no}')")
        return
    _catat_lulus(no)
    print(f"✅ [{no}] Benar! {random.choice(_PUJIAN)}")


_HINT_KE = {}


def hint(no):
    """Petunjuk bertahap. Panggil berulang untuk petunjuk berikutnya."""
    no = str(no)
    if no not in SOAL:
        print(f"Nomor latihan '{no}' tidak dikenal.")
        return
    hs = SOAL[no]["hints"]
    i = _HINT_KE.get(no, 0)
    if i < len(hs):
        print(f"Petunjuk {i + 1}/{len(hs)} untuk [{no}]: {hs[i]}")
        _HINT_KE[no] = i + 1
    else:
        print(f"Semua petunjuk [{no}] sudah keluar. Masih mentok? Jalankan: jawaban('{no}')")


def jawaban(no):
    """Kunci jawaban. Setelah lihat, KETIK ULANG sendiri, jangan copy-paste."""
    no = str(no)
    if no not in SOAL:
        print(f"Nomor latihan '{no}' tidak dikenal.")
        return
    print(f"Kunci [{no}] ({SOAL[no]['judul']}):\n")
    print(SOAL[no]["jawaban"])
    print("\nSekarang TUTUP jawaban ini dan ketik ulang sendiri di sel latihan, jangan copy-paste."
          "\nMengetik sendiri itu yang bikin kodenya nempel di kepala.")


_JUDUL_NB = {"1": "01 Pandas Dasar", "2": "02 EDA", "P": "02b Uji Kertas Kosong", "3": "03 Preprocessing",
             "4": "04 Model & Submission"}


def progres():
    """Tampilkan latihan yang sudah lulus."""
    lulus = _baca_progres().get("lulus", {})
    total = 0
    for kode, judul in _JUDUL_NB.items():
        nomor = [n for n in SOAL if n.split(".")[0] == kode]
        ok = [n for n in nomor if n in lulus]
        total += len(ok)
        baris = " ".join(("■" if n in lulus else "□") for n in nomor)
        print(f"{judul:24s} {len(ok):2d}/{len(nomor):<2d}  {baris}")
    print(f"\nTotal lulus: {total}/{len(SOAL)}")
    skor_ = _baca_progres().get("skor", {})
    for nama, riwayat in skor_.items():
        k = _KOMPETISI[nama]
        terbaik = (max if k["metrik"] == "accuracy" else min)(r["skor"] for r in riwayat)
        print(f"Skor terbaik {nama}: {terbaik:.4f} ({len(riwayat)}x submit)")


# ============================================================ leaderboard lokal

_KOMPETISI = {
    "titanic": dict(folder="titanic", id="PassengerId", target="Survived", metrik="accuracy",
                    file="submission.csv",
                    baseline=[("Tebak kelas terbanyak", 110 / 179), ("Template dasar", 146 / 179)]),
    "mock1": dict(folder="mock1_churn", id="id_pelanggan", target="churn", metrik="accuracy",
                  file="submission_mock1.csv",
                  baseline=[("Tebak kelas terbanyak", 0.7000), ("Template dasar", 0.8240),
                            ("Template + bersih-bersih + LogisticRegression", 0.8420)]),
    "mock2": dict(folder="mock2_rumah", id="id", target="harga_juta", metrik="rmse",
                  file="submission_mock2.csv",
                  baseline=[("Tebak rata-rata", 1294.215), ("Template dasar", 375.7504),
                            ("Template + bersih-bersih + log target", 370.18)]),
}
_ALIAS = {"mock1_churn": "mock1", "churn": "mock1", "mock2_rumah": "mock2", "rumah": "mock2"}


def skor(nama, path=None):
    """Nilai file submission seperti leaderboard Kaggle. Contoh: skor('mock1', 'submission_mock1.csv')"""
    nama = _ALIAS.get(nama, nama)
    if nama not in _KOMPETISI:
        print(f"Nama kompetisi harus salah satu dari: {list(_KOMPETISI)}")
        return
    k = _KOMPETISI[nama]
    path = path or k["file"]
    if not os.path.exists(path):
        ada = [f for f in os.listdir(".") if f.endswith(".csv")]
        print(f"File '{path}' tidak ditemukan di folder ini.\nFile CSV yang ada: {ada}\n"
              f"Simpan dulu: submission.to_csv('{path}', index=False)")
        return
    sub = pd.read_csv(path)
    kunci = _csv("_kunci", f"{k['folder']}.csv")
    masalah = []
    if "Unnamed: 0" in sub.columns:
        masalah.append("Ada kolom 'Unnamed: 0' → saat to_csv lupa index=False.")
    for kol in (k["id"], k["target"]):
        if kol not in sub.columns:
            masalah.append(f"Kolom '{kol}' tidak ada. Kolom yang wajib: ['{k['id']}', '{k['target']}'] (lihat sample_submission.csv).")
    if len(sub) != len(kunci):
        masalah.append(f"Jumlah baris {len(sub)}, harusnya {len(kunci)} (sama dengan test.csv).")
    if not masalah:
        if sub[k["id"]].duplicated().any():
            masalah.append("Ada ID yang dobel.")
        if set(sub[k["id"]]) != set(kunci[k["id"]]):
            masalah.append("ID tidak cocok dengan test.csv. Ambil ID dari data TEST.")
        if sub[k["target"]].isnull().any():
            masalah.append(f"Ada {sub[k['target']].isnull().sum()} prediksi kosong (NaN).")
    if not masalah and k["metrik"] == "accuracy":
        nilai = set(pd.unique(sub[k["target"]]))
        if not nilai <= {0, 1}:
            contoh = ", ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                               for v in sorted(nilai - {0, 1}, key=str)[:4])
            masalah.append(f"Isi kolom {k['target']} harus 0/1, tapi ada nilai seperti: {contoh}."
                           " Pakai model.predict(...), bukan predict_proba, dan model Classifier.")
    if masalah:
        print("❌ Submission DITOLAK (di Kaggle juga bakal ditolak):")
        for m in masalah:
            print("   •", m)
        return
    gab = kunci.merge(sub, on=k["id"], suffixes=("_asli", "_tebak"))
    asli, tebak = gab[k["target"] + "_asli"].to_numpy(), gab[k["target"] + "_tebak"].to_numpy()
    if k["metrik"] == "accuracy":
        nilai = float((asli == tebak).mean())
        arah, fmt = "makin BESAR makin bagus", "{:.4f}"
    else:
        nilai = float(np.sqrt(np.mean((asli - tebak) ** 2)))
        arah, fmt = "makin KECIL makin bagus", "{:.1f}"
    p = _baca_progres()
    riwayat = p.setdefault("skor", {}).setdefault(nama, [])
    riwayat.append({"skor": nilai, "waktu": datetime.datetime.now().isoformat(timespec="seconds")})
    _tulis_progres(p)
    lebih_baik = (lambda a, b: a > b) if k["metrik"] == "accuracy" else (lambda a, b: a < b)
    papan = [(n, v) for n, v in k["baseline"]] + [("KAMU (submit ini)", nilai)]
    papan.sort(key=lambda x: -x[1] if k["metrik"] == "accuracy" else x[1])
    print(f"LEADERBOARD {nama}  ·  metrik: {k['metrik'].upper()} ({arah})\n")
    for i, (n, v) in enumerate(papan, 1):
        tanda = "  ◀" if n.startswith("KAMU") else ""
        print(f"  {i}. {fmt.format(v):>9s}  {n}{tanda}")
    def sama(a, b):
        return abs(a - b) <= 1e-4 * max(1.0, abs(b))

    seri = [n for n, v in k["baseline"] if sama(nilai, v)]
    kalah = [(n, v) for n, v in k["baseline"] if lebih_baik(v, nilai) and not sama(nilai, v)]
    print()
    if seri:
        print(f"Skormu SAMA dengan '{seri[-1]}'.", end=" ")
    if kalah:
        n, v = kalah[0]
        print(f"Target berikutnya: kalahkan '{n}' ({fmt.format(v)}).")
    elif seri:
        print("Itu baseline terbaik. Coba satu perbaikan dari bagian UPGRADE untuk melewatinya.")
    else:
        print("Kamu mengalahkan semua baseline. Siap tanding!")


# ============================================================ penjelas error

_IMPOR = {
    "pd": "import pandas as pd", "np": "import numpy as np", "plt": "import matplotlib.pyplot as plt",
    "sns": "import seaborn as sns", "os": "import os", "warnings": "import warnings",
    "train_test_split": "from sklearn.model_selection import train_test_split",
    "cross_val_score": "from sklearn.model_selection import cross_val_score",
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
    "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
    "LinearRegression": "from sklearn.linear_model import LinearRegression",
    "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
    "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
    "accuracy_score": "from sklearn.metrics import accuracy_score",
    "f1_score": "from sklearn.metrics import f1_score",
    "confusion_matrix": "from sklearn.metrics import confusion_matrix",
    "classification_report": "from sklearn.metrics import classification_report",
    "mean_squared_error": "from sklearn.metrics import mean_squared_error",
    "mean_absolute_error": "from sklearn.metrics import mean_absolute_error",
    "r2_score": "from sklearn.metrics import r2_score",
    "StandardScaler": "from sklearn.preprocessing import StandardScaler",
    "MinMaxScaler": "from sklearn.preprocessing import MinMaxScaler",
    "OneHotEncoder": "from sklearn.preprocessing import OneHotEncoder",
    "LabelEncoder": "from sklearn.preprocessing import LabelEncoder",
    "SimpleImputer": "from sklearn.impute import SimpleImputer",
    "ColumnTransformer": "from sklearn.compose import ColumnTransformer",
    "Pipeline": "from sklearn.pipeline import Pipeline",
    "cek": "from cek import cek, hint, jawaban, progres, skor",
    "hint": "from cek import cek, hint, jawaban, progres, skor",
    "jawaban": "from cek import cek, hint, jawaban, progres, skor",
    "skor": "from cek import cek, hint, jawaban, progres, skor",
}


def jelaskan(etype, evalue):
    """Ubah error Python jadi penjelasan bahasa Indonesia (atau None kalau tidak dikenal)."""
    nama = etype.__name__
    pesan = str(evalue)

    kosong = "Bagian ____ (garis kosong) belum kamu ganti. Ganti ____ dengan kode yang benar, lalu jalankan lagi."
    if "____" in pesan:
        return kosong

    if nama == "NameError":
        m = re.search(r"name '([^']+)' is not defined", pesan)
        var = m.group(1) if m else "?"
        if set(var) == {"_"}:
            return kosong
        if var in _IMPOR:
            return f"'{var}' belum di-import. Tambahkan baris ini di sel atas lalu jalankan:\n    {_IMPOR[var]}"
        return (f"Variabel '{var}' belum ada. Penyebab paling umum:\n"
                "  1) Sel yang membuat variabel itu belum dijalankan → jalankan sel-sel di atas (atau Run All).\n"
                f"  2) Salah ketik / huruf besar-kecil beda (contoh: Train ≠ train).\n"
                "  3) Kernel baru restart → semua variabel hilang, jalankan ulang dari atas.")

    if nama == "KeyError":
        if "not found in axis" in pesan:
            return ("Kolom yang mau di-drop tidak ada: " + pesan + "\n"
                    "Kemungkinan sudah ke-drop sebelumnya (sel drop kejalan 2x) atau salah ketik.\n"
                    "Cek: df.columns.tolist()   |   Aman: df.drop(columns=[...], errors='ignore')")
        if "None of [" in pesan or "not in index" in pesan:
            return ("Nama kolom di dalam [[...]] tidak ada di data: " + pesan + "\n"
                    "Cek ejaan dengan df.columns.tolist(). Ingat: data TEST memang tidak punya kolom target.")
        key = evalue.args[0] if evalue.args else pesan
        if isinstance(key, tuple):
            return ("Mau ambil beberapa kolom? Pakai DUA kurung siku:\n"
                    "    df[['a', 'b']]   bukan   df['a', 'b']")
        if isinstance(key, (int, np.integer)):
            return (f"Kamu memakai angka {key} sebagai nama kolom. Mau ambil baris ke-{key}? Pakai df.iloc[{key}].")
        return (f"Kolom {key!r} tidak ditemukan. Cek:\n"
                "  • ejaan & huruf besar/kecil (Python membedakan 'age' dan 'Age'), spasi tersembunyi\n"
                "  • kolomnya sudah di-drop atau belum dibuat\n"
                "  • data TEST memang tidak punya kolom target\n"
                "Lihat daftar kolom: df.columns.tolist()")

    if nama == "AttributeError":
        if "'NoneType' object has no attribute" in pesan:
            return ("Variabelmu berisi None. Penyebab paling umum: df = df.fillna(..., inplace=True)\n"
                    "atau df = df.drop(..., inplace=True). Dengan inplace=True hasilnya None.\n"
                    "Solusi: hapus inplace=True, tulis df = df.fillna(...). Lalu jalankan ulang dari sel load data.")
        m = re.search(r"'(\w+)' object has no attribute '(\w+)'", pesan)
        if m:
            obj, attr = m.groups()
            if obj == "ndarray":
                return ("Hasil model.predict(...) itu array numpy, bukan tabel pandas.\n"
                        f"Bungkus dulu: pd.Series(pred).{attr}(...)")
            if set(attr) == {"_"}:
                return kosong
            kandidat = {"DataFrame": dir(pd.DataFrame), "Series": dir(pd.Series)}.get(obj, [])
            mirip = difflib.get_close_matches(attr, [k for k in kandidat if not k.startswith("_")], n=3)
            saran = f"\nMungkin maksudmu: {', '.join(mirip)}" if mirip else ""
            return (f".{attr} bukan fungsi/atribut dari {obj}. Kemungkinan salah ketik.{saran}\n"
                    f"Kalau '{attr}' itu nama kolom, ambil dengan df['{attr}'].")

    if nama == "TypeError":
        if "'tuple' object is not callable" in pesan:
            return "Kamu memanggil sesuatu yang bukan fungsi. Paling sering: df.shape() → harusnya df.shape (tanpa kurung)."
        if "'Index' object is not callable" in pesan:
            return "df.columns() → harusnya df.columns (tanpa kurung)."
        if re.search(r"'(Series|DataFrame)' object is not callable", pesan):
            return ("Kamu pakai kurung biasa ( ) di tempat yang harusnya kurung siku [ ].\n"
                    "Contoh: df('Age') → df['Age'].  Atau nama variabelmu menimpa nama fungsi.")
        if re.search(r"'(int|float|str)' object is not callable", pesan):
            return ("Ada nama variabel yang menimpa nama fungsi (misal kamu pernah menulis sum = 5,\n"
                    "lalu memanggil sum(...)). Ganti nama variabelnya lalu restart kernel.")
        if "not subscriptable" in pesan and "NoneType" in pesan:
            return "Variabelmu berisi None. Biasanya karena inplace=True. Hapus inplace=True dan tulis df = df.fungsi(...)."
        if "unsupported operand type(s) for &" in pesan or "unsupported operand type(s) for |" in pesan \
                or "Cannot perform 'rand_'" in pesan or "Cannot perform 'ror_'" in pesan:
            return ("Kondisi gabungan wajib dikurung satu per satu:\n"
                    "    df[(df['a'] == 1) & (df['b'] > 2)]")
        if "agg function failed" in pesan or "does not support operation" in pesan or "Could not convert" in pesan:
            return ("Kamu menghitung mean/median/sum pada kolom TEKS. Pilih kolom angka dulu:\n"
                    "    df.groupby('a')[['b', 'c']].mean()    atau    df.groupby('a').mean(numeric_only=True)")
        if "not supported between instances of 'str' and" in pesan or "not supported between instances of 'float' and 'str'" in pesan \
                or "uniformly strings or numbers" in pesan:
            return ("Kolom berisi campuran teks dan NaN/angka. Isi NaN dulu (fillna) atau samakan tipenya:\n"
                    "    df['kol'] = df['kol'].astype(str)")
        if "missing 1 required positional argument: 'y'" in pesan:
            return "fit butuh dua hal: model.fit(X, y)."
        if re.search(r"Invalid value .* for dtype '?(str|string)", pesan):
            return ("Kamu mengisi kolom TEKS dengan angka (misal df.fillna(0) ke seluruh tabel). Di pandas 3 itu dilarang.\n"
                    "Isi kolom angka dan teks secara terpisah:\n"
                    "    df[num_cols] = df[num_cols].fillna(df[num_cols].median())\n"
                    "    df[cat_cols] = df[cat_cols].fillna('Tidak Ada')")

    if nama == "ValueError":
        if "truth value of a Series is ambiguous" in pesan or "truth value of a DataFrame is ambiguous" in pesan:
            return ("Untuk kondisi pada kolom pakai & dan | (bukan and / or), dan kurung setiap kondisi:\n"
                    "    df[(df['a'] > 1) & (df['b'] == 'x')]")
        m = re.search(r"could not convert string to float: '?([^']*)'?", pesan)
        if m:
            return (f"Masih ada kolom TEKS (contoh nilainya: '{m.group(1)}') yang masuk ke model.\n"
                    "Model cuma bisa baca angka → encode dulu (map / get_dummies / OneHotEncoder) atau buang kolomnya.\n"
                    "Cek: X.select_dtypes(exclude='number').columns.tolist()")
        if "contains NaN" in pesan or "Input contains NaN" in pesan:
            return ("Masih ada missing value (NaN) yang masuk ke model. Cek X.isnull().sum() dan isi dulu\n"
                    "(fillna / SimpleImputer). Jangan lupa cek X_test juga!")
        if "inconsistent numbers of samples" in pesan:
            return ("Dua data yang dipasangkan jumlahnya beda (X vs y saat fit, atau jawaban asli vs prediksi saat menghitung skor).\n"
                    "Cek len() keduanya. Penyebab umum:\n"
                    "  • pasangan hasil train_test_split ketuker (urutannya: X_train, X_val, y_train, y_val)\n"
                    "  • membandingkan y_val dengan prediksi dari X_test/X_train (harusnya prediksi dari X_val)\n"
                    "  • membuang baris di X tapi tidak di y")
        if "feature names should match" in pesan or "Feature names unseen" in pesan or "features, but" in pesan:
            return ("Kolom data yang diprediksi beda dengan kolom waktu fit. Samakan:\n"
                    "    X_test = X_test.reindex(columns=X.columns, fill_value=0)\n"
                    "atau pastikan semua langkah preprocessing dilakukan juga ke X_test.")
        if "does not match length of index" in pesan or "All arrays must be of the same length" in pesan:
            return ("Panjang data beda dengan jumlah baris tabel. Sering terjadi saat bikin submission:\n"
                    "pakai ID dari TEST dan prediksi dari X_test (bukan X_val / X).")
        if "Unknown label type" in pesan and "continuous" in pesan:
            return ("Model KLASIFIKASI dipakai untuk target angka kontinu. Kalau targetnya angka (harga, nilai),\n"
                    "pakai model regresi: RandomForestRegressor / LinearRegression.")
        if "Classification metrics can't handle" in pesan:
            return ("Metrik klasifikasi (accuracy/f1) dipakai untuk prediksi desimal.\n"
                    "Regresi → pakai RMSE/MAE. Klasifikasi → pakai model Classifier dan .predict() (bukan predict_proba).")
        if "Found unknown categories" in pesan:
            return "Test punya kategori yang tidak ada di train. Pakai OneHotEncoder(handle_unknown='ignore')."
        if "Expected 2D array" in pesan:
            return "Model butuh X berbentuk tabel (2D). Pakai df[['kolom']] (dua kurung), bukan df['kolom']."
        if "least populated class" in pesan:
            return "stratify butuh minimal 2 data per kelas. Hapus stratify=y, atau cek targetmu (mungkin ini regresi?)."
        if "previously unseen labels" in pesan or "contains previously unseen" in pesan:
            return "LabelEncoder ketemu label baru di test. Pakai OneHotEncoder(handle_unknown='ignore') atau map dengan kamus."
        if "invalid literal for int()" in pesan or "Unable to parse string" in pesan:
            return "Ada teks yang bukan angka. Ubah dengan aman: pd.to_numeric(df['kol'], errors='coerce')."
        if "time data" in pesan and "does not match format" in pesan:
            return "Format tanggal beda. Pakai: pd.to_datetime(df['kol'], errors='coerce')."
        if "duplicate labels" in pesan:
            return "Ada nama kolom/index kembar. Cek: df.columns[df.columns.duplicated()]"

    if nama == "NotFittedError":
        return "Model belum dilatih. Panggil model.fit(X, y) dulu sebelum predict."
    if nama == "FileNotFoundError":
        return ("File tidak ditemukan. Cek:\n"
                "  1) ejaan nama file & folder (huruf besar/kecil juga)\n"
                "  2) posisi notebook: import os; print(os.getcwd()); print(os.listdir())\n"
                "  3) di Kaggle, data ada di /kaggle/input/... → print(os.listdir('/kaggle/input'))")
    if nama == "ModuleNotFoundError":
        m = re.search(r"No module named '([^']+)'", pesan)
        mod = m.group(1) if m else "?"
        if mod == "cek":
            return "File cek.py tidak ketemu. Pastikan notebook ini ada di folder Dojo_Anti_Ngangong (satu folder dengan cek.py)."
        return f"Library '{mod}' belum terpasang. Jalankan di sel baru: %pip install {mod}   lalu restart kernel."
    if nama == "ImportError" and "cannot import name" in pesan:
        return ("Nama yang di-import salah atau salah tempat. Cek ejaannya.\n"
                "Contoh benar: from sklearn.model_selection import train_test_split")
    if nama == "IndexError":
        return "Kamu minta posisi yang melebihi jumlah data. Ingat: Python mulai menghitung dari 0."
    if nama == "MemoryError":
        return "Memori habis. Biasanya karena one-hot kolom dengan ribuan nilai unik (ID/teks/tanggal). Buang kolom itu."
    return None


def _pasang_penjelas():
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        return
    if ip is None or getattr(ip, "_dojo_penjelas", False):
        return

    def handler(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        try:
            tip = jelaskan(etype, evalue)
        except Exception:
            tip = None
        if tip:
            print("\n" + "─" * 70 + "\n💡 PENJELASAN DOJO (baca baris terakhir error di atas, lalu ini):\n" + tip)
        return None

    ip.set_custom_exc((Exception,), handler)
    try:
        ip._dojo_penjelas = True
    except Exception:
        pass


_pasang_penjelas()
