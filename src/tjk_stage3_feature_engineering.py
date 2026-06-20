#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK AŞAMA 3 – Veri Birleştirme ve Feature Engineering
  Kurşun Geçirmez, Sıfır Veri Sızıntısı (Zero Data Leakage) Pipeline
================================================================================
  Girdiler:
      1) yaris_ana_tablo.csv   – Yarış sonuç verileri
      2) atlar_statik_tablo.csv – Atların statik profil bilgileri
      3) idmanlar_tablo.csv     – Atların geçmiş idman/galop dereceleri
  Çıktı:
      master_feature_matrix.csv – XGBoost / LightGBM'e doğrudan beslenmeye hazır
                                  özellik matrisi (feature matrix)
================================================================================
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()  # .progress_apply() desteği

# ─────────────────────── AYARLAR ───────────────────────
# Veri dosyaları proje kökündeki data/ klasöründe (src/ -> .. -> data)
BASE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
YARIS_CSV = os.path.join(BASE_DIR, "yaris_ana_tablo.csv")
STATIK_CSV = os.path.join(BASE_DIR, "atlar_statik_tablo.csv")
IDMAN_CSV  = os.path.join(BASE_DIR, "idmanlar_tablo.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "master_feature_matrix.csv")

EPSILON = 1e-5  # Sıfıra bölme koruması


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def derece_to_seconds(val):
    """
    TJK süre formatını toplam saniyeye çevirir.
    Formatlar:
      "1.27.23"  → 1 dk 27 sn 23 salise → 87.23 sn
      "0.25.50"  → 0 dk 25 sn 50 salise → 25.50 sn
      "2.23.65"  → 2 dk 23 sn 65 salise → 143.65 sn
    Çevrilemeyenler (Koşmaz, Derecesiz, NaN, vb.) → NaN
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    # Metinsel değerler
    if val_str.lower() in ("koşmaz", "derecesiz", "", "-"):
        return np.nan

    # Nokta ile ayrılmış parçaları bul: "1.27.23" → ['1','27','23']
    parts = re.findall(r'\d+', val_str)

    if len(parts) == 3:
        try:
            dakika  = int(parts[0])
            saniye  = int(parts[1])
            salise  = int(parts[2])
            return dakika * 60 + saniye + salise / 100.0
        except (ValueError, IndexError):
            return np.nan
    elif len(parts) == 2:
        # "25.50" gibi sadece saniye.salise formatı
        try:
            saniye  = int(parts[0])
            salise  = int(parts[1])
            return saniye + salise / 100.0
        except (ValueError, IndexError):
            return np.nan
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    else:
        return np.nan


def extract_at_id_from_url(url):
    """At_URL alanından QueryParameter_AtId değerini çıkarır."""
    if pd.isna(url):
        return np.nan
    m = re.search(r'QueryParameter_AtId=(-?\d+)', str(url))
    return int(m.group(1)) if m else np.nan


def parse_yas(yas_str):
    """
    '3y d  e' gibi string yaş bilgisinden sayısal yaşı çıkarır.
    """
    if pd.isna(yas_str):
        return np.nan
    m = re.match(r'(\d+)', str(yas_str).strip())
    return int(m.group(1)) if m else np.nan


def parse_pist_turu(pist_str):
    """
    'Kum: Normal' → 'Kum', 'Çim: Ağır' → 'Çim', 'Bilinmiyor' / NaN → NaN
    """
    if pd.isna(pist_str):
        return np.nan
    s = str(pist_str).strip()
    if s.lower() == "bilinmiyor" or s == "":
        return np.nan
    if ":" in s:
        return s.split(":")[0].strip()
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 1: VERİ OKUMA VE TEMİZLİK
# ══════════════════════════════════════════════════════════════════════════════

def step1_data_cleaning():
    """Tüm CSV'leri okur, veri tiplerini standardize eder."""
    print("=" * 70)
    print("  ADIM 1: Veri Okuma ve Tip Temizliği")
    print("=" * 70)

    # ── 1a) Yarış Ana Tablosu ──
    print("\n[1/3] yaris_ana_tablo.csv okunuyor...")
    df_yaris = pd.read_csv(YARIS_CSV, encoding="utf-8-sig")
    print(f"      → {len(df_yaris):,} satır yüklendi.")

    # Tarih
    df_yaris["Tarih"] = pd.to_datetime(df_yaris["Tarih"], format="%d.%m.%Y", errors="coerce")
    # NaT tarihli satırları düşür (parse edilemeyen tarihler)
    before_nat = len(df_yaris)
    df_yaris = df_yaris.dropna(subset=["Tarih"]).copy()
    if before_nat - len(df_yaris) > 0:
        print(f"      → {before_nat - len(df_yaris)} satır geçersiz tarih (NaT) nedeniyle çıkarıldı.")

    # at_id: URL'den çıkar
    print("      → At_URL'den at_id çıkarılıyor...")
    df_yaris["at_id"] = df_yaris["At_URL"].apply(extract_at_id_from_url)
    # NaN at_id'leri düşür (yabancı atlar vb.)
    before_drop = len(df_yaris)
    df_yaris = df_yaris.dropna(subset=["at_id"]).copy()
    df_yaris["at_id"] = df_yaris["at_id"].astype(int)
    print(f"      → {before_drop - len(df_yaris)} satır at_id olmadığı için çıkarıldı.")

    # Unique Race ID
    df_yaris["Unique_Race_ID"] = (
        df_yaris["Tarih"].dt.strftime("%Y%m%d") + "_" +
        df_yaris["Sehir"].astype(str).str.strip() + "_" +
        df_yaris["Kosu_ID"].astype(str)
    )

    # Sıralama temizliği: metin değerleri NaN yap, float'a çevir
    df_yaris["Siralama"] = pd.to_numeric(df_yaris["Siralama"], errors="coerce")

    # Derece → saniye
    print("      → Koşu dereceleri saniyeye çevriliyor...")
    df_yaris["Derece_Saniye"] = df_yaris["Derece"].apply(derece_to_seconds)

    # Yaş → sayısal
    df_yaris["Yas_Sayi"] = df_yaris["Yas"].apply(parse_yas)

    # Sıklet → sayısal (zaten çoğunlukla sayısal, ama garantiye alalım)
    df_yaris["Siklet_Sayi"] = pd.to_numeric(df_yaris["Siklet"], errors="coerce")

    # Start → sayısal
    df_yaris["Start_Sayi"] = pd.to_numeric(df_yaris["Start"], errors="coerce")

    # Ganyan → sayısal
    df_yaris["Ganyan_Sayi"] = pd.to_numeric(df_yaris["Ganyan"], errors="coerce")

    # Pist türü
    df_yaris["Pist_Turu"] = df_yaris["Pist_Durumu"].apply(parse_pist_turu)

    # Hedef değişkenler (Target Variables) – Sızıntısız
    df_yaris["Is_Winner"] = (df_yaris["Siralama"] == 1).astype(int)
    df_yaris["Is_Top3"]   = (df_yaris["Siralama"] <= 3).astype(int)

    # Koşmaz / Derecesiz olanları hedef değişkenlerde de 0 yap
    kosmaz_mask = df_yaris["Siralama"].isna()
    df_yaris.loc[kosmaz_mask, "Is_Winner"] = 0
    df_yaris.loc[kosmaz_mask, "Is_Top3"]   = 0

    print(f"      ✓ Unique at_id sayısı: {df_yaris['at_id'].nunique():,}")
    print(f"      ✓ Unique yarış sayısı: {df_yaris['Unique_Race_ID'].nunique():,}")
    print(f"      ✓ Tarih aralığı: {df_yaris['Tarih'].min()} → {df_yaris['Tarih'].max()}")

    # ── 1b) Statik Tablo ──
    print("\n[2/3] atlar_statik_tablo.csv okunuyor...")
    df_statik = pd.read_csv(STATIK_CSV, encoding="utf-8-sig")
    print(f"      → {len(df_statik):,} at profili yüklendi.")

    df_statik["Dogum_Tarihi"] = pd.to_datetime(
        df_statik["Dogum_Tarihi"], format="%d.%m.%Y", errors="coerce"
    )
    df_statik["Handikap_Puani"] = pd.to_numeric(
        df_statik["Handikap_Puani"], errors="coerce"
    )
    df_statik["at_id"] = pd.to_numeric(df_statik["at_id"], errors="coerce")
    df_statik = df_statik.dropna(subset=["at_id"]).copy()
    df_statik["at_id"] = df_statik["at_id"].astype(int)

    # ── 1c) İdman Tablosu ──
    print("\n[3/3] idmanlar_tablo.csv okunuyor...")
    df_idman = pd.read_csv(IDMAN_CSV, encoding="utf-8-sig")
    print(f"      → {len(df_idman):,} idman kaydı yüklendi.")

    df_idman["Idman_Tarihi"] = pd.to_datetime(
        df_idman["Idman_Tarihi"], format="%d.%m.%Y", errors="coerce"
    )
    df_idman["at_id"] = pd.to_numeric(df_idman["at_id"], errors="coerce")
    df_idman = df_idman.dropna(subset=["at_id"]).copy()
    df_idman["at_id"] = df_idman["at_id"].astype(int)

    # İdman derecelerini saniyeye çevir
    idman_derece_cols = [c for c in df_idman.columns if c.startswith("Derece_")]
    print(f"      → İdman derece sütunları ({len(idman_derece_cols)} adet) saniyeye çevriliyor...")
    for col in idman_derece_cols:
        df_idman[col + "_sn"] = df_idman[col].apply(derece_to_seconds)

    print("\n  ADIM 1 TAMAMLANDI ✓")
    return df_yaris, df_statik, df_idman


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 2: ZAMANDA YOLCULUK YAPMAYAN BİRLEŞTİRME (STRICT MERGE)
# ══════════════════════════════════════════════════════════════════════════════

def step2_strict_merge(df_yaris, df_statik, df_idman):
    """
    Statik veriyi standart merge, idman verisini merge_asof ile birleştirir.
    Gelecekteki idmanları ASLA almaz.
    """
    print("\n" + "=" * 70)
    print("  ADIM 2: Zamanda Yolculuk Yapmayan Birleştirme")
    print("=" * 70)

    # ── 2a) Statik Tablo Merge ──
    print("\n[1/2] Statik tablo (Baba, Anne, Handikap) birleştiriliyor...")

    statik_cols = ["at_id", "Dogum_Tarihi", "Handikap_Puani", "Baba", "Anne"]
    statik_available = [c for c in statik_cols if c in df_statik.columns]
    df_statik_slim = df_statik[statik_available].drop_duplicates(subset=["at_id"])

    df = df_yaris.merge(df_statik_slim, on="at_id", how="left")
    matched = df["Baba"].notna().sum()
    print(f"      → {matched:,} / {len(df):,} satır statik veriyle eşleşti.")

    # ── 2b) İdman Tablosu – merge_asof (backward only) ──
    print("\n[2/2] İdman tablosu merge_asof (backward) ile birleştiriliyor...")
    print("      ⚠ Gelecekteki idmanlar kesinlikle alınmıyor!")

    # İdman tablosundan en son dereceleri preprocess et
    idman_sn_cols = [c for c in df_idman.columns if c.endswith("_sn")]
    idman_keep = ["at_id", "Idman_Tarihi", "Idman_Turu"] + idman_sn_cols

    df_idman_slim = df_idman[idman_keep].copy()
    df_idman_slim = df_idman_slim.dropna(subset=["Idman_Tarihi", "at_id"])
    df_idman_slim = df_idman_slim.sort_values(["at_id", "Idman_Tarihi"]).reset_index(drop=True)

    # merge_asof: her yarış satırı için o attan, yarış tarihinden ÖNCE
    # olan en yakın idman satırını getirir
    df_merged = pd.merge_asof(
        df.sort_values("Tarih"),
        df_idman_slim.rename(columns={"Idman_Tarihi": "Son_Idman_Tarihi"}).sort_values("Son_Idman_Tarihi"),
        left_on="Tarih",
        right_on="Son_Idman_Tarihi",
        by="at_id",
        direction="backward",
        suffixes=("", "_idman")
    )

    # İdman – Yarış arası geçen gün sayısı (KRİTİK feature)
    df_merged["Idman_Yaris_Arasi_Gun"] = (
        df_merged["Tarih"] - df_merged["Son_Idman_Tarihi"]
    ).dt.days

    idman_matched = df_merged["Son_Idman_Tarihi"].notna().sum()
    print(f"      → {idman_matched:,} / {len(df_merged):,} satır idman verisiyle eşleşti.")

    print("\n  ADIM 2 TAMAMLANDI ✓")
    return df_merged


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 3: TIME-AWARE TARGET ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def _compute_cumulative_encoding(df, group_col, target_col, col_name):
    """
    Verilen group_col (Jokey_Adi, Antrenor_Adi, Baba) için kronolojik
    expanding().mean().shift(1) hesaplar. Böylece bir satırda sadece
    o satırdan ÖNCEKİ verilerin ortalaması yer alır → Sızıntı = 0.
    """
    # Önce genel ortalamamızı belirleyelim (NaN doldurma için)
    global_mean = df[target_col].mean()

    # Kronolojik sıra zaten step3'te uygulandı (Tarih + Kosu_ID'ye göre)
    # Gruplara expanding cumulative mean uygula, 1 satır kaydır
    df[col_name] = (
        df.groupby(group_col)[target_col]
          .transform(lambda x: x.expanding().mean().shift(1))
    )

    # İlk kez yarışanlar için genel ortalama
    df[col_name] = df[col_name].fillna(global_mean)

    return df


def step3_target_encoding(df):
    """
    Jokey, Antrenör ve Baba için Win Rate ve Top3 Rate hesaplar.
    Tamamen geçmişe dayalı, expanding().mean().shift(1) ile.
    """
    print("\n" + "=" * 70)
    print("  ADIM 3: Time-Aware Target Encoding (Sızıntısız)")
    print("=" * 70)

    # Kronolojik sıralama (çok kritik!)
    df = df.sort_values(["Tarih", "Kosu_ID", "Siralama"]).reset_index(drop=True)

    encoding_groups = [
        ("Jokey_Adi",   "Jokey"),
        ("Antrenor_Adi","Antrenor"),
        ("Baba",        "Baba"),
        ("Anne",        "Anne"),
    ]

    targets = [
        ("Is_Winner", "Win_Rate"),
        ("Is_Top3",   "Top3_Rate"),
    ]

    total_ops = len(encoding_groups) * len(targets)
    op_count = 0

    for group_col, prefix in encoding_groups:
        # NaN'li grup değerlerini "BILINMIYOR" ile doldur
        df[group_col] = df[group_col].fillna("BILINMIYOR")

        for target_col, target_suffix in targets:
            op_count += 1
            col_name = f"{prefix}_{target_suffix}"
            print(f"  [{op_count}/{total_ops}] {col_name} hesaplanıyor "
                  f"(group={group_col})...")

            df = _compute_cumulative_encoding(
                df, group_col, target_col, col_name
            )

    # ── Baba × Anne Interaction (soy hattı kombinasyonu) ──
    print("  [EKSTRA] Baba_Anne_Interaction hesaplanıyor...")
    df["Baba_Anne_Key"] = df["Baba"].astype(str) + "_x_" + df["Anne"].astype(str)
    for target_col, target_suffix in targets:
        col_name = f"BabaAnne_{target_suffix}"
        df = _compute_cumulative_encoding(
            df, "Baba_Anne_Key", target_col, col_name
        )
        print(f"        → {col_name} tamamlandı.")

    print("\n  ADIM 3 TAMAMLANDI ✓")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 4: RAKİPLERE GÖRE GÖRECELİ ÖZELLİKLER (RELATIVE FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

def step4_relative_features(df):
    """
    Her yarışın (Unique_Race_ID) içindeki ortalamalara kıyasla
    göreceli güç oranlarını hesaplar (LTR desteği).
    """
    print("\n" + "=" * 70)
    print("  ADIM 4: Göreceli Özellikler (Relative Features)")
    print("=" * 70)

    relative_configs = [
        ("Handikap_Puani", "Relative_Handikap"),
        ("Siklet_Sayi",    "Relative_Siklet"),
        ("Yas_Sayi",       "Relative_Yas"),
    ]

    for source_col, target_col in relative_configs:
        print(f"  → {target_col} hesaplanıyor ({source_col} / yarış ortalaması)...")

        race_mean = df.groupby("Unique_Race_ID")[source_col].transform("mean")
        df[target_col] = df[source_col] / (race_mean + EPSILON)

    # Ek: Yarıştaki at sayısı (koşu büyüklüğü)
    print("  → Yaris_At_Sayisi hesaplanıyor...")
    df["Yaris_At_Sayisi"] = df.groupby("Unique_Race_ID")["at_id"].transform("count")

    print("\n  ADIM 4 TAMAMLANDI ✓")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SON ADIM: FEATURE SEÇİMİ VE KAYIT
# ══════════════════════════════════════════════════════════════════════════════

def step5_finalize_and_save(df):
    """
    Model eğitimi için gerekli sütunları seçer, gereksizleri atar,
    master_feature_matrix.csv'ye kaydeder.
    """
    print("\n" + "=" * 70)
    print("  SON ADIM: Feature Seçimi ve Kayıt")
    print("=" * 70)

    # Modele girecek sütunlar
    feature_cols = [
        # ─── Identifiers (modelde kullanılmaz, analiz için) ───
        "Unique_Race_ID", "Tarih", "Kosu_ID", "Sehir", "at_id", "At_Adi",

        # ─── Hedef Değişken ───
        "Siralama", "Is_Winner", "Is_Top3",

        # ─── Ham Numerik Özellikler ───
        "Yas_Sayi", "Siklet_Sayi", "Start_Sayi", "Ganyan_Sayi",
        "Handikap_Puani",
        # NOT: "Derece_Saniye" KASITLI OLARAK ÇIKARILDI —
        #      Mevcut yarışın koşu süresidir, yarış bitmeden bilinemez (Data Leakage).

        # ─── Pist Bilgisi (Kategorik) ───
        "Pist_Turu",

        # ─── İdman Özellikleri ───
        "Derece_400m_sn", "Derece_600m_sn", "Derece_800m_sn",
        # NOT: "Derece_1000m_sn" (%97 null) ve "Derece_1200m_sn" (%99.6 null)
        #      KASITLI OLARAK ÇIKARILDI — neredeyse tamamen boş, gürültüden ibaret.
        "Idman_Yaris_Arasi_Gun",

        # ─── Target Encoding (Sızıntısız) ───
        "Jokey_Win_Rate", "Jokey_Top3_Rate",
        "Antrenor_Win_Rate", "Antrenor_Top3_Rate",
        "Baba_Win_Rate", "Baba_Top3_Rate",
        "Anne_Win_Rate", "Anne_Top3_Rate",
        "BabaAnne_Win_Rate", "BabaAnne_Top3_Rate",

        # ─── Göreceli Özellikler ───
        "Relative_Handikap", "Relative_Siklet", "Relative_Yas",
        "Yaris_At_Sayisi",
    ]

    # Sadece mevcut olanları al
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols   = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        print(f"\n  ⚠ Planlanıp bulunamayan {len(missing_cols)} sütun:")
        for c in missing_cols:
            print(f"      - {c}")

    df_final = df[available_cols].copy()

    # Nihai istatistikler
    print(f"\n  ▸ Toplam satır sayısı  : {len(df_final):,}")
    print(f"  ▸ Toplam sütun sayısı  : {len(df_final.columns)}")
    print(f"  ▸ NaN oranları (Top 10):")
    nan_pct = (df_final.isnull().sum() / len(df_final) * 100).sort_values(ascending=False)
    for col, pct in nan_pct.head(10).items():
        if pct > 0:
            print(f"      {col:.<35} %{pct:.1f}")

    # Kaydet
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Dosya kaydedildi: {OUTPUT_CSV}")
    print(f"     Boyut: {os.path.getsize(OUTPUT_CSV) / (1024*1024):.2f} MB")

    return df_final


# ══════════════════════════════════════════════════════════════════════════════
#  ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  TJK AŞAMA 3: MASTER FEATURE MATRIX – PIPELINE BAŞLATILIYOR")
    print("█" * 70)

    # Adım 1: Temizlik
    df_yaris, df_statik, df_idman = step1_data_cleaning()

    # Adım 2: Birleştirme
    df = step2_strict_merge(df_yaris, df_statik, df_idman)

    # Adım 3: Target Encoding
    df = step3_target_encoding(df)

    # Adım 4: Göreceli Özellikler
    df = step4_relative_features(df)

    # Son Adım: Kayıt
    df_final = step5_finalize_and_save(df)

    print("\n" + "█" * 70)
    print("  PIPELINE TAMAMLANDI – master_feature_matrix.csv HAZIR! 🏇")
    print("█" * 70 + "\n")

    return df_final


if __name__ == "__main__":
    main()
