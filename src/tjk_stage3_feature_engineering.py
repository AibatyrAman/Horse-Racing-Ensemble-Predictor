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
            # Salise basamak sayısına göre bölünür: "1.27.3" → 0.3sn, "1.27.30" → 0.30sn
            salise  = int(parts[2]) / (10 ** len(parts[2]))
            return dakika * 60 + saniye + salise
        except (ValueError, IndexError):
            return np.nan
    elif len(parts) == 2:
        # "25.50" gibi sadece saniye.salise formatı
        try:
            saniye  = int(parts[0])
            salise  = int(parts[1]) / (10 ** len(parts[1]))
            return saniye + salise
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


def normalize_sehir(sehir):
    """
    Şehir etiketini normalize eder: 'İzmir  (19. Y.G.)' → 'İzmir'.
    Site parantez içindeki gün etiketini zamanla değiştirdiğinden
    (örn. '1. Yarış Günü' → '19. Y.G.') Unique_Race_ID ham string'e
    bağlanamaz; program-sonuç eşleşmesi bu normalize adla kurulur.
    """
    if pd.isna(sehir):
        return ""
    s = re.sub(r'\(.*?\)', '', str(sehir))
    return re.sub(r'\s+', ' ', s).strip()


def parse_yas(yas_str):
    """
    '3y d  e' gibi string yaş bilgisinden sayısal yaşı çıkarır.
    """
    if pd.isna(yas_str):
        return np.nan
    m = re.match(r'(\d+)', str(yas_str).strip())
    return int(m.group(1)) if m else np.nan


# TJK cinsiyet kodları (Yas alanının SON token'ı):
#   d=dişi tay, k=kısrak (dişi) | e=erkek tay, a=aygır, g=iğdiş (erkek)
CINSIYET_DISI  = {"d", "k"}
CINSIYET_ERKEK = {"e", "a", "g"}


def parse_cinsiyet(yas_str):
    """
    '3y a  e' → 'e'. Yas alanı '[yaş]y [don] [cinsiyet]' formatındadır;
    son token cinsiyet kodudur. Belirsiz/eksikse NaN.
    """
    if pd.isna(yas_str):
        return np.nan
    tokens = str(yas_str).split()
    if len(tokens) >= 3 and tokens[-1].lower() in CINSIYET_DISI | CINSIYET_ERKEK:
        return tokens[-1].lower()
    return np.nan


def parse_don(yas_str):
    """
    '3y a  e' → 'a'. Orta token don (tüy rengi): d=doru, a=al, k=kır, y=yağız.
    (Cinsiyet koduyla harf çakışması olduğundan yalnız 3+ token'da güvenlidir.)
    """
    if pd.isna(yas_str):
        return np.nan
    tokens = str(yas_str).split()
    if len(tokens) >= 3:
        return tokens[1].lower()
    return np.nan


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

    # Unique Race ID (normalize şehir adıyla — site etiketi değişse de sabit kalır)
    df_yaris["Unique_Race_ID"] = (
        df_yaris["Tarih"].dt.strftime("%Y%m%d") + "_" +
        df_yaris["Sehir"].apply(normalize_sehir) + "_" +
        df_yaris["Kosu_ID"].astype(str)
    )

    # Sıralama temizliği: metin değerleri NaN yap, float'a çevir
    df_yaris["Siralama"] = pd.to_numeric(df_yaris["Siralama"], errors="coerce")

    # Derece → saniye
    print("      → Koşu dereceleri saniyeye çevriliyor...")
    df_yaris["Derece_Saniye"] = df_yaris["Derece"].apply(derece_to_seconds)

    # Yaş → sayısal + cinsiyet/don (Yas alanı '[yaş]y [don] [cinsiyet]' taşır)
    df_yaris["Yas_Sayi"]  = df_yaris["Yas"].apply(parse_yas)
    df_yaris["Cinsiyet"]  = df_yaris["Yas"].apply(parse_cinsiyet)
    df_yaris["Don"]       = df_yaris["Yas"].apply(parse_don)
    df_yaris["Cinsiyet_Disi"] = np.where(
        df_yaris["Cinsiyet"].isin(list(CINSIYET_DISI)), 1.0,
        np.where(df_yaris["Cinsiyet"].isin(list(CINSIYET_ERKEK)), 0.0, np.nan)
    )

    # Sıklet → sayısal (zaten çoğunlukla sayısal, ama garantiye alalım)
    df_yaris["Siklet_Sayi"] = pd.to_numeric(df_yaris["Siklet"], errors="coerce")

    # Start → sayısal
    df_yaris["Start_Sayi"] = pd.to_numeric(df_yaris["Start"], errors="coerce")

    # Ganyan → sayısal
    df_yaris["Ganyan_Sayi"] = pd.to_numeric(df_yaris["Ganyan"], errors="coerce")

    # Mesafe / İkramiye / AGF / Fark — yeni ham alanlar (eski satırlarda olmayabilir)
    df_yaris["Mesafe_m"]    = pd.to_numeric(df_yaris.get("Mesafe"), errors="coerce")
    df_yaris["Ikramiye_Sayi"] = pd.to_numeric(df_yaris.get("Ikramiye_1"), errors="coerce")
    df_yaris["AGF_Oran"]    = pd.to_numeric(df_yaris.get("AGF_Oran"), errors="coerce")
    df_yaris["AGF_Sira"]    = pd.to_numeric(df_yaris.get("AGF_Sira"), errors="coerce")
    df_yaris["Fark_Boy"]    = pd.to_numeric(df_yaris.get("Fark_Boy"), errors="coerce")
    # İkramiye çok çarpık → log; sınıf vekili olarak kullanılır
    df_yaris["Ikramiye_Log"] = np.log1p(df_yaris["Ikramiye_Sayi"])
    # Mesafe kovası: sprint (<1400) / mil (1400-1800) / uzun (>1800)
    df_yaris["Mesafe_Bucket"] = pd.cut(
        df_yaris["Mesafe_m"], bins=[0, 1399, 1800, 9999],
        labels=["sprint", "mil", "uzun"]
    ).astype("object")

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
    Verilen group_col (Jokey_Adi, Antrenor_Adi, Baba, ...) için YARIŞ
    granülaritesinde geciktirilmiş kümülatif ortalama hesaplar: bir satırın
    değeri, o grubun MEVCUT YARIŞ HARİÇ önceki tüm yarışlardaki ortalamasıdır.

    Satır-bazlı shift(1) yeterli DEĞİLDİR: aynı yarışta koşan aynı
    antrenör/baba/anne'nin ikinci atı, birincinin o yarıştaki sonucunu
    görürdü (kardeş sızıntısı). Yarış-bazlı gecikme bunu tamamen engeller.

    df kronolojik sıralı olmalıdır (step3 başında sort ediliyor).
    """
    # (grup, yarış) agregatları — sort=False ile kronolojik ilk-görülme sırası korunur
    agg = (
        df.groupby([group_col, "Unique_Race_ID"], sort=False)[target_col]
          .agg(race_sum="sum", race_cnt="count")
          .reset_index()
    )
    cum_sum = agg.groupby(group_col, sort=False)["race_sum"].cumsum() - agg["race_sum"]
    cum_cnt = agg.groupby(group_col, sort=False)["race_cnt"].cumsum() - agg["race_cnt"]
    agg[col_name] = cum_sum / cum_cnt  # cum_cnt==0 → NaN (ilk yarış)

    df = df.merge(
        agg[[group_col, "Unique_Race_ID", col_name]],
        on=[group_col, "Unique_Race_ID"], how="left",
    )

    # İlk kez yarışanlar: yarış-bazlı geciktirilmiş GLOBAL ortalama
    # (tüm veri setinin ortalaması kullanılmaz — o gelecek bilgisi içerirdi)
    gagg = (
        df.groupby("Unique_Race_ID", sort=False)[target_col]
          .agg(g_sum="sum", g_cnt="count")
          .reset_index()
    )
    g_prior = (gagg["g_sum"].cumsum() - gagg["g_sum"]) / (gagg["g_cnt"].cumsum() - gagg["g_cnt"])
    prior_map = pd.Series(g_prior.values, index=gagg["Unique_Race_ID"])
    df[col_name] = df[col_name].fillna(df["Unique_Race_ID"].map(prior_map))

    # Veri setinin ilk yarışı için geçmiş yok → temkinli sabit
    df[col_name] = df[col_name].fillna(0.5)

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
#  ADIM 3B: ATIN KENDİ FORM GEÇMİŞİ (yarış-bazlı lag — sızıntısız)
# ══════════════════════════════════════════════════════════════════════════════

def step3b_form_features(df):
    """
    Atın KENDİ geçmiş koşularından türetilen form özellikleri. Tümü yalnız
    geçmiş yarışları kullanır (at yarışta bir kez koşar → satır-shift =
    yarış-shift). Hız figürü mesafeden bağımsızdır: aynı yarışta herkes aynı
    mesafeyi koştuğundan, yarış medyan süresine göre göreli hız (sn; + = hızlı)
    parkur/mesafe etkisini otomatik kontrol eder.
    """
    print("\n" + "=" * 70)
    print("  ADIM 3B: At Form Geçmişi (sızıntısız)")
    print("=" * 70)

    # Yarış-içi göreli hız (yalnız derecesi olan atlar arasında)
    race_med = df.groupby("Unique_Race_ID")["Derece_Saniye"].transform("median")
    df["_rel_speed"] = race_med - df["Derece_Saniye"]

    g = df.groupby("at_id", sort=False)
    print("  → At_Yaris_Sayisi / At_Son_Yaris_Gun / At_Son3_Ort_Siralama...")
    df["At_Yaris_Sayisi"]  = g.cumcount()                       # geçmiş koşu adedi
    df["At_Son_Yaris_Gun"] = g["Tarih"].diff().dt.days          # dinlenme süresi
    df["At_Son3_Ort_Siralama"] = g["Siralama"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    print("  → Hız figürü: At_Son_RelSpeed / At_RelSpeed_Son3 / At_RelSpeed_Best...")
    df["At_Son_RelSpeed"]  = g["_rel_speed"].shift(1)
    df["At_RelSpeed_Son3"] = g["_rel_speed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["At_RelSpeed_Best"] = g["_rel_speed"].transform(
        lambda x: x.shift(1).expanding().max())

    print("  → At_Win_Rate / At_Top3_Rate / At_PistTuru_Win_Rate...")
    df = _compute_cumulative_encoding(df, "at_id", "Is_Winner", "At_Win_Rate")
    df = _compute_cumulative_encoding(df, "at_id", "Is_Top3",   "At_Top3_Rate")
    df["_At_Pist_Key"] = df["at_id"].astype(str) + "_" + df["Pist_Turu"].astype(str)
    df = _compute_cumulative_encoding(df, "_At_Pist_Key", "Is_Winner", "At_PistTuru_Win_Rate")

    # ── Mesafe uzmanlığı (sprinter/stayer) — Pist_Turu deseninin mesafe analogu ──
    print("  → At_Mesafe_Win_Rate / At_Mesafe_RelSpeed_Best (mesafe uzmanlığı)...")
    df["_At_Mesafe_Key"] = df["at_id"].astype(str) + "_" + df["Mesafe_Bucket"].astype(str)
    df = _compute_cumulative_encoding(df, "_At_Mesafe_Key", "Is_Winner", "At_Mesafe_Win_Rate")
    # Atın BU mesafe kovasındaki geçmiş en iyi göreli hızı (sızıntısız: shift+expanding)
    gm = df.groupby("_At_Mesafe_Key", sort=False)
    df["At_Mesafe_RelSpeed_Best"] = gm["_rel_speed"].transform(
        lambda x: x.shift(1).expanding().max())

    # ── Sınıf hareketi: bugünkü sınıf − atın geçmiş ortalama sınıfı ──
    # (+ = sınıf yükseliyor/zorlaşıyor, − = sınıf düşüyor/kolaylaşıyor: değer sinyali)
    # NOT: yukarıdaki encoding'ler df'yi merge ile yeniden kurduğundan taze
    # groupby şart (eski `g` bayat kalır).
    print("  → Sinif_Degisim (ikramiye bazlı sınıf iniş/çıkışı)...")
    g2 = df.groupby("at_id", sort=False)
    df["_At_Gecmis_Sinif"] = g2["Ikramiye_Log"].transform(
        lambda x: x.shift(1).expanding().mean())
    df["Sinif_Degisim"] = df["Ikramiye_Log"] - df["_At_Gecmis_Sinif"]

    df = df.drop(columns=["_rel_speed", "_At_Pist_Key", "_At_Mesafe_Key", "_At_Gecmis_Sinif"])
    print("\n  ADIM 3B TAMAMLANDI ✓")
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

    # Cinsiyet kompozisyonu (yarış öncesi bilinir — sızıntı yok).
    # Ganyancı gözlemi: erkek ağırlıklı yarıştaki azınlık dişinin şansı yüksek.
    if "Cinsiyet_Disi" in df.columns:
        print("  → Yaris_Disi_Orani / Azinlik_Disi hesaplanıyor...")
        df["Yaris_Disi_Orani"] = (
            df.groupby("Unique_Race_ID")["Cinsiyet_Disi"].transform("mean")
        )
        df["Azinlik_Disi"] = (
            (df["Cinsiyet_Disi"] == 1.0) & (df["Yaris_Disi_Orani"] <= 0.25)
        ).astype(float)

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

        # ─── Cinsiyet / Don (Yas alanından parse; kompozisyon yarış öncesi bilinir) ───
        "Cinsiyet", "Don",
        "Cinsiyet_Disi", "Yaris_Disi_Orani", "Azinlik_Disi",

        # ─── At Form Geçmişi (kendi koşuları — yarış-bazlı lag, sızıntısız) ───
        "At_Win_Rate", "At_Top3_Rate", "At_Yaris_Sayisi",
        "At_Son_Yaris_Gun", "At_Son3_Ort_Siralama", "At_PistTuru_Win_Rate",
        "At_Son_RelSpeed", "At_RelSpeed_Son3", "At_RelSpeed_Best",

        # ─── Mesafe / Sınıf (yeni; bahis öncesi bilinir, sızıntısız) ───
        # Mesafe_Bucket yalnız iç anahtar (At_Mesafe_* için); modele Mesafe_m yeter.
        "Mesafe_m",
        "At_Mesafe_Win_Rate", "At_Mesafe_RelSpeed_Best",
        "Ikramiye_Log", "Sinif_Degisim",

        # ─── Piyasa: halk parası (AGF) — MARKET_FEATURES, ablation'da çıkarılır ───
        "AGF_Oran", "AGF_Sira",
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

    # Adım 3B: At form geçmişi (kendi koşuları — sızıntısız)
    df = step3b_form_features(df)

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
