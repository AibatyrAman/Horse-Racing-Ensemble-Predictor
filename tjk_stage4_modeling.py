#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK AŞAMA 4 – Modelleme ve Ensemble Pipeline
  Sıfır Veri Sızıntısı (Zero Data Leakage) | İleri-zincirli TimeSeriesSplit CV | ROI Değerlendirme
================================================================================
  Girdi:
      master_feature_matrix.csv  – Stage 3 çıktısı
  Çıktılar:
      models/                    – Eğitilmiş modeller (.pkl) + metadata (.json)
      reports/model_comparison.csv – Tüm modellerin karşılaştırma tablosu
================================================================================
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

# Gradient Boosting
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[UYARI] xgboost kurulu değil. pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[UYARI] lightgbm kurulu değil. pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[UYARI] catboost kurulu değil. pip install catboost")

# Hiperparametre optimizasyonu
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[UYARI] optuna kurulu değil. pip install optuna  (Optuna adımı atlanacak)")

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[UYARI] shap kurulu değil. pip install shap  (SHAP adımı atlanacak)")

import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlarda da çalışsın
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────── AYARLAR ───────────────────────
BASE_DIR    = Path(__file__).parent
INPUT_CSV   = BASE_DIR / "master_feature_matrix.csv"
MODEL_DIR   = BASE_DIR / "models"
REPORT_DIR  = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
EPSILON      = 1e-9

# Yarıştan ÖNCE bilinemeyen (leakage) sütunlar — kesinlikle feature olamaz
LEAKAGE_COLS = ["Derece_Saniye", "Siralama"]

# Kimlik / meta sütunlar (modelde kullanılmaz)
IDENTIFIER_COLS = ["Unique_Race_ID", "Tarih", "Kosu_ID", "at_id", "At_Adi"]

# Hedef değişkenler
TARGET_COLS = ["Is_Winner", "Is_Top3"]

# Neredeyse tamamen boş — stage 3'te kaldırıldı, yine de kontrol
ULTRA_SPARSE = ["Derece_1000m_sn", "Derece_1200m_sn"]


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 1: VERİ YÜKLEME VE DOĞRULAMA
# ══════════════════════════════════════════════════════════════════════════════

def load_and_validate():
    """
    master_feature_matrix.csv'yi yükler, leakage kontrolü yapar,
    temel feature mühendisliği ile zenginleştirir.
    """
    print("=" * 70)
    print("  ADIM 1: Veri Yükleme ve Doğrulama")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")
    # Kronolojik sıra + pozisyonel hizalama garantisi (zaman-bazlı CV için kritik)
    df = df.sort_values("Tarih", kind="stable").reset_index(drop=True)
    print(f"  → {len(df):,} satır, {len(df.columns)} sütun yüklendi.")
    print(f"  → Tarih aralığı: {df['Tarih'].min().date()} → {df['Tarih'].max().date()}")
    print(f"  → Unique yarış sayısı: {df['Unique_Race_ID'].nunique()}")

    # Leakage güvenlik kontrolü
    for col in LEAKAGE_COLS + ULTRA_SPARSE:
        if col in df.columns:
            # LEAKAGE sütunları feature listesinde kullanılmayacak,
            # varlığını logla ama kaldırma (analiz için gerekebilir)
            print(f"  ⚠ '{col}' mevcut — feature listesine dahil edilmeyecek.")

    # Sehir temizliği: "Adana (1. Yarış Günü)" → "Adana"
    df["Sehir_Temiz"] = df["Sehir"].astype(str).str.split("(").str[0].str.strip()

    # Yabancı yarış flag'i
    yabanci_keywords = ["ABD", "İngiltere", "Fransa", "Almanya", "Park", "Emirlik"]
    df["Is_Foreign"] = df["Sehir_Temiz"].apply(
        lambda s: int(any(k in s for k in yabanci_keywords))
    )

    # Ganyan'dan implied probability (piyasa sinyali)
    df["Ganyan_Implied_Prob"] = np.where(
        df["Ganyan_Sayi"].notna() & (df["Ganyan_Sayi"] > 0),
        1.0 / df["Ganyan_Sayi"],
        np.nan
    )

    # Yarış içi Ganyan sıralaması (1 = favorit, yüksek = uzun ihtimal)
    df["Ganyan_Rank_InRace"] = df.groupby("Unique_Race_ID")["Ganyan_Sayi"].rank(
        method="min", ascending=True
    )

    # Normalize start pozisyonu
    df["Start_Normalized"] = df["Start_Sayi"] / (df["Yaris_At_Sayisi"] + EPSILON)

    # Jokey + Antrenör sinerji skoru
    df["Jokey_Trainer_Synergy"] = (
        df["Jokey_Win_Rate"].fillna(0) + df["Antrenor_Win_Rate"].fillna(0)
    ) / 2.0

    # Soy hattı skorları (ayrı ayrı — model kendi ağırlığını öğrensin)
    df["Baba_Bloodline"] = (
        df["Baba_Win_Rate"].fillna(0) + df["Baba_Top3_Rate"].fillna(0)
    ) / 2.0
    df["Anne_Bloodline"] = (
        df["Anne_Win_Rate"].fillna(0) + df["Anne_Top3_Rate"].fillna(0)
    ) / 2.0

    # Missingness flag'leri (boş olması da bilgi taşır)
    for col, flag in [
        ("Derece_400m_sn", "Has_400m_data"),
        ("Derece_600m_sn", "Has_600m_data"),
        ("Derece_800m_sn", "Has_800m_data"),
        ("Idman_Yaris_Arasi_Gun", "Has_Idman"),
        ("Handikap_Puani", "Has_Handikap"),
    ]:
        df[flag] = df[col].notna().astype(int)

    print(f"  ✓ Is_Winner oranı: %{df['Is_Winner'].mean()*100:.1f}")
    print(f"  ✓ Is_Top3 oranı  : %{df['Is_Top3'].mean()*100:.1f}")
    print(f"  ✓ Yabancı yarış  : {df['Is_Foreign'].sum()} satır")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 2: FEATURE SEÇİMİ VE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Final feature listesi (leakage yok, ultra-sparse yok)
FEATURE_COLS = [
    # Temel numerik
    "Yas_Sayi", "Siklet_Sayi", "Start_Sayi", "Start_Normalized",
    # Bahis piyasası sinyali
    "Ganyan_Sayi", "Ganyan_Implied_Prob", "Ganyan_Rank_InRace",
    # Handikap
    "Handikap_Puani", "Relative_Handikap",
    # Göreceli
    "Relative_Siklet", "Relative_Yas", "Yaris_At_Sayisi",
    # İnsan uzmanlığı (target encoded, sızıntısız)
    "Jokey_Win_Rate", "Jokey_Top3_Rate",
    "Antrenor_Win_Rate", "Antrenor_Top3_Rate",
    "Baba_Win_Rate", "Baba_Top3_Rate",
    "Anne_Win_Rate", "Anne_Top3_Rate",
    "BabaAnne_Win_Rate", "BabaAnne_Top3_Rate",
    # Mühendislik
    "Jokey_Trainer_Synergy", "Baba_Bloodline", "Anne_Bloodline",
    # İdman (seyrek ama bilgi taşıyor)
    "Derece_400m_sn", "Derece_600m_sn", "Derece_800m_sn",
    "Idman_Yaris_Arasi_Gun",
    # Missingness flag'leri
    "Has_400m_data", "Has_600m_data", "Has_800m_data",
    "Has_Idman", "Has_Handikap",
    # Mekan
    "Sehir_Encoded", "Is_Foreign",
]

# Kategorik sütunlar (OrdinalEncoder uygulanacak)
CAT_COLS = ["Sehir_Temiz"]

# Piyasa (bahis oranı) sinyalinden türeyen feature'lar — ablation modunda çıkarılır.
# Hepsi Ganyan'ın (kazanma oranı) fonksiyonudur → "piyasa olmadan" senaryosu için kaldırılır.
# NOT: Ganyan_Sayi yine de df'te kalır; ROI/baseline onu df'ten okur (feature değil).
MARKET_FEATURES = ["Ganyan_Sayi", "Ganyan_Implied_Prob", "Ganyan_Rank_InRace"]


def prepare_features(df, drop_features=None):
    """
    HAM (NaN'lı) feature matrisini döndürür. Imputation ve OrdinalEncoder
    BURADA UYGULANMAZ — sızıntıyı önlemek için her CV fold'unda yalnızca
    eğitim kısmında fit edilir (bkz. prepare_fold).

    'Sehir_Encoded' kolonu placeholder (NaN) olarak eklenir; gerçek değer
    fold içinde (veya tüm veride, production için) hesaplanır.

    drop_features: ablation için feature listesinden çıkarılacak sütunlar
                   (örn. MARKET_FEATURES = piyasa/Ganyan sinyali).
    """
    print("\n" + "=" * 70)
    print("  ADIM 2: Feature Hazırlığı (ham matris — fold içi fit)")
    print("=" * 70)

    # Sehir_Encoded placeholder — gerçek encode prepare_fold içinde yapılır
    df["Sehir_Encoded"] = np.nan

    drop_set = set(drop_features or [])
    available = [c for c in FEATURE_COLS if c in df.columns and c not in drop_set]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠ Eksik feature'lar ({len(missing)} adet): {missing}")
    if drop_set:
        dropped = [c for c in FEATURE_COLS if c in drop_set]
        print(f"  ⓧ ABLATION — çıkarılan feature'lar ({len(dropped)}): {dropped}")

    X_raw = df[available].copy()
    print(f"  → Kullanılan feature sayısı: {len(available)}")
    print(f"  ✓ Ham matris hazır ({X_raw.shape[0]:,} satır). Imputation fold içinde yapılacak.")

    return X_raw, available


def _recompute_relative_handikap(X_part, race_ids):
    """Fold içi yarış ortalamasına göre Relative_Handikap'ı yeniden hesaplar.
    Bir yarış tamamen tek tarafta (train ya da test) olduğundan sızıntı yoktur."""
    if "Relative_Handikap" not in X_part.columns or "Handikap_Puani" not in X_part.columns:
        return X_part
    tmp = pd.DataFrame({"_r": np.asarray(race_ids), "_h": X_part["Handikap_Puani"].to_numpy()})
    race_mean = tmp.groupby("_r")["_h"].transform("mean").to_numpy()
    X_part["Relative_Handikap"] = X_part["Handikap_Puani"].to_numpy() / (race_mean + EPSILON)
    return X_part


def prepare_fold(X_raw, df, train_idx, test_idx):
    """
    Tek bir fold için preprocessing: OrdinalEncoder + KNNImputer + median
    doldurma YALNIZCA eğitim kısmında fit edilir, her iki tarafa uygulanır.
    Production için train_idx == test_idx == tüm satırlar verilerek çağrılabilir.
    """
    X_tr = X_raw.iloc[train_idx].copy()
    X_te = X_raw.iloc[test_idx].copy()

    # ── Sehir ordinal encode (train'de fit) ──
    if "Sehir_Encoded" in X_tr.columns and "Sehir_Temiz" in df.columns:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        sehir_tr = df[["Sehir_Temiz"]].iloc[train_idx]
        sehir_te = df[["Sehir_Temiz"]].iloc[test_idx]
        enc.fit(sehir_tr)
        X_tr["Sehir_Encoded"] = enc.transform(sehir_tr).astype(float)
        X_te["Sehir_Encoded"] = enc.transform(sehir_te).astype(float)

    # ── KNN impute (Handikap/Yas/Start) — train'de fit ──
    knn_cols = [c for c in ["Handikap_Puani", "Yas_Sayi", "Start_Sayi"] if c in X_tr.columns]
    if knn_cols:
        knn = KNNImputer(n_neighbors=5)
        X_tr[knn_cols] = knn.fit_transform(X_tr[knn_cols])
        X_te[knn_cols] = knn.transform(X_te[knn_cols])

    # ── Relative_Handikap fold içi yeniden hesap (impute sonrası) ──
    X_tr = _recompute_relative_handikap(X_tr, df["Unique_Race_ID"].iloc[train_idx].to_numpy())
    X_te = _recompute_relative_handikap(X_te, df["Unique_Race_ID"].iloc[test_idx].to_numpy())

    # ── Kalan null'ları train medyanı ile doldur (all-NaN kolon güvenliği: fillna(0)) ──
    medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(medians).fillna(0.0)
    X_te = X_te.fillna(medians).fillna(0.0)

    return X_tr, X_te


def make_time_series_splits(df, n_splits=5):
    """
    İleri-zincirli (forward-chaining) zaman-bazlı CV bölmeleri üretir.
    Yarışlar (Unique_Race_ID) ilk tarihlerine göre kronolojik sıralanır;
    sklearn TimeSeriesSplit yarış listesi üzerinde çalışır; her yarışın TÜM
    satırları ilgili tarafa bütün olarak atanır (yarış asla bölünmez).
    Döndürür: [(train_idx, test_idx), ...] pozisyonel dizi listesi.
    """
    race_first_date = df.groupby("Unique_Race_ID")["Tarih"].min().sort_values()
    ordered_races = race_first_date.index.to_numpy()
    race_to_rows = df.groupby("Unique_Race_ID").indices  # race -> pozisyonel satır indeksleri

    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_race_pos, te_race_pos in tss.split(ordered_races):
        tr_races = ordered_races[tr_race_pos]
        te_races = ordered_races[te_race_pos]
        train_idx = np.sort(np.concatenate([race_to_rows[r] for r in tr_races]))
        test_idx  = np.sort(np.concatenate([race_to_rows[r] for r in te_races]))
        splits.append((train_idx, test_idx))
    return splits


def prepare_all_folds(X_raw, df, splits):
    """
    Her fold için preprocessing'i BİR KEZ hesaplar (tüm modeller + Optuna
    yeniden kullanır → KNN tekrar tekrar çalışmaz).
    Döndürür: [(X_tr, X_te, train_idx, test_idx), ...]
    """
    prepared = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, X_te = prepare_fold(X_raw, df, train_idx, test_idx)
        prepared.append((X_tr, X_te, train_idx, test_idx))
        print(f"      Fold {fold_idx}: train={len(train_idx):,} | test={len(test_idx):,}")
    return prepared


def create_split_strategy(df, min_dates_for_time_split=10):
    """
    Veri büyüklüğüne göre otomatik CV stratejisi seç.
    """
    n_dates = df["Tarih"].dt.date.nunique()
    n_races = df["Unique_Race_ID"].nunique()

    if n_dates >= min_dates_for_time_split:
        strategy = "time_based"
    elif n_dates >= 4:
        strategy = "time_based_holdout"
    else:
        strategy = "race_group_cv"

    print(f"\n  → {n_dates} unique tarih, {n_races} yarış.")
    print(f"  → CV stratejisi: '{strategy}'")
    return strategy, n_races


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 3: MODEL TANIMLARI
# ══════════════════════════════════════════════════════════════════════════════

def get_pos_weight(y):
    """Is_Winner için class imbalance oranı."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / (n_pos + EPSILON)


def build_models(y_winner):
    """
    Tüm modelleri (baseline + advanced + ensemble) dict olarak döndürür.
    Her model için (name, model, is_calibrated) tuple.
    """
    spw = get_pos_weight(y_winner)
    print(f"\n  → Is_Winner scale_pos_weight (XGBoost için): {spw:.2f}")

    models = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", C=0.1,
            solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        class_weight="balanced_subsample", max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=15, random_state=RANDOM_STATE
    )

    # ── Gradient Boosting (Gelişmiş) ──────────────────────────────────────────
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=spw,
            tree_method="hist", eval_metric="auc",
            random_state=RANDOM_STATE, n_jobs=-1,
            verbosity=0
        )

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, is_unbalance=True,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05,
            l2_leaf_reg=5.0, auto_class_weights="Balanced",
            boosting_type="Ordered",  # Küçük veri için kritik!
            random_seed=RANDOM_STATE, verbose=0
        )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    # Soft Voting (ağırlıklı olasılık ortalaması)
    voting_estimators = []
    if HAS_LGB:
        voting_estimators.append(("lgbm", lgb.LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            min_child_samples=20, is_unbalance=True,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )))
    if HAS_XGB:
        voting_estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            scale_pos_weight=spw, tree_method="hist",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )))
    if HAS_CAT:
        voting_estimators.append(("cat", CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05,
            l2_leaf_reg=5.0, auto_class_weights="Balanced",
            boosting_type="Ordered", random_seed=RANDOM_STATE, verbose=0
        )))
    voting_estimators.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        class_weight="balanced_subsample", max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1
    )))

    if len(voting_estimators) >= 2:
        # Gradient boosting modellerine 3x ağırlık, RF'e 1x
        weights = [3 if name in ("lgbm", "xgb", "cat") else 1
                   for name, _ in voting_estimators]
        models["VotingEnsemble"] = VotingClassifier(
            estimators=voting_estimators, voting="soft", weights=weights, n_jobs=-1
        )

    # Bagging (LightGBM üzerine — variance reduction)
    if HAS_LGB:
        models["BaggingLGBM"] = BaggingClassifier(
            estimator=lgb.LGBMClassifier(
                n_estimators=200, num_leaves=15, is_unbalance=True,
                random_state=RANDOM_STATE, verbose=-1
            ),
            n_estimators=30, max_samples=0.8, max_features=0.8,
            bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1
        )

    # Stacking (Linear meta-learner)
    stacking_estimators = []
    if HAS_XGB:
        stacking_estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=200, max_depth=3, scale_pos_weight=spw,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )))
    if HAS_LGB:
        stacking_estimators.append(("lgbm", lgb.LGBMClassifier(
            n_estimators=200, num_leaves=15, is_unbalance=True,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )))
    if HAS_CAT:
        stacking_estimators.append(("cat", CatBoostClassifier(
            iterations=200, depth=4, auto_class_weights="Balanced",
            boosting_type="Ordered", random_seed=RANDOM_STATE, verbose=0
        )))
    stacking_estimators.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
    )))
    stacking_estimators.append(("lr", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000))
    ])))

    if len(stacking_estimators) >= 3:
        models["StackingEnsemble"] = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            # İç CV: cross_val_predict bir PARTITION ister (TimeSeriesSplit olmaz).
            # Dış eğitim fold'u tarihe göre sıralı + yarış-ardışık geldiğinden,
            # shuffle=False KFold ardışık ZAMAN BLOKLARI üretir → orijinal
            # StratifiedKFold'a kıyasla yarış-içi ve zamansal sızıntıyı azaltır.
            cv=KFold(n_splits=5, shuffle=False),
            stack_method="predict_proba",
            n_jobs=-1
        )

    return models


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 4: DEĞERLENDİRME METRİKLERİ
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k_per_race(df_eval, prob_col, target_col, k=1):
    """
    Her yarışta top-k tahmin edilen atın gerçekte top-k bitirme oranı.
    k=1 → kazananı doğru tahmin etme oranı
    k=3 → top-3'e giren atı tahmin etme oranı
    """
    hits = []
    for _, group in df_eval.groupby("Unique_Race_ID"):
        if len(group) < k:
            continue
        top_k_idx = group[prob_col].nlargest(k).index
        actual = group.loc[top_k_idx, target_col].sum()
        hits.append(int(actual > 0))
    return np.mean(hits) if hits else 0.0


def calculate_roi(df_eval, prob_col, odds_col, target_col,
                  strategy="top1", value_threshold=1.15, bet_amount=1.0):
    """
    Bahis stratejisi simülasyonu ve ROI hesabı.

    strategy:
        'top1'  — Her yarışta en yüksek olasılıklı ata bahis
        'value' — Model olasılığı > piyasa implied_prob * threshold olanlara bahis
        'kelly' — Kelly kriteri ile bahis boyutu
    """
    total_staked = 0.0
    total_return = 0.0
    n_bets = 0
    n_wins = 0

    for _, group in df_eval.groupby("Unique_Race_ID"):
        group = group.copy()
        group["implied_prob"] = np.where(
            group[odds_col].notna() & (group[odds_col] > 0),
            1.0 / group[odds_col], np.nan
        )

        if strategy == "top1":
            bet_horse = group.loc[group[prob_col].idxmax()]
            stake = bet_amount
            total_staked += stake
            n_bets += 1
            if bet_horse[target_col] == 1:
                total_return += stake * float(bet_horse[odds_col])
                n_wins += 1

        elif strategy == "value":
            # Sadece modelin piyasadan daha yüksek değerlendirdiği atlara bahis
            group["edge"] = group[prob_col] - group["implied_prob"].fillna(0)
            value_horses = group[
                group[prob_col] > group["implied_prob"].fillna(0) * value_threshold
            ]
            if len(value_horses) > 0:
                bet_horse = value_horses.loc[value_horses[prob_col].idxmax()]
                if pd.notna(bet_horse[odds_col]) and bet_horse[odds_col] > 0:
                    total_staked += bet_amount
                    n_bets += 1
                    if bet_horse[target_col] == 1:
                        total_return += bet_amount * float(bet_horse[odds_col])
                        n_wins += 1

        elif strategy == "kelly":
            # Kelly kriteri: f = (p*b - q) / b  |  b = net odds (ganyan - 1)
            best_horse = group.loc[group[prob_col].idxmax()]
            if pd.isna(best_horse[odds_col]) or best_horse[odds_col] <= 1:
                continue
            p = float(best_horse[prob_col])
            b = float(best_horse[odds_col]) - 1.0
            q = 1.0 - p
            kelly_f = max(0.0, (p * b - q) / (b + EPSILON))
            kelly_f = min(kelly_f, 0.25)  # Max %25 bankroll koruması
            stake = bet_amount * kelly_f
            if stake < 0.01:
                continue
            total_staked += stake
            n_bets += 1
            if best_horse[target_col] == 1:
                total_return += stake * float(best_horse[odds_col])
                n_wins += 1

    roi = (total_return - total_staked) / (total_staked + EPSILON)
    return {
        "roi": round(roi, 4),
        "profit": round(total_return - total_staked, 2),
        "n_bets": n_bets,
        "win_rate": round(n_wins / (n_bets + EPSILON), 4),
    }


def ganyan_baseline(df_eval, target_col, odds_col="Ganyan_Sayi"):
    """Referans: Her yarışta favoriti (en düşük ganyan) seç.

    ROI YALNIZCA Is_Winner için anlamlıdır (kazanma bahsi → kazanma ganyanı).
    Is_Top3 için ROI hesaplanmaz (veride plase ganyanı yok; kazanma ganyanıyla
    plase finişe ödeme yapmak yanlış olur)."""
    hits_p1, hits_p3 = [], []
    total_staked, total_return = 0.0, 0.0

    for _, group in df_eval.groupby("Unique_Race_ID"):
        group = group.dropna(subset=[odds_col])
        if len(group) == 0:
            continue
        fav = group.loc[group[odds_col].idxmin()]
        hits_p1.append(int(fav[target_col] == 1))
        top3_idx = group[odds_col].nsmallest(3).index
        hits_p3.append(int(group.loc[top3_idx, target_col].sum() > 0))
        # ROI sadece Is_Winner için: favori kazanırsa kazanma ganyanı kadar döner
        total_staked += 1.0
        if fav["Is_Winner"] == 1:
            total_return += float(fav[odds_col])

    roi = None
    if target_col == "Is_Winner":
        roi = round((total_return - total_staked) / (total_staked + EPSILON), 4)

    return {
        "P@1": round(np.mean(hits_p1), 4),
        "P@3": round(np.mean(hits_p3), 4),
        "ROI_top1": roi,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 5: CROSS-VALIDATION VE DEĞERLENDİRME
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_cv(model, prepared_folds, y, df_full, target_col, model_name):
    """
    İleri-zincirli zaman-bazlı CV ile modeli değerlendirir (prepared_folds:
    fold içi imputasyonu önceden yapılmış [(X_tr, X_te, train_idx, test_idx), ...]).
    Yalnızca TEST EDİLEN satırlar için OOF tahmin toplanır (en erken yarışlar
    hiçbir test fold'unda yer almaz → metriklerden hariç tutulur).

    ROI (top1/value/kelly) YALNIZCA Is_Winner için hesaplanır; Is_Top3 için None.
    """
    n = len(df_full)
    oof_probs   = np.full(n, np.nan)
    tested_mask = np.zeros(n, dtype=bool)
    fold_aucs, fold_aps = [], []

    for fold_idx, (X_tr, X_te, train_idx, test_idx) in enumerate(prepared_folds):
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        try:
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            oof_probs[test_idx]   = probs
            tested_mask[test_idx] = True

            if len(np.unique(y_te)) > 1:
                fold_aucs.append(roc_auc_score(y_te, probs))
                fold_aps.append(average_precision_score(y_te, probs))
        except Exception as e:
            print(f"    [!] {model_name} fold {fold_idx+1} hatası: {e}")
            oof_probs[test_idx]   = 0.5
            tested_mask[test_idx] = True

    # OOF metriklerini SADECE test edilen satırlar üzerinde hesapla
    df_oof = df_full.loc[tested_mask].copy()
    df_oof["oof_prob"] = oof_probs[tested_mask]

    p1 = precision_at_k_per_race(df_oof, "oof_prob", target_col, k=1)
    p3 = precision_at_k_per_race(df_oof, "oof_prob", target_col, k=3)

    # ROI yalnız Is_Winner için anlamlı (kazanma bahsi → kazanma ganyanı)
    if target_col == "Is_Winner":
        roi_top1  = calculate_roi(df_oof, "oof_prob", "Ganyan_Sayi", target_col, "top1")
        roi_value = calculate_roi(df_oof, "oof_prob", "Ganyan_Sayi", target_col, "value")
        roi_kelly = calculate_roi(df_oof, "oof_prob", "Ganyan_Sayi", target_col, "kelly")
        roi_top1_v, roi_value_v, roi_kelly_v = roi_top1["roi"], roi_value["roi"], roi_kelly["roi"]
        bets_value, winrate_top1 = roi_value["n_bets"], roi_top1["win_rate"]
    else:
        roi_top1_v = roi_value_v = roi_kelly_v = None
        bets_value = winrate_top1 = None

    result = {
        "Model":         model_name,
        "AUC_mean":      round(np.mean(fold_aucs), 4) if fold_aucs else 0,
        "AUC_std":       round(np.std(fold_aucs), 4)  if fold_aucs else 0,
        "AP_mean":       round(np.mean(fold_aps), 4)  if fold_aps  else 0,
        "P@1":           round(p1, 4),
        "P@3":           round(p3, 4),
        "ROI_top1":      roi_top1_v,
        "ROI_value":     roi_value_v,
        "ROI_kelly":     roi_kelly_v,
        "Bets_value":    bets_value,
        "WinRate_top1":  winrate_top1,
    }
    return result, oof_probs


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 6: HİPERPARAMETRE OPTİMİZASYONU (OPTUNA)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_model(model_name, prepared_folds, y, n_trials=50):
    """
    Optuna ile zaman-bazlı CV (prepared_folds) AUC'ünü maksimize eden
    hiperparametreleri bulur. prepared_folds fold içi imputasyonu önceden
    yapılmış [(X_tr, X_te, train_idx, test_idx), ...] listesidir.
    """
    if not HAS_OPTUNA:
        return {}

    def objective(trial):
        # Trial isimleri = gerçek parametre isimleri → study.best_params
        # doğrudan **kwargs olarak modele geçilebilir.
        if model_name == "XGBoost" and HAS_XGB:
            params = {
                "max_depth":        trial.suggest_int("max_depth", 2, 5),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 12.0),
            }
            model = xgb.XGBClassifier(**params, tree_method="hist",
                                       random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

        elif model_name == "LightGBM" and HAS_LGB:
            params = {
                "num_leaves":        trial.suggest_int("num_leaves", 8, 32),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            }
            model = lgb.LGBMClassifier(**params, is_unbalance=True,
                                        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)

        elif model_name == "CatBoost" and HAS_CAT:
            params = {
                "depth":               trial.suggest_int("depth", 3, 7),
                "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "iterations":          trial.suggest_int("iterations", 100, 500),
                "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
                "random_strength":     trial.suggest_float("random_strength", 0.1, 5.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            }
            model = CatBoostClassifier(**params, boosting_type="Ordered",
                                        auto_class_weights="Balanced",
                                        random_seed=RANDOM_STATE, verbose=0)
        else:
            raise ValueError(f"Bilinmeyen model: {model_name}")

        # Zaman-bazlı CV (prepared_folds) — AUC hesapla
        aucs = []
        for X_tr, X_te, train_idx, test_idx in prepared_folds:
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            try:
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:, 1]
                if len(np.unique(y_te)) > 1:
                    aucs.append(roc_auc_score(y_te, probs))
            except Exception:
                aucs.append(0.5)
        return np.mean(aucs) if aucs else 0.5

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"    → {model_name} best AUC: {study.best_value:.4f} | params: {study.best_params}")
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 7: FEATURE IMPORTANCE VE SHAP
# ══════════════════════════════════════════════════════════════════════════════

def analyze_feature_importance(model, X, y, feature_names, model_name, target):
    """
    1. Native feature importance (gain-based, ağaç modelleri için)
    2. Permutation importance (model-agnostic)
    3. SHAP değerleri (tam açıklanabilirlik, varsa)
    """
    print(f"\n  Feature Importance — {model_name} ({target})")

    save_prefix = REPORT_DIR / f"fi_{model_name}_{target}"

    # Native importance
    try:
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            imp = imp.sort_values(ascending=False)
            print("  Top 10 (Native Gain):")
            for fname, val in imp.head(10).items():
                print(f"      {fname:<35} {val:.4f}")

            fig, ax = plt.subplots(figsize=(10, 6))
            imp.head(20).sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"{model_name} — Native Importance ({target})")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig(str(save_prefix) + "_native.png", dpi=120)
            plt.close()
    except Exception as e:
        print(f"    [!] Native importance hatası: {e}")

    # Permutation importance
    try:
        perm = permutation_importance(
            model, X, y, n_repeats=10, random_state=RANDOM_STATE,
            scoring="roc_auc", n_jobs=-1
        )
        perm_imp = pd.Series(perm.importances_mean, index=feature_names)
        perm_imp = perm_imp.sort_values(ascending=False)
        print("  Top 10 (Permutation AUC):")
        for fname, val in perm_imp.head(10).items():
            print(f"      {fname:<35} {val:+.4f}")
    except Exception as e:
        print(f"    [!] Permutation importance hatası: {e}")

    # SHAP
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                              show=False, max_display=20)
            plt.title(f"SHAP Summary — {model_name} ({target})")
            plt.tight_layout()
            plt.savefig(str(save_prefix) + "_shap.png", dpi=120)
            plt.close()
            print(f"  ✓ SHAP grafiği: {save_prefix}_shap.png")
        except Exception as e:
            print(f"    [!] SHAP hatası: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 8: MODEL KAYDETME
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, model_name, target, result_dict, feature_names):
    """Modeli ve meta verisini diske yazar."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"{model_name}_{target}_{ts}.pkl"
    meta_path  = MODEL_DIR / f"{model_name}_{target}_{ts}_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_name":   model_name,
        "target":       target,
        "timestamp":    ts,
        "feature_names": list(feature_names),
        "cv_auc_mean":  result_dict.get("AUC_mean"),
        "cv_auc_std":   result_dict.get("AUC_std"),
        "precision_at_1": result_dict.get("P@1"),
        "roi_value":    result_dict.get("ROI_value"),
        "roi_top1":     result_dict.get("ROI_top1"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Model kaydedildi: {model_path.name}")
    return str(model_path)


def select_and_register_production_model(results_df, target):
    """
    Composite score ile en iyi modeli seçer, production registry'e yazar.
    """
    df = results_df[results_df["Target"] == target].copy()
    if df.empty:
        return None

    use_roi = (target == "Is_Winner")  # ROI yalnız Is_Winner için anlamlı

    # Hard gate'ler
    df = df[df["AUC_mean"] >= 0.55]      # Minimum AUC
    df = df[df["AUC_std"] < 0.15]        # Kararlılık
    if use_roi:
        df = df[df["ROI_value"].fillna(-999) >= -0.20]   # ROI tabanı (sadece Is_Winner)

    if df.empty:
        print(f"  ⚠ {target}: Hiç model production kriterlerini geçemedi.")
        return None

    def _norm(col):
        col_min, col_max = df[col].min(), df[col].max()
        return (df[col] - col_min) / (col_max - col_min + EPSILON)

    df["stability"] = 1.0 - df["AUC_std"] / (df["AUC_mean"] + EPSILON)
    df["stability"] = (df["stability"] - df["stability"].min()) / \
                      (df["stability"].max() - df["stability"].min() + EPSILON)

    if use_roi:
        # Is_Winner: ROI ağırlıklı composite
        df["composite_score"] = (
            0.40 * _norm("ROI_value") +
            0.30 * _norm("P@1") +
            0.20 * _norm("AUC_mean") +
            0.10 * df["stability"]
        )
    else:
        # Is_Top3: ROI yok → P@1 / P@3 / AUC / kararlılık
        df["composite_score"] = (
            0.40 * _norm("P@1") +
            0.25 * _norm("P@3") +
            0.25 * _norm("AUC_mean") +
            0.10 * df["stability"]
        )

    best = df.loc[df["composite_score"].idxmax()]
    roi_disp = f"{best['ROI_value']:.4f}" if use_roi and pd.notna(best['ROI_value']) else "N/A"
    print(f"\n  ★ Production Model ({target}): {best['Model']}")
    print(f"    AUC={best['AUC_mean']:.4f}, P@1={best['P@1']:.4f}, P@3={best['P@3']:.4f}, "
          f"ROI_value={roi_disp}, Score={best['composite_score']:.4f}")

    registry_path = MODEL_DIR / "production_registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)

    registry[target] = {
        "model_name":      best["Model"],
        "cv_auc":          best["AUC_mean"],
        "precision_at_1":  best["P@1"],
        "precision_at_3":  best["P@3"],
        "roi_value":       (best["ROI_value"] if use_roi and pd.notna(best["ROI_value"]) else None),
        "composite_score": best["composite_score"],
        "promoted_at":     datetime.now().isoformat(),
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    return best["Model"]


# ══════════════════════════════════════════════════════════════════════════════
#  ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Argümanlar: ablation modu (piyasa/Ganyan sinyali olmadan) ──
    ablation = ("--ablation" in sys.argv) or ("--no-ganyan" in sys.argv)
    drop_features = MARKET_FEATURES if ablation else None
    suffix = "_ablation" if ablation else ""

    print("\n" + "█" * 70)
    print("  TJK AŞAMA 4: MODELLEME VE ENSEMBLE PIPELINE")
    if ablation:
        print("  MOD: ABLATION (Ganyansız — piyasa sinyali ÇIKARILDI)")
    print("█" * 70)

    # ── 1. Veri yükle ─────────────────────────────────────────────────────────
    df = load_and_validate()

    # ── 2. Feature hazırlığı (ham matris) ─────────────────────────────────────
    X_raw, feature_names = prepare_features(df, drop_features=drop_features)

    n_races = df["Unique_Race_ID"].nunique()
    n_cv_splits = 5 if n_races >= 10 else max(2, n_races // 3)
    print(f"\n  → İleri-zincirli TimeSeriesSplit | n_splits={n_cv_splits} | yarış={n_races:,}")

    # ── Fold içi preprocessing'i bir kez hesapla (tüm modeller + Optuna paylaşır)
    print("  → Fold içi preprocessing hesaplanıyor (imputasyon train'de fit)...")
    splits = make_time_series_splits(df, n_splits=n_cv_splits)
    prepared_folds = prepare_all_folds(X_raw, df, splits)

    # Production (nihai) modeller için tüm veride fit edilmiş matris
    all_idx = np.arange(len(df))
    _, X_full = prepare_fold(X_raw, df, all_idx, all_idx)

    # ── 3. Ganyan Baseline (referans) ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GANYAN BASELINE (referans — her yarışta favoriti seç)")
    print("=" * 70)
    for target in TARGET_COLS:
        bl = ganyan_baseline(df, target_col=target)
        roi_txt = f"{bl['ROI_top1']:+.3f}" if bl['ROI_top1'] is not None else "N/A (Top3'te ROI yok)"
        print(f"  {target}: P@1={bl['P@1']:.3f} | P@3={bl['P@3']:.3f} | ROI={roi_txt}")

    # ── 4. Tüm modelleri eğit ve değerlendir ───────────────────────────────────
    all_results = []
    best_models = {}  # {(model_name, target): fitted_model}

    for target in TARGET_COLS:
        print("\n" + "=" * 70)
        print(f"  HEDEF: {target}")
        print("=" * 70)

        y = df[target]
        models_dict = build_models(y)

        # ── Optuna optimizasyonu (seçili modeller için) ──────────────────────
        optuna_models = [m for m in ["XGBoost", "LightGBM", "CatBoost"]
                         if m in models_dict and HAS_OPTUNA]
        if optuna_models:
            print(f"\n  [Optuna] {len(optuna_models)} model optimize ediliyor "
                  f"(50 trial each)...")
        for model_name in optuna_models:
            print(f"  → {model_name} optimize ediliyor...")
            best_params = optimize_model(model_name, prepared_folds, y, n_trials=50)
            if not best_params:
                continue
            # En iyi parametrelerle modeli güncelle
            if model_name == "XGBoost":
                models_dict[model_name] = xgb.XGBClassifier(
                    **best_params, tree_method="hist",
                    random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
                )
            elif model_name == "LightGBM":
                models_dict[model_name] = lgb.LGBMClassifier(
                    **best_params, is_unbalance=True,
                    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
                )
            elif model_name == "CatBoost":
                models_dict[model_name] = CatBoostClassifier(
                    **best_params, boosting_type="Ordered",
                    auto_class_weights="Balanced",
                    random_seed=RANDOM_STATE, verbose=0
                )

        # ── CV değerlendirmesi ───────────────────────────────────────────────
        print(f"\n  CV değerlendirmesi ({len(models_dict)} model)...")
        for model_name, model in models_dict.items():
            print(f"  [{model_name}] eğitiliyor...", end=" ", flush=True)
            try:
                result, oof_probs = evaluate_with_cv(
                    model, prepared_folds, y, df, target, model_name
                )
                result["Target"] = target
                all_results.append(result)
                roi_txt = (f"{result['ROI_value']:+.4f}"
                           if result['ROI_value'] is not None else "N/A")
                print(f"AUC={result['AUC_mean']:.4f} | "
                      f"P@1={result['P@1']:.4f} | "
                      f"ROI_value={roi_txt}")

                # Tüm veriyle fit et (production/importance için). Ablation modunda
                # model kaydı/importance yapılmadığından bu adım atlanır (hız).
                if not ablation:
                    model.fit(X_full, y)
                    best_models[(model_name, target)] = model

            except Exception as e:
                print(f"HATA: {e}")

    # ── 5. Karşılaştırma tablosu ───────────────────────────────────────────────
    if all_results:
        report_df = pd.DataFrame(all_results)
        report_cols = ["Target", "Model", "AUC_mean", "AUC_std", "AP_mean",
                       "P@1", "P@3", "ROI_top1", "ROI_value", "ROI_kelly",
                       "Bets_value", "WinRate_top1"]
        report_df = report_df[[c for c in report_cols if c in report_df.columns]]
        report_df = report_df.sort_values(["Target", "AUC_mean"], ascending=[True, False])

        report_path = REPORT_DIR / f"model_comparison{suffix}.csv"
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 70)
        print("  MODEL KARŞILAŞTIRMA TABLOSU")
        print("=" * 70)
        print(report_df.to_string(index=False))
        print(f"\n  ✅ Rapor kaydedildi: {report_path}")

    # ── 6. Feature Importance + SHAP (en iyi model için) ──────────────────────
    # Ablation modunda atlanır (tam-model SHAP/importance PNG'lerini ezmemek için).
    if ablation:
        print("\n" + "█" * 70)
        print("  ABLATION TAMAMLANDI! (Ganyansız)")
        print(f"  Karşılaştırma: {REPORT_DIR / ('model_comparison' + suffix + '.csv')}")
        print("  (Production kaydı / SHAP / model .pkl ablation'da üretilmez.)")
        print("█" * 70 + "\n")
        return

    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE ANALİZİ")
    print("=" * 70)
    for target in TARGET_COLS:
        y_target = df[target]  # her hedef için doğru y (eski bug: son döngüden kalan y)
        # En iyi AUC'lu ağaç modelini seç (LR/Pipeline değil)
        tree_models = {k: v for k, v in best_models.items()
                       if k[1] == target and not isinstance(v, Pipeline)}
        if not tree_models:
            continue
        # Öncelik: LightGBM > XGBoost > CatBoost > RF
        for preferred in ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]:
            key = (preferred, target)
            if key in tree_models:
                analyze_feature_importance(
                    tree_models[key], X_full, y_target, feature_names, preferred, target
                )
                break

    # ── 7. Production model seçimi ve kayıt ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  PRODUCTION MODEL SEÇİMİ VE KAYIT")
    print("=" * 70)

    if all_results:
        full_report = pd.DataFrame(all_results)
        for target in TARGET_COLS:
            best_name = select_and_register_production_model(full_report, target)
            if best_name and (best_name, target) in best_models:
                model = best_models[(best_name, target)]
                target_results = [r for r in all_results
                                   if r["Model"] == best_name and r["Target"] == target]
                result_dict = target_results[0] if target_results else {}
                save_model(model, best_name, target, result_dict, feature_names)

    print("\n" + "█" * 70)
    print("  STAGE 4 TAMAMLANDI!")
    print(f"  Modeller: {MODEL_DIR}")
    print(f"  Raporlar: {REPORT_DIR}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
