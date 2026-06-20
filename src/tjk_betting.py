#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK Bahis Matematiği — saf kütüphane (I/O yok, yalnız numpy)
================================================================================
  Modelin per-at KAZANMA olasılıklarını (prob_winner) egzotik bahis
  kombinasyon olasılıklarına çevirir ve EV / Kelly hesaplar.

  Yöntem:
    • Harville / Plackett-Luce  → sıralı/sırasız ilk-k kombinasyon olasılığı
    • Favori-uzunatış kalibrasyonu (Lo-Bacon-Shor üssü λ)
    • Piyasa-ima ödeme: payout ≈ (1 − takeout) / P_market(kombinasyon)
    • Kelly kasa yönetimi (fractional)

  Tüm olasılık fonksiyonları "win-probability vektörü" alır (koşu-içi, toplamı
  ≈ 1). İndeksler 0-tabanlıdır; çağıran taraf at sırasına eşler.

  Bahis türü ↔ fonksiyon eşlemesi (TJK):
    Ganyan          → p[i]                         (kazanan)
    Plase           → place_prob / dışarıdan prob_top3
    İkili           → quinella_prob  (sırasız ilk-2)
    Sıralı İkili    → exacta_prob    (sıralı ilk-2)
    Üçlü Bahis      → ordered_topk_prob k=3        (sıralı ilk-3)
    Üçlü Sırasız    → unordered_topk_prob k=3
    Tabela          → ordered_topk_prob k=4        (sıralı ilk-4)
    Tabela Sırasız  → unordered_topk_prob k=4
    Çifte / Pick-N  → pickn_ticket_prob            (çoklu-koşu)
================================================================================
"""
from itertools import permutations
import numpy as np

EPSILON = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
#  OLASILIK HAZIRLIĞI
# ──────────────────────────────────────────────────────────────────────────────
def normalize(p):
    """Negatifleri kırp, toplamı 1'e normalize et."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= EPSILON:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def market_probs(odds):
    """
    Ganyan oranlarından (decimal, ör. 2.60 = 1 birime 2.60 dönüş) piyasanın ima
    ettiği kazanma olasılıkları. Overround (fazla yük / takeout) normalize edilir:
        p_i = (1/oran_i) / Σ(1/oran_j)
    Geçersiz/eksik oran → o atın payı 0 (sonra normalize).
    """
    odds = np.asarray(odds, dtype=float)
    inv = np.where(np.isfinite(odds) & (odds > EPSILON), 1.0 / odds, 0.0)
    return normalize(inv)


def calibrate(p, lam=1.0):
    """
    Favori-uzunatış kalibrasyonu (Lo-Bacon-Shor):  p_i^λ / Σ p_j^λ.
    λ<1 dağılımı düzleştirir (Harville'in favorileri abartmasını törpüler).
    λ=1 → değişiklik yok.
    """
    p = normalize(p)
    if abs(lam - 1.0) < EPSILON:
        return p
    pw = np.power(p, lam)
    return normalize(pw)


# ──────────────────────────────────────────────────────────────────────────────
#  HARVILLE / PLACKETT-LUCE
# ──────────────────────────────────────────────────────────────────────────────
def ordered_topk_prob(p, seq):
    """
    Belirli bir SIRALI bitiş dizisinin olasılığı (Plackett-Luce):
        P(seq) = Π_i  p[seq_i] / (1 − Σ_{j<i} p[seq_j])
    seq: at indeksleri (ör. exacta için (i, j); trifecta için (i, j, k)).
    """
    p = np.asarray(p, dtype=float)
    prob = 1.0
    used = 0.0
    for idx in seq:
        denom = 1.0 - used
        if denom <= EPSILON:
            return 0.0
        prob *= p[idx] / denom
        used += p[idx]
    return float(prob)


def unordered_topk_prob(p, idx_set):
    """
    Bir at KÜMESİNİN (sırasız) ilk-k'yı oluşturma olasılığı: tüm sıralamaların
    Harville toplamı. k küçük (≤4) olduğundan permütasyon sayımı ucuz.
    """
    return float(sum(ordered_topk_prob(p, perm) for perm in permutations(idx_set)))


def exacta_prob(p, i, j):
    """Sıralı İkili: i 1., j 2."""
    return ordered_topk_prob(p, (i, j))


def quinella_prob(p, i, j):
    """İkili (sırasız ilk-2): {i, j} herhangi sırada."""
    return exacta_prob(p, i, j) + exacta_prob(p, j, i)


def place_prob(p, i, n_places=3):
    """
    Plase ≈ atın ilk n_places içinde bitirme olasılığı (Harville ile).
    Not: Mümkünse modelin doğrudan prob_top3 değeri tercih edilir; bu, yalnız
    win-vektöründen türetilen yaklaşık alternatiftir.
    """
    others = [x for x in range(len(p)) if x != i]
    total = 0.0
    # i'nin 1..n_places. sırada bitmesi: kalanların önceki sıraları doldurması
    def _rec(prefix):
        if len(prefix) == n_places:
            return 0.0
        acc = ordered_topk_prob(p, prefix + [i])  # i tam bu sırada biter
        for o in others:
            if o in prefix:
                continue
            acc += _rec(prefix + [o])
        return acc
    total = _rec([])
    return float(min(total, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
#  ÇOKLU-KOŞU (Çifte / Pick-N)
# ──────────────────────────────────────────────────────────────────────────────
def pickn_ticket_prob(leg_selected_probs):
    """
    Pick-N bileti tutma olasılığı: her bacakta seçili atların kazanma
    olasılıklarının TOPLAMI, bacaklar arası ÇARPIM.
        leg_selected_probs = [[p,..], [p,..], ...]  (bacak başına seçili atlar)
    """
    prob = 1.0
    for sel in leg_selected_probs:
        prob *= float(np.sum(sel))
    return float(prob)


def pickn_ticket_cost(leg_sizes, unit=1.0):
    """Pick-N bilet maliyeti: Π |seçili bacak| × birim pay."""
    cost = unit
    for s in leg_sizes:
        cost *= int(s)
    return float(cost)


# ──────────────────────────────────────────────────────────────────────────────
#  ÖDEME / EV / KELLY
# ──────────────────────────────────────────────────────────────────────────────
def implied_payout(p_market, takeout=0.20):
    """
    Piyasa-ima brüt ödeme katsayısı (pay dahil dönüş):
        payout ≈ (1 − takeout) / P_market(kombinasyon)
    p_market çok küçükse büyük ama sonlu değer döner.
    """
    p_market = max(float(p_market), EPSILON)
    return (1.0 - takeout) / p_market


def ev(p_model, payout, stake=1.0):
    """Beklenen değer (net): stake·(p_model·payout − 1)."""
    return float(stake * (p_model * payout - 1.0))


def ev_ratio(p_model, payout):
    """EV / stake = p_model·payout − 1. >0 ise pozitif-EV (value)."""
    return float(p_model * payout - 1.0)


def kelly_fraction(p, b):
    """
    Kelly oranı: f* = (p·b − (1−p)) / b = p − (1−p)/b
    p: tutma olasılığı, b: NET oran (payout − 1). [0,1]'e kırpılır.
    """
    if b <= EPSILON:
        return 0.0
    f = p - (1.0 - p) / b
    return float(min(max(f, 0.0), 1.0))


def fractional_kelly(p, b, frac=0.25, cap=0.05):
    """
    Kademeli Kelly: frac·f*, üst sınır cap (kasanın oranı olarak).
    Varyansı düşürür; cap tek bahiste aşırı riski engeller.
    """
    return float(min(frac * kelly_fraction(p, b), cap))


# ──────────────────────────────────────────────────────────────────────────────
#  ÖZ-TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("== tjk_betting öz-test ==")
    ok = True

    def check(name, cond):
        global ok
        ok = ok and cond
        print(f"  [{'✓' if cond else '✗'}] {name}")

    # 4 atlı koşu
    p = normalize([0.5, 0.3, 0.15, 0.05])

    # 1) ordered top-1 == p
    check("ordered_topk k=1 == p", abs(ordered_topk_prob(p, (0,)) - p[0]) < 1e-9)

    # 2) tüm sıralı ikili olasılıkları toplamı ≈ 1
    s_ex = sum(exacta_prob(p, i, j) for i in range(4) for j in range(4) if i != j)
    check("Σ exacta == 1", abs(s_ex - 1.0) < 1e-9)

    # 3) tüm sıralı üçlü toplamı ≈ 1
    s_tri = sum(ordered_topk_prob(p, perm) for perm in permutations(range(4), 3))
    check("Σ ordered top3 == 1", abs(s_tri - 1.0) < 1e-9)

    # 4) quinella ≥ exacta (sırasız ⊇ sıralı)
    check("quinella >= exacta", quinella_prob(p, 0, 1) >= exacta_prob(p, 0, 1) - 1e-12)

    # 5) sırasız üçlü kümeleri toplamı ≈ 1 (her 3'lü küme bir kez)
    from itertools import combinations
    s_trio = sum(unordered_topk_prob(p, c) for c in combinations(range(4), 3))
    check("Σ unordered top3 sets == 1", abs(s_trio - 1.0) < 1e-9)

    # 6) market_probs: overround temizlenir, toplam 1
    mp = market_probs([2.0, 4.0, 5.0, 10.0])  # Σ(1/o)=0.85 → normalize
    check("market_probs sum==1", abs(mp.sum() - 1.0) < 1e-9)
    check("market_probs favori en yüksek", mp[0] == mp.max())

    # 7) kelly: adil oranda (p*payout=1) f*≈0; edge varsa f*>0
    check("kelly fair ~0", abs(kelly_fraction(0.5, 1.0) - 0.0) < 1e-9)   # p=0.5,b=1 → 0
    check("kelly edge >0", kelly_fraction(0.6, 1.0) > 0)                  # p=0.6,b=1 → 0.2

    # 8) implied_payout monoton: küçük olasılık → büyük ödeme
    check("payout monoton", implied_payout(0.1, 0.2) > implied_payout(0.5, 0.2))

    # 9) ev_ratio işareti
    check("ev_ratio +", ev_ratio(0.5, 2.5) > 0)
    check("ev_ratio -", ev_ratio(0.5, 1.5) < 0)

    # 10) calibrate λ<1 favoriyi törpüler
    pc = calibrate(p, lam=0.8)
    check("calibrate sum==1", abs(pc.sum() - 1.0) < 1e-9)
    check("calibrate favori azalır", pc[0] < p[0])

    # 11) pickn: 2 bacak, tek at
    check("pickn 2 leg", abs(pickn_ticket_prob([[0.5], [0.4]]) - 0.20) < 1e-9)
    check("pickn cost 2x3", pickn_ticket_cost([2, 3], unit=1.0) == 6.0)

    print("\nSONUÇ:", "TÜM TESTLER GEÇTİ ✓" if ok else "BAŞARISIZ ✗")
    import sys
    sys.exit(0 if ok else 1)
