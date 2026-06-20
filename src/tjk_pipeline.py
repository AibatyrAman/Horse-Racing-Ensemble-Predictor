#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK TAM PIPELINE ORCHESTRATOR — Stage 1 → Stage 2 → Stage 3 → Stage 4
================================================================================
  Tüm aşamaları sırasıyla çalıştırır.
  Her aşamada hata olursa durur ve kullanıcıyı bilgilendirir.
  Kullanım:
      python tjk_pipeline.py              → Tüm stage'leri çalıştır
      python tjk_pipeline.py --from 2     → Stage 2'den başla
      python tjk_pipeline.py --only 3     → Sadece Stage 3
      python tjk_pipeline.py --from 3     → Stage 3 + 4 çalıştır
================================================================================
"""

import sys
import os
import time
import importlib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/ (modüller burada)
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")  # proje kökü/data

STAGES = {
    1: {
        "name": "Yarış Sonuçları Scraping",
        "module": "tjk_scraper_stage1",
        "output": "yaris_ana_tablo.csv",
        "type": "selenium",
    },
    2: {
        "name": "At Statik & İdman Verileri Scraping",
        "module": "tjk_scraper_stage2",
        "output": "atlar_statik_tablo.csv + idmanlar_tablo.csv",
        "type": "selenium",
    },
    3: {
        "name": "Feature Engineering (Veri Birleştirme)",
        "module": "tjk_stage3_feature_engineering",
        "output": "master_feature_matrix.csv",
        "type": "pandas",
    },
    4: {
        "name": "ML Modelleme & Değerlendirme",
        "module": "tjk_stage4_modeling",
        "output": "models/ + reports/",
        "type": "ml",
    },
}


def print_banner():
    print("\n" + "█" * 70)
    print("  TJK PIPELINE ORCHESTRATOR")
    print("  Stage 1 (Scrape) → Stage 2 (Profil) → Stage 3 (Features) → Stage 4 (ML)")
    print("█" * 70)
    print(f"  Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dizin: {BASE_DIR}")
    print()


def check_prerequisites(stage_num):
    """Stage çalışmadan önce gerekli dosyaların varlığını kontrol et."""
    if stage_num == 2:
        path = os.path.join(DATA_DIR, "yaris_ana_tablo.csv")
        if not os.path.isfile(path):
            print(f"  ❌ Stage 2 için 'yaris_ana_tablo.csv' gerekli ama bulunamadı!")
            print(f"     Önce Stage 1'i çalıştırın.")
            return False
    elif stage_num == 3:
        required = ["yaris_ana_tablo.csv", "atlar_statik_tablo.csv", "idmanlar_tablo.csv"]
        for f in required:
            path = os.path.join(DATA_DIR, f)
            if not os.path.isfile(path):
                print(f"  ❌ Stage 3 için '{f}' gerekli ama bulunamadı!")
                print(f"     Önce Stage 1 ve 2'yi çalıştırın.")
                return False
    elif stage_num == 4:
        path = os.path.join(DATA_DIR, "master_feature_matrix.csv")
        if not os.path.isfile(path):
            print(f"  ❌ Stage 4 için 'master_feature_matrix.csv' gerekli ama bulunamadı!")
            print(f"     Önce Stage 3'ü çalıştırın.")
            return False
    return True


def run_stage(stage_num):
    """Belirtilen stage'i çalıştır."""
    stage = STAGES[stage_num]
    
    print("\n" + "=" * 70)
    print(f"  STAGE {stage_num}: {stage['name']}")
    print(f"  Modül: {stage['module']}.py")
    print(f"  Çıktı: {stage['output']}")
    print(f"  Tip:   {stage['type']}")
    print("=" * 70)
    
    # Dosya kontrolü
    if not check_prerequisites(stage_num):
        return False
    
    # Uyarılar
    if stage['type'] == 'selenium':
        print(f"\n  ⚠ Bu aşama Selenium (tarayıcı) kullanıyor — uzun sürebilir.")
        print(f"    Ctrl+C ile güvenle durdurabilirsiniz (resume ile kaldığı yerden devam eder).")
    
    start_time = time.time()
    
    try:
        module = importlib.import_module(stage['module'])
        # Her modülün main() fonksiyonu var
        module.main()
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n  ✅ Stage {stage_num} tamamlandı! (Süre: {minutes}dk {seconds}sn)")
        return True
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n  ⏸ Stage {stage_num} kullanıcı tarafından durduruldu ({int(elapsed)}sn).")
        print(f"    Tekrar çalıştırıldığında kaldığı yerden devam edecektir.")
        return False
        
    except Exception as e:
        print(f"\n  ❌ Stage {stage_num} hatayla sonlandı: {e}")
        return False


def main():
    print_banner()
    
    # Argüman parse
    start_from = 1
    only_stage = None
    
    args = sys.argv[1:]
    if "--from" in args:
        idx = args.index("--from")
        if idx + 1 < len(args):
            start_from = int(args[idx + 1])
    if "--only" in args:
        idx = args.index("--only")
        if idx + 1 < len(args):
            only_stage = int(args[idx + 1])
    
    # Çalıştırılacak stage'leri belirle
    if only_stage:
        stages_to_run = [only_stage]
        print(f"  Mod: Sadece Stage {only_stage}")
    else:
        stages_to_run = [s for s in sorted(STAGES.keys()) if s >= start_from]
        if start_from > 1:
            print(f"  Mod: Stage {start_from}'den başla → Stage 4'e kadar")
        else:
            print(f"  Mod: Tam pipeline (Stage 1 → 4)")
    
    print(f"  Çalıştırılacak stage'ler: {stages_to_run}")
    
    # Mevcut veri durumu
    print("\n  Mevcut dosyalar:")
    files_to_check = [
        "yaris_ana_tablo.csv",
        "atlar_statik_tablo.csv",
        "idmanlar_tablo.csv",
        "master_feature_matrix.csv",
    ]
    for f in files_to_check:
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    ✓ {f:.<45} {size_mb:.2f} MB")
        else:
            print(f"    ✗ {f:.<45} bulunamadı")
    
    # Stage'leri sırasıyla çalıştır
    pipeline_start = time.time()
    results = {}
    
    for stage_num in stages_to_run:
        success = run_stage(stage_num)
        results[stage_num] = success
        
        if not success:
            print(f"\n  ⛔ Stage {stage_num} başarısız — pipeline durduruldu.")
            print(f"     Düzeltme sonrası: python tjk_pipeline.py --from {stage_num}")
            break
    
    # Özet
    pipeline_elapsed = time.time() - pipeline_start
    pipeline_min = int(pipeline_elapsed // 60)
    pipeline_sec = int(pipeline_elapsed % 60)
    
    print("\n" + "█" * 70)
    print("  PIPELINE SONUÇ ÖZETİ")
    print("█" * 70)
    
    for stage_num, success in results.items():
        status = "✅ Başarılı" if success else "❌ Başarısız"
        print(f"    Stage {stage_num}: {STAGES[stage_num]['name']:.<45} {status}")
    
    not_run = [s for s in stages_to_run if s not in results]
    for stage_num in not_run:
        print(f"    Stage {stage_num}: {STAGES[stage_num]['name']:.<45} ⏭ Atlandı")
    
    print(f"\n  Toplam süre: {pipeline_min}dk {pipeline_sec}sn")
    
    if all(results.values()):
        print("\n  🏇 TÜM PIPELINE BAŞARIYLA TAMAMLANDI!")
    
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
