import time
import random
import re
import os
import uuid
from typing import Any, Dict, List, Optional
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ─── Sabitler ──────────────────────────────────────────────────────────────────
BASE_URL  = "https://www.tjk.org"
# Veri dosyaları proje kökündeki data/ klasöründe (src/ -> .. -> data)
BASE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_CSV  = os.path.join(BASE_DIR, "yaris_ana_tablo.csv")
STATIC_CSV = os.path.join(BASE_DIR, "atlar_statik_tablo.csv")
IDMAN_CSV  = os.path.join(BASE_DIR, "idmanlar_tablo.csv")

STATIC_COLS = ["at_id", "At_Adi", "Dogum_Tarihi", "Handikap_Puani",
               "Baba", "Anne", "Sahip", "Antrenor"]

IDMAN_COLS  = ["idman_id", "at_id", "Idman_Tarihi", "Hipodrom",
               "Pist_Durumu", "Idman_Turu", "Idman_Jokeyi",
               "Derece_400m", "Derece_600m", "Derece_800m",
               "Derece_1000m", "Derece_1200m"]

# Statik bilgi tablosundaki Türkçe etiket → sütun adı eşlemesi
LABEL_MAP = {
    "baba"           : "Baba",
    "anne"           : "Anne",
    "handikap puanı" : "Handikap_Puani",
    "handikap p"     : "Handikap_Puani",
    "doğum tarihi"   : "Dogum_Tarihi",
    "doğ. trh"       : "Dogum_Tarihi",
    "antrenör"       : "Antrenor",
    "gerçek sahip"   : "Sahip",
    "antreno"        : "Antrenor",
    "sahip"          : "Sahip",
}


# ─── Driver ────────────────────────────────────────────────────────────────────
def setup_driver():
    """Selenium Chrome WebDriver'ı headless modda başlatır."""
    options = Options()
    options.add_argument("--headless") # Arkaplanda çalışma modülü
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


# ─── Tekilleştirme + Resume ────────────────────────────────────────────────────
def get_unique_horses(input_csv: str, static_csv: str) -> list:
    """
    yaris_ana_tablo.csv'den benzersiz atları okur.
    atlar_statik_tablo.csv varsa daha önce işlenmiş at_id'leri atlar (resume).
    """
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df = df.dropna(subset=["At_URL"])
    df = df.drop_duplicates(subset=["At_URL"])

    # Regex ile At_Id ekstraksiyonu (negatif ID'ler dahil — yabancı atlar)
    def extract_id(url):
        m = re.search(r"QueryParameter_AtId=(-?\d+)", str(url))
        return m.group(1) if m else None

    df["at_id"] = df["At_URL"].apply(extract_id)
    df = df.dropna(subset=["at_id"])

    all_horses = df[["at_id", "At_Adi", "At_URL"]].rename(
        columns={"At_URL": "url"}
    ).to_dict("records")

    # Resume mantığı: işlenmiş atları bul ve es geç
    processed_ids = set()
    if os.path.isfile(static_csv):
        try:
            df_done = pd.read_csv(static_csv, encoding="utf-8-sig", usecols=["at_id"])
            processed_ids = set(df_done["at_id"].astype(str).tolist())
        except Exception as e:
            print(f"[UYARI] {static_csv} okunurken hata (resume atlandı): {e}")

    remaining = [h for h in all_horses if str(h["at_id"]) not in processed_ids]

    print(f"Toplam benzersiz at: {len(all_horses)}")
    print(f"Daha önce işlenmiş (skip): {len(processed_ids)}")
    print(f"İşlenecek at sayısı: {len(remaining)}")
    return remaining


# ─── Statik Profil (Doğum Tarihi, Anne, Baba vs.) ──────────────────────────────
def scrape_static_info(driver, at_id: str, at_adi: str, horse_url: str) -> dict:
    result: dict = {col: None for col in STATIC_COLS}
    result["at_id"] = at_id
    result["At_Adi"] = at_adi

    try:
        # Manuel string manipülasyonu YOK -> Orijinal URL direkt geçilir
        driver.get(horse_url)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(random.uniform(1.0, 2.0))

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # HTML'de Tablo Mantığıyla Veri Arama
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                label_text = cells[0].get_text(strip=True).lower()
                value_text = cells[1].get_text(strip=True)

                for key, col_name in LABEL_MAP.items():
                    if key in label_text:
                        result[col_name] = value_text if value_text else None
                        break

        # Sitenin yedek HTML yapısı için (Tablo olmaz da Span olursa) fallback
        if all(result[c] is None for c in ["Baba", "Anne", "Dogum_Tarihi"]):
            keys = soup.find_all("span", class_=re.compile(r"key|label", re.I))
            for label in keys:
                value = label.find_next_sibling("span", class_=re.compile(r"value|val", re.I))
                if value:
                    label_text = label.get_text(strip=True).lower()
                    value_text = value.get_text(strip=True)
                    for key, col_name in LABEL_MAP.items():
                        if key in label_text:
                            result[col_name] = value_text if value_text else None
                            break
    except Exception as e:
        print(f"  [HATA] {at_adi} (at_id={at_id}) statik veri çekilirken sorun: {e}")

    return result


# ─── İdman Verileri (Galop / Sprint) ───────────────────────────────────────────
def scrape_training_info(driver, at_id: str, horse_url: str) -> list:
    rows_out = []

    distance_col_map = {
        "400"  : "Derece_400m",
        "600"  : "Derece_600m",
        "800"  : "Derece_800m",
        "1000" : "Derece_1000m",
        "1200" : "Derece_1200m",
    }

    try:
        # Yabancı atlar (negatif at_id) için idman verisi TJK'da bulunmaz
        if str(at_id).startswith("-"):
            return rows_out

        # URL manipülasyonu yok -> Sitedeki butonu / sekmeyi bul
        idman_link = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, 'IdmanIstatistikleri') and contains(@href, 'QueryParameter_AtId=')]"))
        )
        idman_href = idman_link.get_attribute("href")
        
        # Eğer href özelliği varsa url'e git, yoksa DOM üzerinden JavaScript click yapıp bekle
        if idman_href and "javascript" not in idman_href.lower():
            driver.get(idman_href)
        else:
            driver.execute_script("arguments[0].click();", idman_link)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(random.uniform(1.5, 3.0))

        soup = BeautifulSoup(driver.page_source, "html.parser")

        table = soup.find("table", class_=re.compile(r"tablesorter|table", re.I))
        if not table:
            table = soup.find("table")

        if not table:
            return rows_out

        header_row = table.find("thead")
        if not header_row:
            header_row = table.find("tr")

        headers = []
        if header_row:
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))

        col_idx: Dict[str, int] = {}
        dist_idx: Dict[str, int] = {}

        keyword_map: Dict[str, List[str]] = {
            "Idman_Tarihi" : ["tarih", "date"],
            "Hipodrom"     : ["hipodrom", "hipo", "pist yeri", "yer"],
            "Pist_Durumu"  : ["pist dur", "zemin", "durum"],
            "Idman_Turu"   : ["idman türü", "tür", "galop", "sprint", "type"],
            "Idman_Jokeyi" : ["jokey", "binici", "jockey"],
        }

        # Tablonun kolon sıralamaları değişse dahi dinamik eşleştir
        for i, h in enumerate(headers):
            h_lower = h.lower()
            for col_name, keywords in keyword_map.items():
                if any(kw in h_lower for kw in keywords):
                    if col_name not in col_idx:
                        col_idx[col_name] = i
                    break
            for dist in distance_col_map:
                if dist in re.sub(r"\s", "", h):
                    dist_idx[dist] = i

        tbody = table.find("tbody")
        body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

        for tr in body_rows:
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue

            def cell_val(idx):
                if idx is None or idx >= len(cells):
                    return None
                val = cells[idx].get_text(strip=True)
                return val if val else None

            row = {col: None for col in IDMAN_COLS}
            row["idman_id"]     = str(uuid.uuid4())
            row["at_id"]        = at_id
            row["Idman_Tarihi"] = cell_val(col_idx.get("Idman_Tarihi"))
            row["Hipodrom"]     = cell_val(col_idx.get("Hipodrom"))
            row["Pist_Durumu"]  = cell_val(col_idx.get("Pist_Durumu"))
            row["Idman_Turu"]   = cell_val(col_idx.get("Idman_Turu"))
            row["Idman_Jokeyi"] = cell_val(col_idx.get("Idman_Jokeyi"))

            for dist, mapped_col in distance_col_map.items():
                if dist in dist_idx:
                    row[mapped_col] = cell_val(dist_idx[dist])

            data_fields = [row[c] for c in IDMAN_COLS if c not in ("idman_id", "at_id")]
            if any(v is not None for v in data_fields):
                rows_out.append(row)

    except TimeoutException:
        print(f"  [BİLGİ] at_id={at_id} — İdman İstatistikleri sekmesi bulunamadı (veri yok).")
    except Exception as e:
        # Stacktrace'i kısalt — sadece ilk satırı göster
        err_msg = str(e).split('\n')[0][:120]
        print(f"  [HATA] at_id={at_id} idman verisi çekilirken sorun: {err_msg}")

    return rows_out


# ─── CSV Yardımcısı ────────────────────────────────────────────────────────────
def append_to_csv(rows: list, filename: str, columns: list):
    """
    Verilen satır listesini belirlenen CSV dosyasına append modunda yazar.
    Duplicate kontrolü: at_id bazlı tekrar yazmayı önler (resume güvenliği).
    """
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=columns)

    # Eğer dosya zaten varsa, at_id'ye göre mevcut kayıtlarla karşılaştır
    if os.path.isfile(filename) and "at_id" in columns:
        try:
            df_existing = pd.read_csv(filename, encoding="utf-8-sig", usecols=["at_id"])
            existing_ids = set(df_existing["at_id"].astype(str).tolist())
            before = len(df_new)
            df_new = df_new[~df_new["at_id"].astype(str).isin(existing_ids)]
            skipped = before - len(df_new)
            if skipped > 0:
                print(f"      [Duplicate kontrol] {skipped} kayıt zaten mevcut, atlandı.")
        except Exception:
            pass  # Okuma hatası olursa mevcut davranışa dön

    if df_new.empty:
        return

    file_exists = os.path.isfile(filename)
    df_new.to_csv(filename, mode="a", index=False,
                  header=not file_exists, encoding="utf-8-sig")


# ─── Ana Akış ──────────────────────────────────────────────────────────────────
def main():
    horses = get_unique_horses(INPUT_CSV, STATIC_CSV)
    if not horses:
        print("İşlenecek yeni at yok (Veya input csv okunmuyor). Çıkılıyor.")
        return

    driver = setup_driver()
    print(f"\nSelenium başlatıldı. {len(horses)} kayıt işlenecek.\n")

    try:
        for i, horse in enumerate(horses, start=1):
            at_id  = horse["at_id"]
            at_adi = horse.get("At_Adi", "Bilinmiyor")
            url    = horse["url"]

            print(f"[{i}/{len(horses)}] {at_adi} (at_id={at_id}) işleniyor...")

            # Tarayıcı çökme kontrolü
            try:
                _ = driver.title
            except Exception:
                print("  [!] Tarayıcı çöktü, yeniden başlatılıyor...")
                try: driver.quit()
                except: pass
                driver = setup_driver()

            # 1. Statik Kısım Extract
            static_data = scrape_static_info(driver, at_id, at_adi, url)
            append_to_csv([static_data], STATIC_CSV, STATIC_COLS)

            # 2. İdman Kısmı Extract (yabancı atları atla)
            if str(at_id).startswith("-"):
                print(f"  ⏭ Yabancı at (at_id={at_id}), idman verisi atlanıyor.")
                training_rows = []
            else:
                training_rows = scrape_training_info(driver, at_id, url)
                append_to_csv(training_rows, IDMAN_CSV, IDMAN_COLS)

            print(f"  ✓ {at_adi} Tamamlandı — Bulunan idman kaydı sayısı: {len(training_rows)}")

            # Bot/IP Ban koruması için zaman periyodu
            time.sleep(random.uniform(1.5, 3.5))

    except KeyboardInterrupt:
        print("\n[!] Kullanıcı tarafından manuel durduruldu. Sonraki çalışmada kaldığı yerden devam edecektir.")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print(f"\nStage 2 veri toplama tamamlandı. Kayıtlar -> {STATIC_CSV} & {IDMAN_CSV} adresinde.")


if __name__ == "__main__":
    main()
