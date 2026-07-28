import time
import random
import re
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from urllib.parse import urljoin

BASE_URL = "https://www.tjk.org"
BASE_URL_FOR_JOIN = "https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari"

# CSV'ler proje kökündeki data/ klasörüne yazılır (src/ -> .. -> data)
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def setup_driver():
    """
    Selenium WebDriver'ı başlatır.
    """
    options = Options()
    # Headless: ortam değişkeni TJK_HEADLESS=1 ise (sunucu/arka plan çalıştırması)
    if os.environ.get("TJK_HEADLESS") == "1":
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("window-size=1920,1080")
    # TJK gibi siteler otomasyonu engelliyorsa, bazı anti-bot ayarları da eklenebilir.
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

    # Sistemdeki Chromium/Chromedriver'ı kullan (CHROME_BIN/CHROMEDRIVER_PATH).
    # ARM64 Linux'ta Selenium Manager sürücü indiremiyor → Service ile sabitle.
    chrome_bin = os.environ.get("CHROME_BIN")
    if chrome_bin:
        options.binary_location = chrome_bin
    driver_path = os.environ.get("CHROMEDRIVER_PATH")
    if driver_path:
        from selenium.webdriver.chrome.service import Service
        driver = webdriver.Chrome(service=Service(driver_path), options=options)
    else:
        driver = webdriver.Chrome(options=options)
    return driver

def get_daily_cities(driver):
    """
    Mevcut DOM üzerinden yarış yapılan şehirlerin ID ve isimlerini döndürür.
    (Artık URL yönlendirmesi yapmaz, sadece parse eder)
    """
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    ul_tabs = soup.find('ul', class_='gunluk-tabs')
    if not ul_tabs:
        return []
        
    cities = []
    for li in ul_tabs.find_all('li'):
        a_tag = li.find('a', attrs={'data-sehir-id': True})
        if a_tag:
            city_id = a_tag['data-sehir-id']
            city_name = a_tag.text.strip()
            cities.append({'id': city_id, 'name': city_name})
            
    return cities

def extract_rank(title_str):
    """
    Title attribute'u içinden (Örn: "GÖKSAGUN bu koşuyu 5. olarak bitirmiştir.") atın sırasını çıkarır.
    """
    if not title_str:
        return "Derecesiz"
    
    match = re.search(r'(\d+)\.', title_str)
    if match:
        return match.group(1)
    return "Derecesiz"

# Yarış türü ana etiketleri (race-details başlığındaki ilk kelime)
YARIS_TURU_KEYWORDS = [
    "ŞARTLI", "MAIDEN", "HANDİKAP", "HANDIKAP", "SATIŞ", "AÇIK",
    "KV", "G1", "G2", "G3", "ŞAMPİYONA", "DENEME", "KOŞUSU",
]


def _parse_agf(raw):
    """AGF ham metnini (ör. '%11(4)') → (oran, sira) çift.
    oran = para payı 0..1 (0.11), sira = para favoriliği sırası (4).
    Eşleşmezse (None, None)."""
    if not raw:
        return None, None
    m = re.search(r"%?\s*(\d+(?:[.,]\d+)?)\s*(?:\(\s*(\d+)\s*\))?", str(raw))
    if not m:
        return None, None
    oran = float(m.group(1).replace(",", ".")) / 100.0
    sira = int(m.group(2)) if m.group(2) else None
    return oran, sira


def _parse_fark(raw):
    """Mağlubiyet farkı metnini (ör. '1,5 Boy', '4,5 Boy', 'Baş', 'Burun') →
    boy cinsinden sayı. Kısa kafa/baş/burun mesafeleri ~0.1 boy sayılır.
    Kazanan/eşleşmeyen → None."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    for kw, val in (("burun", 0.05), ("kısa baş", 0.1), ("baş", 0.2), ("boyun", 0.25),
                    ("kafa", 0.1)):
        if kw in s:
            return val
    m = re.search(r"(\d+(?:[.,]\d+)?)", s)
    return float(m.group(1).replace(",", ".")) if m else None


def parse_race_details(div):
    """
    Koşu div'inin ÖNCESİNDEKİ `div.race-details` kardeşinden yarış-seviyesi
    meta bilgileri çıkarır. Örnek metin (canlıda doğrulandı, 20.06.2026):
      '1. Koşu 16.15 ŞARTLI 5/DHT/Y2 , 4 ve Yukarı Araplar,\\n58 kg, 60 kg,\\n1900\\nKum , E.İ.D. : 2.10.52'
    Ayrıca `div.race-share` kardeşinden birincilik ikramiyesi alınır.
    Döndürür: dict(Kosu_Saati, Yaris_Turu, Yaris_Turu_Detay, Mesafe, Ikramiye_1)
    """
    out = {"Kosu_Saati": None, "Yaris_Turu": None, "Yaris_Turu_Detay": None,
           "Mesafe": None, "Ikramiye_1": None}
    try:
        details = div.find_previous_sibling("div", class_="race-details")
        if details is not None:
            text = details.get_text(" ", strip=True)
            # Saat: "1. Koşu 16.15 ..."
            m = re.search(r"Koşu\s*([01]?\d|2[0-3])[.:]([0-5]\d)", text)
            if m:
                out["Kosu_Saati"] = f"{int(m.group(1)):02d}:{m.group(2)}"
            # Tür bloğu: saatten ilk virgüle kadar (örn. "ŞARTLI 5/DHT/Y2")
            ilk_blok = text.split(",")[0]
            if m:
                tur_detay = ilk_blok[m.end():].strip()
                if tur_detay:
                    out["Yaris_Turu_Detay"] = tur_detay
                    ana = tur_detay.split()[0].upper()
                    out["Yaris_Turu"] = ana if any(
                        ana.startswith(k) for k in YARIS_TURU_KEYWORDS) else ana
            # Mesafe: kg takip etmeyen 800-3600 arası sayı
            for mm in re.finditer(r"\b(\d{3,4})\b(?!\s*kg)", text):
                val = int(mm.group(1))
                if 800 <= val <= 3600:
                    out["Mesafe"] = val
                    break
        share = div.find_previous_sibling("div", class_="race-share")
        if share is not None:
            m = re.search(r"1\.\)\s*([\d.]+)", share.get_text(" ", strip=True))
            if m:
                out["Ikramiye_1"] = m.group(1).replace(".", "")
    except Exception:
        pass
    return out


def parse_race_table(html_source, date_str, city_name):
    """
    Yüklenmiş DOM üzerinden tüm at yarış tablolarını parse eder.
    """
    soup = BeautifulSoup(html_source, 'html.parser')
    data_list = []

    # Sayfadaki tüm koşu div'lerini bul ('kosubilgisi-XYZ' şeklinde ID'leri var. Yurtdışı yarışları eksi değerli olabiliyor)
    race_divs = soup.find_all('div', id=re.compile(r'^kosubilgisi--?\d+'))

    # Pist Durumu bilgisini al
    track_condition = "Bilinmiyor"
    weather_span = soup.find('span', class_='raceWeatherBrown')
    if weather_span:
        track_condition = weather_span.text.strip()

    for div in race_divs:
        div_id = div.get('id', '')
        race_id = div_id.replace('kosubilgisi-', '')

        # Yarış-seviyesi meta (tür, mesafe, saat, ikramiye) — her satıra kopyalanır
        race_meta = parse_race_details(div)

        table = div.find('table', class_='tablesorter')
        if not table:
            continue
            
        tbody = table.find('tbody')
        if not tbody:
            continue
            
        rows = tbody.find_all('tr')
        for row in rows:
            horse_data = {
                'Tarih': date_str,
                'Sehir': city_name,
                'Pist_Durumu': track_condition,
                'Kosu_ID': race_id,
                'Siralama': "Derecesiz",
                'At_Adi': None,
                'Yas': None,
                'Siklet': None,
                'Start': None,
                'At_URL': None,
                'Jokey_Adi': None,
                'Jokey_URL': None,
                'Antrenor_Adi': None,
                'Antrenor_URL': None,
                'Ganyan': None,
                'Derece': None,
                # Yarış-seviyesi meta (race-details / race-share'den)
                'Kosu_Saati': race_meta.get('Kosu_Saati'),
                'Yaris_Turu': race_meta.get('Yaris_Turu'),
                'Yaris_Turu_Detay': race_meta.get('Yaris_Turu_Detay'),
                'Mesafe': race_meta.get('Mesafe'),
                'Ikramiye_1': race_meta.get('Ikramiye_1'),
                # Satır-seviyesi ek alanlar (Fark=geriden geliş, AGF=oynanma payı,
                # Gec_Cikis, HP=o günkü resmi handikap puanı)
                'Fark': None,
                'AGF': None,
                'Gec_Cikis': None,
                'HP': None,
                # Ayrıştırılmış sayısal alanlar (modelin doğrudan kullanacağı):
                #   AGF_Oran = halkın para payı (0..1), AGF_Sira = para favoriliği sırası,
                #   Fark_Boy = mağlubiyet farkı (boy).
                'AGF_Oran': None,
                'AGF_Sira': None,
                'Fark_Boy': None,
            }
            
            # Sıralama
            try:
                td_sira = row.select_one('td[class*="-SONUCNO"]')
                if td_sira:
                    horse_data['Siralama'] = td_sira.text.strip()
            except Exception:
                pass

            # At Bilgileri
            try:
                td_at = row.select_one('td[class*="-AtAdi"]')
                if td_at:
                    a_at = td_at.find('a')
                    if a_at:
                        # At adının yanındaki numarayı (örn: "(7)") temizle
                        isim = re.sub(r'\(\d+\)', '', a_at.text).strip()
                        horse_data['At_Adi'] = isim
                        if a_at.has_attr('href'):
                            # URL Düzeltmesi
                            href_fixed = a_at['href'].replace('../../', '/TR/map/').split('&Era=')[0]
                            if not href_fixed.startswith('/'):
                                href_fixed = '/' + href_fixed
                            horse_data['At_URL'] = BASE_URL + href_fixed
                        
                        # Eğer sıralama yukarıda bulunamadıysa ve eski sayfadaysa fallback yap
                        if horse_data['Siralama'] == "Derecesiz":
                            span_sira = a_at.find('span', attrs={'title': True})
                            if span_sira:
                                horse_data['Siralama'] = extract_rank(span_sira['title'])
            except AttributeError:
                pass
            except Exception as e:
                print(f"      Hata: At bilgileri çekilirken sorun -> {e}")

            # Yaş Bilgisi
            try:
                td_yas = row.select_one('td[class*="-Yas"]')
                if td_yas:
                    horse_data['Yas'] = td_yas.text.strip()
            except Exception:
                pass

            # Sıklet (Kilo) Bilgisi
            try:
                td_siklet = row.select_one('td[class*="-Kilo"]')
                if td_siklet:
                    siklet_text = td_siklet.text.strip()
                    # Varsa "+" veya harfleri sadece sayı bölümünü alarak temizle. Örn "58" veya "58,5"
                    match = re.search(r'(\d+(?:[.,]\d+)?)', siklet_text)
                    if match:
                        horse_data['Siklet'] = match.group(1).replace(',', '.')
                    else:
                        horse_data['Siklet'] = siklet_text
            except Exception:
                pass
            
            # Start (Kulvar) Bilgisi
            try:
                td_start = row.select_one('td[class*="-StartId"], td[class*="-Kura"]')
                if td_start:
                    horse_data['Start'] = td_start.text.strip()
            except Exception:
                pass

            # Jokey Bilgileri
            try:
                td_jokey = row.select_one('td[class*="-JokeAdi"]')
                if td_jokey:
                    a_jokey = td_jokey.find('a')
                    if a_jokey:
                        horse_data['Jokey_Adi'] = a_jokey.text.strip()
                        if a_jokey.has_attr('href'):
                            href_fixed = a_jokey['href'].replace('../../', '/TR/map/').split('&Era=')[0]
                            if not href_fixed.startswith('/'):
                                href_fixed = '/' + href_fixed
                            horse_data['Jokey_URL'] = BASE_URL + href_fixed
            except AttributeError:
                pass
            except Exception as e:
                print(f"      Hata: Jokey bilgisi çekilirken sorun ({horse_data.get('At_Adi', 'Bilinmeyen At')}) -> {e}")
            
            # Antrenör Bilgileri
            try:
                td_antrenor = row.select_one('td[class*="-AntronorAdi"]')
                if td_antrenor:
                    a_antrenor = td_antrenor.find('a')
                    if a_antrenor:
                        horse_data['Antrenor_Adi'] = a_antrenor.text.strip()
                        if a_antrenor.has_attr('href'):
                            href_fixed = a_antrenor['href'].replace('../../', '/TR/map/').split('&Era=')[0]
                            if not href_fixed.startswith('/'):
                                href_fixed = '/' + href_fixed
                            horse_data['Antrenor_URL'] = BASE_URL + href_fixed
            except AttributeError:
                pass
            except Exception as e:
                print(f"      Hata: Antrenör bilgisi çekilirken sorun ({horse_data.get('At_Adi', 'Bilinmeyen At')}) -> {e}")
            
            # Ganyan
            try:
                td_ganyan = row.select_one('td[class*="-Gny"]')
                if td_ganyan:
                    span_ganyan = td_ganyan.find('span')
                    if span_ganyan:
                        horse_data['Ganyan'] = span_ganyan.text.replace(',', '.').strip()
                    else:
                        horse_data['Ganyan'] = td_ganyan.text.replace(',', '.').strip()
            except AttributeError:
                pass
            except Exception as e:
                print(f"      Hata: Ganyan bilgisi çekilirken sorun ({horse_data.get('At_Adi', 'Bilinmeyen At')}) -> {e}")
            
            # Derece
            try:
                # "DERECE" büyük-küçük harf durumu olabiliyor
                td_derece = row.select_one('td[class*="-Derece"], td[class*="-DERECE"]')
                if td_derece:
                    span_derece = td_derece.select_one('span#aciklamaFancyDrc')
                    if span_derece:
                        horse_data['Derece'] = span_derece.text.strip()
                    else:
                        horse_data['Derece'] = td_derece.text.strip()
            except AttributeError:
                pass
            except Exception as e:
                print(f"      Hata: Derece bilgisi çekilirken sorun ({horse_data.get('At_Adi', 'Bilinmeyen At')}) -> {e}")

            # Ek satır alanları: Fark / AGF / Geç Çıkış / HP (hataya dayanıklı)
            for key, sel in [("Fark", 'td[class*="-Fark"]'),
                             ("AGF", 'td[class*="-AGFORAN"]'),
                             ("Gec_Cikis", 'td[class*="-GecCikis"]'),
                             ("HP", 'td[class*="-Hc"]')]:
                try:
                    td = row.select_one(sel)
                    if td:
                        val = td.get_text(" ", strip=True)
                        horse_data[key] = val if val else None
                except Exception:
                    pass

            # AGF ve Fark'ı sayısala çevir (ham metin de korunur)
            horse_data["AGF_Oran"], horse_data["AGF_Sira"] = _parse_agf(horse_data["AGF"])
            horse_data["Fark_Boy"] = _parse_fark(horse_data["Fark"])

            data_list.append(horse_data)

    return data_list


# Egzotik/ikramiye bahis etiketleri (uzun → kısa; uzun olan önce eşleşmeli)
PAYOUT_LABELS = [
    "TABELA BAHİS SIRASIZ", "TABELA BAHİS", "SIRALI İKİLİ", "ÜÇLÜ BAHİS",
    "İKİLİ BAHİS", "SIRALI 5'Lİ BAHİS", "7'Lİ PLASE",
    "7'Lİ GANYAN", "6'LI GANYAN", "5'Lİ GANYAN", "4'LÜ GANYAN", "3'LÜ GANYAN",
    "1. ÇİFTE", "2. ÇİFTE", "GANYAN", "PLASE", "İKİLİ", "ÇİFTE",
]
# Tutar (TR biçimi: binlik '.', ondalık ',') ve kombinasyon (6/10/19 gibi) desenleri
_AMOUNT_RE = re.compile(r'(?:\d{1,3}(?:\.\d{3})+|\d+),\d{2}')
_COMBO_RE  = re.compile(r'\d+(?:/\d+)*')


def parse_payouts(html_source, date_str, city_name):
    """
    Sonuç sayfasından egzotik/ikramiye bahis ÖDEMELERİNİ çıkarır (forward-test'te
    GERÇEK ödeme skorlaması için). Her koşu div'i (kosubilgisi-*) içindeki ödeme
    bloğunda etiket → kombinasyon → tutar üçlülerini yakalar.

    DİKKAT: Seçiciler/biçim siteye göre değişebilir; metin-tabanlı, hataya dayanıklı
    (eşleşme yoksa boş döner). İlk canlı çalıştırmada gözlemlenip ayarlanmalı.
    Çıktı satırı: {Tarih, Sehir, Kosu_ID, Bahis, Kombinasyon, Tutar}
    """
    out = []
    try:
        soup = BeautifulSoup(html_source, 'html.parser')
        race_divs = soup.find_all('div', id=re.compile(r'^kosubilgisi--?\d+'))
        for div in race_divs:
            race_id = div.get('id', '').replace('kosubilgisi-', '')
            # Ödeme bölgesi metni (satır satır taranır)
            text = div.get_text(separator="\n")
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            for idx, ln in enumerate(lines):
                up = ln.upper()
                for label in PAYOUT_LABELS:
                    if up.startswith(label):
                        # Aynı satırda + sonraki birkaç satırda kombinasyon & tutar ara
                        window = " ".join(lines[idx:idx + 3])
                        rest = window[len(label):]
                        amt = _AMOUNT_RE.search(rest)
                        combo = _COMBO_RE.search(rest)
                        if amt:
                            out.append({
                                "Tarih": date_str, "Sehir": city_name,
                                "Kosu_ID": race_id, "Bahis": label,
                                "Kombinasyon": combo.group(0) if combo else "",
                                "Tutar": amt.group(0).replace('.', '').replace(',', '.'),
                            })
                        break  # bu satır için ilk eşleşen etiket yeter
    except Exception as e:
        print(f"      (ödeme parse atlandı: {e})")
    return out


INCOMPLETE_JSON = os.path.join(BASE_DIR, "incomplete_dates.json")


def _load_incomplete():
    """Yarım kalmış (şehir hatası / gün-içi çekim) tarihlerin sidecar listesi."""
    if os.path.isfile(INCOMPLETE_JSON):
        try:
            with open(INCOMPLETE_JSON, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def _save_incomplete(dates: set):
    try:
        with open(INCOMPLETE_JSON, "w", encoding="utf-8") as f:
            json.dump(sorted(dates), f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [!] incomplete_dates.json yazılamadı: {e}")


def _drop_date_rows(csv_path: str, date_str: str):
    """Bir tarihin eski satırlarını CSV'den siler (yeniden işleme öncesi temizlik)."""
    if not os.path.isfile(csv_path):
        return
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "Tarih" not in df.columns:
            return
        before = len(df)
        df = df[df["Tarih"].astype(str) != date_str]
        if len(df) < before:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"  [Temizlik] {os.path.basename(csv_path)}: {date_str} için {before - len(df)} eski satır silindi.")
    except Exception as e:
        print(f"  [!] {csv_path} tarih temizliği hatası: {e}")


def _append_aligned(df: pd.DataFrame, csv_path: str):
    """CSV'ye şema-uyumlu ekleme.

    Kör `mode='a'` append, scraper'a yeni kolon eklenince (ör. Yaris_Turu,
    Mesafe, Fark) eski 16-kolonluk başlığın altına 25 alanlı satır yazar ve
    dosya pandas tarafından OKUNAMAZ hale gelir. Burada:
      - kolonlar başlıkla birebir aynıysa → hızlı append,
      - df eski şemanın alt kümesiyse → başlığa hizala, append,
      - df YENİ kolon getiriyorsa → dosya birleşik şemayla bir kez yeniden
        yazılır (eski satırlar yeni kolonlarda boş kalır).
    """
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return
    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if list(df.columns) == header:
        df.to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8-sig")
        return
    new_cols = [c for c in df.columns if c not in header]
    if not new_cols:
        df.reindex(columns=header).to_csv(csv_path, mode="a", index=False,
                                          header=False, encoding="utf-8-sig")
        return
    # dtype=str + keep_default_na=False: mevcut satırların metni aynen korunur
    old = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    union = header + new_cols
    combined = pd.concat([old, df], ignore_index=True).reindex(columns=union)
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [Şema] {os.path.basename(csv_path)}: {len(new_cols)} yeni kolon "
          f"({', '.join(new_cols)}) — dosya birleşik şemayla yeniden yazıldı.")


def main():
    # Döngü için başlangıç ve bitiş tarihleri.
    # Bitiş = bugün → cron/UI ile güncel günlerin sonuçları da çekilir (resume sayesinde
    # zaten çekilmiş günler atlanır, sadece yeni günler işlenir).
    start_date = datetime(2025, 4, 15)
    end_date = datetime.now()

    csv_filename = os.path.join(BASE_DIR, "yaris_ana_tablo.csv")
    payouts_filename = os.path.join(BASE_DIR, "payouts_tablo.csv")

    # ── Resume mantığı: daha önce çekilen tarihleri tespit et ──
    # TJK_FORCE_RESCRAPE=1 → resume atlaması kapatılır (backfill: yeni kolonları
    # -AGF_Oran, Fark_Boy, Mesafe vb.- geçmiş günlere de doldurmak için tüm
    # tarihler yeniden çekilir; per-date temizlik satırları yerinde günceller).
    # scraped_dates HER ZAMAN dosyadan yüklenir (per-date drop/dedup buna bağlı).
    # force_rescrape yalnızca ATLAMA kararını etkiler, drop'u değil — aksi halde
    # backfill'de eski satırlar silinmeden yenileri eklenip çift kayıt olurdu.
    force_rescrape = os.environ.get("TJK_FORCE_RESCRAPE", "").strip() in ("1", "true", "yes")
    scraped_dates = set()
    backfilled_dates = set()  # yeni kolonları (AGF_Oran) zaten dolu tarihler
    if os.path.isfile(csv_filename):
        try:
            cols = pd.read_csv(csv_filename, nrows=0, encoding="utf-8-sig").columns
            use = ["Tarih"] + (["AGF_Oran"] if "AGF_Oran" in cols else [])
            df_existing = pd.read_csv(csv_filename, encoding="utf-8-sig", usecols=use)
            scraped_dates = set(df_existing["Tarih"].dropna().unique())
            if "AGF_Oran" in df_existing.columns:
                ok = df_existing.dropna(subset=["AGF_Oran"])
                backfilled_dates = set(ok["Tarih"].unique())
            print(f"[Resume] {csv_filename} mevcut — {len(scraped_dates)} tarih çekilmiş, "
                  f"{len(backfilled_dates)} tarihte yeni kolonlar dolu.")
        except Exception as e:
            print(f"[Resume] CSV okunurken hata (sıfırdan başlanacak): {e}")
    if force_rescrape:
        print("[Resume] TJK_FORCE_RESCRAPE aktif — yeni kolonu eksik tarihler "
              "yeniden çekilecek (backfill, resume-edilebilir).")

    # Yarım kalmış günler: bir sonraki koşuda yeniden işlenir (kendini iyileştirme)
    incomplete_dates = _load_incomplete()
    if incomplete_dates:
        print(f"[Resume] {len(incomplete_dates)} yarım gün yeniden işlenecek: {sorted(incomplete_dates)}")
    
    # 1. Driver'ı başlat ve ana sayfaya BİR KERE git
    driver = setup_driver()
    print("Ana sonuç sayfasına ilk kez bağlanılıyor...")
    driver.get("https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari")
    time.sleep(3)
    
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    day_count = 0

    try:
        while current_date <= end_date:
            day_count += 1
        # Form için GG/AA/YYYY formatında metin hazırlıyoruz
            date_str_input = current_date.strftime("%d/%m/%Y")
            date_str_display = current_date.strftime("%d.%m.%Y")
            
            is_today = current_date.date() == datetime.now().date()

            # Resume: bu tarih zaten çekilmişse atla — İSTİSNALAR:
            #  • bugün (gün içinde koşulan yeni yarışlar için her zaman yeniden çekilir)
            #  • yarım kalmış günler (şehir hatası / gün-içi çekim → kendini iyileştirir)
            # Normal resume: zaten çekilmiş günü atla.
            # Backfill (force): yalnızca yeni kolonu HÂLÂ eksik günleri işle
            # (böylece kesilip yeniden başlarsa tamamlananları tekrar etmez).
            already = (date_str_display in backfilled_dates if force_rescrape
                       else date_str_display in scraped_dates)
            if (already
                    and date_str_display not in incomplete_dates
                    and not is_today):
                current_date += timedelta(days=1)
                continue

            print(f"\n[{day_count}/{total_days}] [{date_str_display}] İşlem başlıyor...")
        
            try:
                _ = driver.title
            except Exception:
                print("  [!] Tarayıcı bağlantısı koptu/çöktü, yeniden başlatılıyor...")
                try: driver.quit()
                except: pass
                driver = setup_driver()
                driver.get("https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari")
                time.sleep(3)

            try:
                tarih_input = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "QueryParameter_Tarih"))
                )
                tarih_input.clear()
                driver.execute_script("arguments[0].value = '';", tarih_input)
                tarih_input.send_keys(date_str_input)
                tarih_input.send_keys(Keys.ENTER)
                time.sleep(1)
                
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'gunluk-tabs'))
                    )
                    time.sleep(2)
                except Exception:
                    pass
                
                cities = get_daily_cities(driver)
                
                if not cities:
                    print(f"  --> {date_str_display} tarihinde yarış bulunamadı veya sayfa yüklenemedi.")
                    current_date += timedelta(days=1)
                    continue

                print(f"  Bulunan şehirler: {[c['name'] for c in cities]}")
                daily_data = []
                daily_payouts = []
                city_failed = False

                for index, city in enumerate(cities):
                    city_id = city['id']
                    city_name = city['name']
                    print(f"  --> {city_name} işleniyor...")
                    
                    # Yabancı yarışları filtrele (İdman/statik veri kalitesi düşük olduğu ve pipeline'ı yavaşlattığı için)
                    # Türk şehirleri sayfada "Y.G." kısaltmasıyla gelir (ör. "Ankara  (6. Y.G.)")
                    if "Y.G." not in city_name and "Yarış Günü" not in city_name:
                        print(f"      ⏭ Yabancı yarış atlanıyor: {city_name}")
                        continue
                    
                    if index > 0:
                        try:
                            city_tab = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, f"a[data-sehir-id='{city_id}']"))
                            )
                            driver.execute_script("arguments[0].click();", city_tab)
                            time.sleep(random.uniform(2.5, 4.5))
                        except Exception as e:
                            print(f"      Şehir sekmesine ({city_name}) tıklanırken hata -> {e}")
                            city_failed = True  # gün yarım → sonraki koşuda yeniden işlenecek
                            continue
                    else:
                        time.sleep(random.uniform(1.0, 2.0))
                    
                    try:
                        all_races_btn = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.ID, "allRaces"))
                        )
                        driver.execute_script("arguments[0].click();", all_races_btn)
                        time.sleep(random.uniform(1.5, 3.0))
                    except Exception:
                        print(f"      'Tüm Koşular' butonuna tıklanamadı ({city_name})")
                    
                    html_source = driver.page_source
                    races_data = parse_race_table(html_source, date_str_display, city_name)

                    if races_data:
                        daily_data.extend(races_data)
                        print(f"      {len(races_data)} at verisi çekildi.")
                    else:
                        print(f"      Veri bulunamadı veya tablolar ayrıştırılamadı.")

                    # Egzotik ödemeler (forward-test gerçek ödeme skorlaması; hataya dayanıklı)
                    try:
                        payouts = parse_payouts(html_source, date_str_display, city_name)
                        if payouts:
                            daily_payouts.extend(payouts)
                    except Exception:
                        pass

                    time.sleep(random.uniform(1.5, 3.5))
                    
                if daily_data:
                    # Yeniden işlenen gün: eski satırları sil (duplicate önleme)
                    if date_str_display in scraped_dates:
                        _drop_date_rows(csv_filename, date_str_display)
                        _drop_date_rows(payouts_filename, date_str_display)
                    df = pd.DataFrame(daily_data)
                    _append_aligned(df, csv_filename)
                    scraped_dates.add(date_str_display)
                    print(f"[{date_str_display}] Başarıyla eklendi! Toplam {len(daily_data)} kayıt -> {csv_filename}")

                if daily_payouts:
                    pdf = pd.DataFrame(daily_payouts)
                    _append_aligned(pdf, payouts_filename)
                    print(f"[{date_str_display}] {len(daily_payouts)} ödeme kaydı -> payouts_tablo.csv")

                # Yarım-gün takibi: şehir hatası veya gün-içi (bugün) çekim →
                # sonraki koşuda yeniden işlenmek üzere işaretle; temiz geçmiş
                # gün → işaret varsa kaldır.
                if city_failed or is_today:
                    if date_str_display not in incomplete_dates:
                        incomplete_dates.add(date_str_display)
                        _save_incomplete(incomplete_dates)
                elif date_str_display in incomplete_dates:
                    incomplete_dates.discard(date_str_display)
                    _save_incomplete(incomplete_dates)

            except Exception as e:
                print(f"HATA oluştu ({date_str_display}): {e}")
                # Gün ortasında patladıysa yarım kabul et
                if date_str_display not in incomplete_dates:
                    incomplete_dates.add(date_str_display)
                    _save_incomplete(incomplete_dates)
                
            current_date += timedelta(days=1)
            time.sleep(random.uniform(4.0, 7.0))

    except KeyboardInterrupt:
        print("\n\n[!] Kullanıcı tarafından durduruldu. Sonraki çalışmada kaldığı yerden devam edecektir.")
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        
        print("\nVeri toplama işlemi (Stage 1) tamamlandı.")

if __name__ == "__main__":
    main()

