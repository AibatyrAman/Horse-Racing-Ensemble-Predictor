import time
import random
import re
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

# CSV her zaman script'in yanına yazılsın (çalışma dizininden bağımsız)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_driver():
    """
    Selenium WebDriver'ı başlatır.
    """
    options = Options()
    # Tarayıcıyı arka planda çalıştırmak için headless mod (Görünmez)
    #options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("window-size=1920,1080")
    # TJK gibi siteler otomasyonu engelliyorsa, bazı anti-bot ayarları da eklenebilir.
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    
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
                'Derece': None
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
            
            data_list.append(horse_data)
            
    return data_list

def main():
    # Döngü için başlangıç ve bitiş tarihleri (Test/Kısa süre için değiştirilebilirsiniz)
    start_date = datetime(2025, 4, 15)
    end_date = datetime(2026, 4, 15)
    
    csv_filename = os.path.join(BASE_DIR, "yaris_ana_tablo.csv")
    
    # ── Resume mantığı: daha önce çekilen tarihleri tespit et ──
    scraped_dates = set()
    if os.path.isfile(csv_filename):
        try:
            df_existing = pd.read_csv(csv_filename, encoding="utf-8-sig", usecols=["Tarih"])
            scraped_dates = set(df_existing["Tarih"].dropna().unique())
            print(f"[Resume] {csv_filename} mevcut — {len(scraped_dates)} benzersiz tarih zaten çekilmiş.")
        except Exception as e:
            print(f"[Resume] CSV okunurken hata (sıfırdan başlanacak): {e}")
    
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
            
            # Resume: bu tarih zaten çekilmişse atla
            if date_str_display in scraped_dates:
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
                
                for index, city in enumerate(cities):
                    city_id = city['id']
                    city_name = city['name']
                    print(f"  --> {city_name} işleniyor...")
                    
                    # Yabancı yarışları filtrele (İdman/statik veri kalitesi düşük olduğu ve pipeline'ı yavaşlattığı için)
                    if "Yarış Günü" not in city_name:
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
                        
                    time.sleep(random.uniform(1.5, 3.5))
                    
                if daily_data:
                    df = pd.DataFrame(daily_data)
                    file_exists = os.path.isfile(csv_filename)
                    df.to_csv(csv_filename, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                    print(f"[{date_str_display}] Başarıyla eklendi! Toplam {len(daily_data)} kayıt -> {csv_filename}")
                    
            except Exception as e:
                print(f"HATA oluştu ({date_str_display}): {e}")
                
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

