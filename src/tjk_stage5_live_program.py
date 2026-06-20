#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK AŞAMA 5 — GÜNLÜK PROGRAM SCRAPER (yarış OYNANMADAN)
================================================================================
  Koşacak atların listesini çeker (sonuç DEĞİL). Stage 1 deseninin program
  sayfasına uyarlanmış hâlidir.

  Sayfa: https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisProgrami
  (JS-render → Selenium gerekir.)

  Çıktı `program_tablo.csv` sütunları (build_live_features ile uyumlu):
      Tarih, Sehir, Pist_Durumu, Kosu_ID, At_Adi, Yas, Siklet, Start,
      At_URL, Jokey_Adi, Jokey_URL, Antrenor_Adi, Antrenor_URL, Ganyan

  Kullanım:
      python tjk_stage5_live_program.py                 # bugün
      python tjk_stage5_live_program.py --date 21/06/2026

  NOT: Program sayfasının td class adları sonuç sayfasından farklı olabilir;
       seçiciler ilk canlı çalıştırmada gözlemlenip gerekiyorsa ince ayar
       yapılmalıdır (parse_program_table içindeki SELECTOR'lar).
================================================================================
"""

import os
import re
import sys
import time
import random
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

BASE_URL    = "https://www.tjk.org"
PROGRAM_URL = "https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisProgrami"
# Program CSV proje kökündeki data/ klasörüne yazılır (src/ -> .. -> data)
BASE_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_CSV  = os.path.join(BASE_DIR, "program_tablo.csv")


def setup_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def get_daily_cities(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    ul_tabs = soup.find("ul", class_="gunluk-tabs")
    if not ul_tabs:
        return []
    cities = []
    for li in ul_tabs.find_all("li"):
        a = li.find("a", attrs={"data-sehir-id": True})
        if a:
            cities.append({"id": a["data-sehir-id"], "name": a.text.strip()})
    return cities


def _fix_url(href):
    href = href.replace("../../", "/TR/map/").split("&Era=")[0]
    if not href.startswith("/"):
        href = "/" + href
    return BASE_URL + href


def parse_program_table(html_source, date_str, city_name):
    """Program DOM'undan koşacak atları ayrıştırır (sonuç alanları yok)."""
    soup = BeautifulSoup(html_source, "html.parser")
    rows_out = []

    # Pist durumu (varsa)
    track = "Bilinmiyor"
    w = soup.find("span", class_="raceWeatherBrown")
    if w:
        track = w.text.strip()

    # Koşu blokları — program/sonuç sayfasında benzer id deseni
    race_divs = soup.find_all("div", id=re.compile(r"^(kosubilgisi|programkosu)--?\d+"))
    for div in race_divs:
        race_id = re.sub(r"^(kosubilgisi|programkosu)-", "", div.get("id", ""))
        table = div.find("table", class_=re.compile(r"tablesorter|table", re.I))
        if not table:
            continue

        # Koşu saati — başlıktan ("2. Koşu 16.45" → "16:45"). "Koşu" öneki sayesinde
        # tablodaki derece zamanlarıyla (ör. 1.10.02) karışmaz; ilk eşleşme başlıktır.
        saat = None
        mt = re.search(r"Koşu\s*([01]?\d|2[0-3])[.:]([0-5]\d)", div.get_text(" "))
        if mt:
            saat = f"{mt.group(1).zfill(2)}:{mt.group(2)}"

        tbody = table.find("tbody") or table
        for row in tbody.find_all("tr"):
            d = {
                "Tarih": date_str, "Sehir": city_name, "Pist_Durumu": track,
                "Kosu_ID": race_id, "Kosu_Saati": saat, "At_Adi": None, "Yas": None,
                "Siklet": None, "Start": None, "At_URL": None, "Jokey_Adi": None,
                "Jokey_URL": None, "Antrenor_Adi": None, "Antrenor_URL": None,
                "Ganyan": None,
            }

            # At adı + URL
            td = row.select_one('td[class*="-AtAdi"]')
            if td and td.find("a"):
                a = td.find("a")
                d["At_Adi"] = re.sub(r"\(\d+\)", "", a.text).strip()
                if a.has_attr("href"):
                    d["At_URL"] = _fix_url(a["href"])
            if not d["At_Adi"]:
                continue  # at adı yoksa geçerli satır değil

            # Yaş / Kilo / Start
            for sel, key, num in [
                ('td[class*="-Yas"]', "Yas", False),
                ('td[class*="-Kilo"]', "Siklet", True),
                ('td[class*="-StartId"], td[class*="-Kura"]', "Start", False),
            ]:
                cell = row.select_one(sel)
                if cell:
                    txt = cell.text.strip()
                    if num:
                        m = re.search(r"(\d+(?:[.,]\d+)?)", txt)
                        txt = m.group(1).replace(",", ".") if m else txt
                    d[key] = txt

            # Jokey
            td = row.select_one('td[class*="-JokeAdi"]')
            if td and td.find("a"):
                a = td.find("a")
                d["Jokey_Adi"] = a.text.strip()
                if a.has_attr("href"):
                    d["Jokey_URL"] = _fix_url(a["href"])

            # Antrenör
            td = row.select_one('td[class*="-AntronorAdi"]')
            if td and td.find("a"):
                a = td.find("a")
                d["Antrenor_Adi"] = a.text.strip()
                if a.has_attr("href"):
                    d["Antrenor_URL"] = _fix_url(a["href"])

            # Muhtemel Ganyan / AGF (yarış öncesi piyasa sinyali — varsa)
            td = row.select_one('td[class*="-Gny"], td[class*="-AGF"], td[class*="-MuhtemelGanyan"]')
            if td:
                span = td.find("span")
                val = (span.text if span else td.text).replace(",", ".").strip()
                d["Ganyan"] = val or None

            rows_out.append(d)
    return rows_out


def scrape_program(date_str_input, headless=False):
    """Verilen tarih (GG/AA/YYYY) için tüm şehirlerin programını çeker."""
    date_disp = date_str_input.replace("/", ".")
    driver = setup_driver(headless=headless)
    all_rows = []
    try:
        driver.get(PROGRAM_URL)
        time.sleep(3)

        # Tarih gir
        try:
            inp = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "QueryParameter_Tarih"))
            )
            inp.clear()
            driver.execute_script("arguments[0].value = '';", inp)
            inp.send_keys(date_str_input)
            inp.send_keys(Keys.ENTER)
            time.sleep(2)
        except Exception:
            print("  [!] Tarih kutusu bulunamadı — sayfa bugünü gösteriyor olabilir.")

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gunluk-tabs"))
            )
            time.sleep(2)
        except Exception:
            pass

        cities = get_daily_cities(driver)
        if not cities:
            print(f"  --> {date_disp}: program bulunamadı / yarış yok.")
            return []
        print(f"  Bulunan şehirler: {[c['name'] for c in cities]}")

        for idx, city in enumerate(cities):
            name = city["name"]
            # Yerli yarış günleri "Y.G." veya "Yarış Günü" içerir; yabancılar (ABD,
            # Avustralya, ...) ve "Karma" atlanır (eğitim verisiyle tutarlı).
            if ("Y.G." not in name) and ("Yarış Günü" not in name):
                print(f"      ⏭ Atlanıyor (yabancı/karma): {name}")
                continue
            print(f"  --> {name} işleniyor...")
            if idx > 0:
                try:
                    tab = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, f"a[data-sehir-id='{city['id']}']"))
                    )
                    driver.execute_script("arguments[0].click();", tab)
                    time.sleep(random.uniform(2.5, 4.0))
                except Exception as e:
                    print(f"      Sekme tıklanamadı ({name}): {e}")
                    continue
            # "Tüm koşular"
            try:
                btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "allRaces"))
                )
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(random.uniform(1.5, 3.0))
            except Exception:
                pass

            rows = parse_program_table(driver.page_source, date_disp, name)
            print(f"      {len(rows)} at çekildi.")
            all_rows.extend(rows)
            time.sleep(random.uniform(1.5, 3.0))
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    return all_rows


def main():
    date_arg = None
    headless = "--headless" in sys.argv
    if "--date" in sys.argv:
        date_arg = sys.argv[sys.argv.index("--date") + 1]
    if not date_arg:
        date_arg = datetime.now().strftime("%d/%m/%Y")

    print("█" * 60)
    print(f"  STAGE 5 — GÜNLÜK PROGRAM SCRAPER  ({date_arg})")
    print("█" * 60)

    rows = scrape_program(date_arg, headless=headless)
    if not rows:
        print("Program verisi alınamadı.")
        return

    df = pd.DataFrame(rows)
    date_disp = date_arg.replace("/", ".")
    # Aynı tarihi tekrar yazma (idempotent): mevcut dosyadan o tarihi düşür
    if os.path.isfile(OUTPUT_CSV):
        old = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        old = old[old["Tarih"].astype(str) != date_disp]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ {len(rows)} program satırı kaydedildi → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
