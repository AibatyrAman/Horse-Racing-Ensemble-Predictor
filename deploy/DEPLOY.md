# VPS Kurulumu + Kendi Domainine Taşıma

TJK panelini (Streamlit) ve canlı zamanlayıcıyı (scheduler) bir Linux VPS'te
(Ubuntu 22/24 varsayıldı) çalıştırma rehberi. Arayüz `https://panel.SENIN-DOMAIN.com`
adresinde, **şifre korumalı** yayınlanır.

> Yer tutucular: `PANEL_DOMAIN` = panel.ornek.com · `APPDIR` = /opt/Ganyan · `USER` = tjk

---

## 0. Ön koşullar
- Bir VPS (1-2 vCPU, 2-4 GB RAM yeterli; Selenium+Chrome için ≥2 GB önerilir).
- Bir alan adı ve **DNS A kaydı**: `PANEL_DOMAIN` → VPS'in public IP'si.
- VPS'e SSH erişimi.

---

## 1. Sistem paketleri + headless Chrome
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git nginx \
                        chromium-browser chromium-chromedriver
# (alternatif) google-chrome-stable + eşleşen chromedriver; Selenium 4
# Selenium Manager sürücüyü otomatik indirebilir, yine de chromium kurmak güvenli.

# Saat dilimi — ŞART (scheduler tetikleri TJK saatine göre)
sudo timedatectl set-timezone Europe/Istanbul
```

## 2. Kullanıcı + kod
```bash
sudo useradd -m -s /bin/bash USER || true
sudo mkdir -p APPDIR && sudo chown USER:USER APPDIR
sudo -iu USER

git clone https://github.com/AibatyrAman/Horse-Racing-Ensemble-Predictor.git APPDIR
cd APPDIR
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 3. Veri + modelleri taşı (KRİTİK — git'te yoklar)
`.pkl` ve `.csv` dosyaları `.gitignore`'da olduğundan klonda gelmez. Yerel makinenden:
```bash
# Yerel makinede (proje kökünde):
rsync -avz data/   USER@VPS_IP:APPDIR/data/
rsync -avz models/ USER@VPS_IP:APPDIR/models/
```
**Alternatif:** VPS'te baştan üret (uzun, Selenium): `.venv/bin/python src/tjk_pipeline.py`
ardından `.venv/bin/python src/tjk_stage4_modeling.py` (+ `--ablation`, `--dump-oof`).

Hızlı doğrulama:
```bash
TJK_HEADLESS=1 .venv/bin/python src/tjk_stage5_live_program.py --date $(date +%d/%m/%Y)
.venv/bin/python src/tjk_stage8_betting_strategy.py --backtest   # OOF varsa
```

## 4. Scheduler'ı servis yap (her sabah otomatik)
`deploy/` içindeki birim dosyalarını kopyala (yolları/USER'ı düzenle):
```bash
sudo cp deploy/tjk-scheduler.service /etc/systemd/system/
sudo cp deploy/tjk-scheduler.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tjk-scheduler.timer
# Test (bugünü hemen kuru-çalıştır):
sudo -iu USER bash -c 'cd APPDIR && .venv/bin/python src/tjk_live_scheduler.py --dry-run'
```
Timer her sabah 08:00'de scheduler'ı başlatır; daemon gün boyu (post−lead çekimleri +
akşam reconcile) çalışıp gün sonunda biter. Loglar: `APPDIR/runs/scheduler.log`.

## 5. Streamlit'i servis yap (yalnız localhost'ta dinler)
```bash
sudo cp deploy/tjk-streamlit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tjk-streamlit.service
sudo systemctl status tjk-streamlit.service   # 127.0.0.1:8501 dinlemeli
```

## 6. Domain + nginx reverse proxy + ŞİFRE
```bash
# Basic auth kullanıcısı oluştur (panel şifresi):
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin     # şifre sorar

# nginx site config:
sudo cp deploy/nginx-tjk.conf /etc/nginx/sites-available/tjk
sudo sed -i 's/PANEL_DOMAIN/panel.ornek.com/g' /etc/nginx/sites-available/tjk
sudo ln -sf /etc/nginx/sites-available/tjk /etc/nginx/sites-enabled/tjk
sudo nginx -t && sudo systemctl reload nginx
```

## 7. HTTPS (Let's Encrypt — ücretsiz)
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d panel.ornek.com
# Otomatik yenileme zaten kurulur; test: sudo certbot renew --dry-run
```
Artık panel: **https://panel.ornek.com** (tarayıcı kullanıcı adı+şifre sorar).

---

## Güvenlik notları (oku!)
- **Basic auth tüm paneli korur** — İşlemler sekmesindeki butonlar (scrape/retrain)
  yalnız şifreyi bilenlerce tetiklenir. Şifreyi güçlü tut.
- Streamlit **sadece 127.0.0.1**'i dinler (servis dosyası öyle ayarlı) → dışarıdan
  doğrudan 8501'e erişilemez; tek giriş nginx (şifreli).
- **TJK rate-limit:** scheduler her tetikte tüm şehirleri çeker; `--lead` ve Stage 5'in
  rastgele beklemeleri korunmalı. Aşırı sık çekim IP banı riski taşır.
- İstersen herkese açık **salt-okunur** bir görünüm + ayrı şifreli yönetim ayırabiliriz
  (app'e `TJK_PUBLIC=1` ile İşlemler sekmesini gizleme — istenirse eklenir).

## Bakım
```bash
sudo systemctl restart tjk-streamlit       # arayüzü yeniden başlat
journalctl -u tjk-streamlit -f             # arayüz logu
journalctl -u tjk-scheduler -f             # scheduler logu (o gün çalışıyorsa)
cd APPDIR && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart tjk-streamlit
```

## Docker Compose ile Deploy (aizho.me/ganyan)

Aşağıdaki adımlar paneli `https://aizho.me/ganyan` adresinde Docker ile yayınlar.
VPS'te **Docker** ve **Docker Compose** (v2) kurulu olmalı; nginx zaten var varsayılır.

### D.0 — Ön koşullar
- VPS'te Docker + Docker Compose v2 kurulu.
- `aizho.me` DNS kaydı VPS IP'sine işaret ediyor.
- Nginx `/etc/nginx/sites-available/aizho.me` config'i var ve çalışıyor.
- Saat dilimi ayarlı: `sudo timedatectl set-timezone Europe/Istanbul`

### D.1 — Projeyi VPS'e çek
```bash
cd /var/www/aizho.me
git clone https://github.com/AibatyrAman/Horse-Racing-Ensemble-Predictor.git ganyan
cd ganyan
```

### D.2 — Data + Model dosyalarını taşı (KRİTİK — git'te yoklar!)
`.pkl` ve büyük `.csv` dosyaları `.gitignore`'da; clone'da gelmez.
**Yerel makineden:**
```bash
# Yerel makinede (proje kökünde):
rsync -avz data/   KULLANICI@VPS_IP:/var/www/aizho.me/ganyan/data/
rsync -avz models/ KULLANICI@VPS_IP:/var/www/aizho.me/ganyan/models/
```

### D.3 — Docker image oluştur + container başlat
```bash
cd /var/www/aizho.me/ganyan
docker compose up -d --build
# Doğrula:
docker compose logs -f streamlit       # "You can now view your Streamlit app" görmeli
curl -s http://127.0.0.1:8501/ganyan/_stcore/health   # {"status":"ok"}
```

### D.4 — nginx'e /ganyan location bloğu ekle
`deploy/nginx-ganyan-location.conf` dosyasındaki location bloklarını mevcut
`/etc/nginx/sites-available/aizho.me` config'inin `server { ... }` bloğu içine
yapıştır veya include ile ekle:
```bash
# Yöntem 1 — include (tavsiye):
#   server bloğunun içine şu satırı ekle:
#     include /var/www/aizho.me/ganyan/deploy/nginx-ganyan-location.conf;
# Yöntem 2 — kopyala-yapıştır:
#   deploy/nginx-ganyan-location.conf içeriğini elle server bloğuna ekle.

sudo nano /etc/nginx/sites-available/aizho.me
# ... düzenle ...
sudo nginx -t && sudo systemctl reload nginx
```

**Şifre koruması istenirse:**
```bash
sudo apt-get install -y apache2-utils   # (zaten varsa atla)
sudo htpasswd -c /etc/nginx/.htpasswd_ganyan admin    # şifre sorar
# Sonra nginx-ganyan-location.conf içindeki auth_basic satırlarının
# başındaki # işaretlerini kaldır, nginx'i yeniden yükle.
```

### D.5 — Scheduler (her sabah otomatik)
Timer dosyası değişmez; servis dosyası Docker versiyonunu kullanır:
```bash
sudo cp deploy/tjk-scheduler-docker.service /etc/systemd/system/tjk-scheduler.service
sudo cp deploy/tjk-scheduler.timer          /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tjk-scheduler.timer
# Test (bugünü kuru-çalıştır):
docker compose exec streamlit python src/tjk_live_scheduler.py --dry-run
```

### D.6 — HTTPS (Let's Encrypt)
```bash
# aizho.me için zaten certbot varsa /ganyan otomatik kapsar — sadece nginx reload yeterli.
# İlk seferse:
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d aizho.me
# Test: sudo certbot renew --dry-run
```

### D.7 — Doğrulama
```bash
# Container sağlığı
docker compose ps                     # STATUS: healthy
# Panel erişimi
curl -sI https://aizho.me/ganyan/     # HTTP 200 + Streamlit HTML
# Scheduler dry-run
docker compose exec streamlit python src/tjk_live_scheduler.py --dry-run
```

### Docker Bakım komutları
```bash
# Logları izle
docker compose logs -f streamlit

# Container yeniden başlat
docker compose restart streamlit

# Kod güncellemesi (git pull + rebuild)
cd /var/www/aizho.me/ganyan
git pull
docker compose up -d --build

# Scheduler logları
cat runs/scheduler.log
journalctl -u tjk-scheduler -f     # (o gün çalışıyorsa)
```
