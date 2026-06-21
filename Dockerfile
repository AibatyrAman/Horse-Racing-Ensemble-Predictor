# ─────────────────────────────────────────────────────────────
#  TJK Tahmin Paneli — Docker image
#  Python 3.12 + Chromium (headless Selenium) + Streamlit
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Sistem bağımlılıkları: Chromium + ChromeDriver (Selenium scraping için)
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        chromium-driver \
        fonts-liberation \
        libnss3 \
        libxss1 \
        libasound2 \
        libatk-bridge2.0-0 \
        libgtk-3-0 \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Chromium'un Selenium tarafından bulunması için
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Saat dilimi
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Python bağımlılıkları (cache-friendly: requirements önce)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kod
COPY src/ ./src/
COPY deploy/ ./deploy/
COPY reports/ ./reports/

# Volume mount noktaları (docker-compose'da tanımlı)
# data/, models/, outputs/, runs/ → host'tan mount edilir

# Streamlit ayarları
ENV TJK_HEADLESS=1
EXPOSE 8501

# Streamlit'i /ganyan base path ile başlat
CMD ["streamlit", "run", "src/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.baseUrlPath", "/ganyan", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
