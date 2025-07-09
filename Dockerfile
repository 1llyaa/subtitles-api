FROM python:3.10-slim

# FFmpeg & základní balíky
RUN apt-get update && apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Pracovní adresář
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

