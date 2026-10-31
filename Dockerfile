FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    FOREST_SMOKE_PUBLIC_MODE=1 \
    FOREST_SMOKE_DEVICE=cpu \
    FOREST_SMOKE_MODEL=models/hardneg_finetune_cont30_full_20261029_034435_best.pt

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.10.0.84

COPY . .
RUN mkdir -p /app/outputs/predictions/public /tmp/matplotlib /tmp/ultralytics

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
