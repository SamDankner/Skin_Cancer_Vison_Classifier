FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ROOT=/models

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY models/convnext/convnext_convnext_tiny_20260921T060517413359Z_convnext.pt /models/convnext/
COPY models/efficientnet/efficientnet_efficientnet_v2_s_20260921T053334483303Z_efficientnet.pt /models/efficientnet/
COPY models/multimodal/multimodal_dinov2_vits14_20260921T224823181902Z_multimodal_serious_01.pt /models/multimodal/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
