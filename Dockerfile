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
COPY configs/final_model.yaml ./configs/final_model.yaml

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
