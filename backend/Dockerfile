FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PULSE_DATA_DIR=/var/lib/pulse/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pulse ./pulse
EXPOSE 8000
CMD ["uvicorn", "pulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
