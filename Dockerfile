# Stage 1 : build du frontend React
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2 : runtime Python
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch==2.6.0+cpu \
        torchvision==0.21.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY app/ ./app/
COPY indexes/ ./indexes_image/
COPY run.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

COPY --from=frontend-builder /app/static ./app/static/

EXPOSE 8000

CMD ["./entrypoint.sh"]