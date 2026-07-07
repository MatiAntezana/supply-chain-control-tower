# Serves the demo: FastAPI (:8000, internal) + Streamlit dashboard (:7860).
# Hugging Face Spaces (Docker SDK) exposes 7860 by default.
# Precomputed pipeline artifacts (data/processed, models, reports) ship in the
# image — the M5 raw dataset is NOT needed at serving time.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY data/processed/ data/processed/
COPY models/ models/
COPY reports/ reports/
COPY start.sh .

# Streamlit/HF Spaces run as non-root; keep caches writable.
RUN useradd -m appuser && chown -R appuser /code
USER appuser
ENV HOME=/home/appuser PYTHONPATH=/code API_URL=http://localhost:8000

EXPOSE 7860
CMD ["bash", "start.sh"]
