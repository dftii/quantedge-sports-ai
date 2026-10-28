FROM python:3.12-slim AS builder

ENV PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    QUANTEDGE_DATABASE_URL=sqlite:////data/quantedge.db

RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --no-create-home --shell /usr/sbin/nologin app \
 && mkdir /data \
 && chown app:app /data

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY app/ ./app/

USER app

VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
