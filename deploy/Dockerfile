FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FRONTEND_DIR=/srv/frontend

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.txt

COPY deploy/app/ app/
COPY frontend/ frontend/
COPY deploy/scripts/ scripts/
COPY deploy/models/category_maps.json deploy/models/model_spec.json models/
COPY data/final_submission_lgb_model.txt \
     data/quantile_final_p8.txt \
     data/quantile_final_p50.txt \
     data/quantile_final_p92.txt \
     /tmp/model-source/

RUN python scripts/export_artifacts.py \
      --source-dir /tmp/model-source \
      --output-dir /srv/models \
      --spec /srv/models/model_spec.json \
      --category-maps /srv/models/category_maps.json \
    && rm -rf /tmp/model-source \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home app \
    && mkdir -p /srv/logs \
    && chown -R 10001:10001 /srv

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
