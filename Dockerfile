# The scoring service of the credit risk engine (slice 9).
#
#   docker build -t credit-risk-engine .
#   docker run --rm -p 8000:8000 credit-risk-engine
#   then open http://localhost:8000
#
# What goes in: src/, service/, tests/ and five files from models/. No data.
# The service never reads data/ - everything it learned is in models/.

FROM python:3.13-slim

# The image is Linux; the saved files were written on Windows. The same code
# rounds the last bit of a float differently on the two (src/tolerance.py),
# so inside the container "equal" means equal to 1e-12, not to the last bit.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CRE_FLOAT_TOL=1e-12

WORKDIR /app

# dependencies first: this layer is rebuilt only when the pins change
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ src/
COPY service/ service/
COPY tests/ tests/
# named one by one, so the losing LightGBM bundle and anything else that
# lands in models/ later does not ride along
COPY models/calibrated_logreg.joblib models/feature_spec.json \
     models/economics.json models/policy.json models/monitor_policy.json models/

# The image tests itself as it is built. The contract and service tests need
# only models/ and run here; the skew tests need data/processed, which is not
# in the image, and skip with that reason. A build that finishes has loaded
# the model, built features and answered requests on this exact image.
RUN python -m pytest tests -q -p no:cacheprovider

RUN useradd --create-home app
USER app

# A hosting platform tells the container which port to listen on through
# PORT. Run by hand, nothing sets it and the service stays on 8000.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:' + os.environ['PORT'] + '/health', timeout=2)"

CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT}"]
