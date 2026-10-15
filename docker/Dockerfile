FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
LABEL repairbench.runtime="1"
COPY requirements/runtime.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt && mkdir /workspace && chmod 555 /workspace
COPY docker/runner.py /opt/runner.py
USER 65532:65532
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HOME=/tmp
ENTRYPOINT ["python", "/opt/runner.py"]
