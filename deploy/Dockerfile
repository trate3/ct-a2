# Trusted-workload container for a Confidential Space deployment.
#
# Runs `serve_session` behind the mock case's static-TreeCut environment as a
# GPU-free run of the attestation path.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY ct_core ct_core
COPY util util
COPY evals evals
COPY evals_runner evals_runner
COPY tee tee

RUN pip install --no-cache-dir -e ".[gcp]"

# The public spec is deterministic from (case, n, seed), so bake it at build time
# rather than needing a shell step inside the measured Confidential Space image.
RUN python -m evals_runner protocol-integrity-exp spec \
    --case mock --n 8 --spec /app/spec.json

# Confidential Space's launcher reads this to decide what to publish from the
# container to the host network -- without it the workload listens inside the
# container but is unreachable from outside the VM.
EXPOSE 7401

# Confidential Space runs the container's entrypoint directly; no shell needed
# beyond what's already generated at build time.
ENTRYPOINT ["python", "-m", "evals_runner", "protocol-integrity-exp"]
CMD ["workload", "--case", "mock", "--spec", "/app/spec.json", \
     "--listen-host", "0.0.0.0", "--port", "7401", "--attestation", "gcp"]
