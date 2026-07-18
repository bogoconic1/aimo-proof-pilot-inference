# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG VCS_REF=unknown

LABEL org.opencontainers.image.source="https://github.com/bogoconic1/aimo-proof-pilot-inference"
LABEL org.opencontainers.image.revision="$VCS_REF"
LABEL org.opencontainers.image.title="AIMO Proof Pilot Inference"
LABEL org.opencontainers.image.description="OpenRouter generate-verify-refine proof inference"

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aimo-proof-pilot-inference
COPY evaluation/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /opt/aimo-proof-pilot-inference

RUN chmod 0755 docker/entrypoint.sh run_submission.sh

ENV REPO=/opt/aimo-proof-pilot-inference

VOLUME ["/workspace"]
STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/aimo-proof-pilot-inference/docker/entrypoint.sh"]
CMD ["submission"]
