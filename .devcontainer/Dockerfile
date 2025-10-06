FROM python:3.12.1-slim-bullseye

USER root
ARG USER=root
ARG UID=1000
ARG GID=1000

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common curl git build-essential direnv wget && \
    pip install --upgrade pip

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /workspace/requirements.txt
