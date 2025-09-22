# FROM python:3.12.1-slim-bullseye
FROM jupyter/datascience-notebook

# ARG USER=vscode 
# ARG UID=1000
# ARG GID=1000

USER root
ARG USER=root
ARG UID=1000
ARG GID=1000

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common curl git build-essential direnv wget
    # rm -rf /var/lib/apt/lists/*
    # libcurl4-openssl-dev libssl-dev libxml2-dev \
    # libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5 libjpeg62-turbo \
    # libxt6 locales && \


RUN groupadd -f jovyan && usermod -aG jovyan jovyan
    # useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USER} && \

USER ${NB_USER}
WORKDIR /home/jovyan
