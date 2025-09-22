#!/usr/bin/env bash

#!/usr/bin/env bash
# install_r.sh — add R + IRkernel to a devcontainer
# Usage from devcontainer.json:
# "postCreateCommand": "python -m pip install --upgrade pip && pip install -r /workspace/requirements.txt && bash install_r.sh"
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

need() { command -v "$1" >/dev/null 2>&1 || apt-get update && apt-get install -y --no-install-recommends "$1"; }

# Minimal tools we need before touching apt sources
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg dirmngr lsb-release

ID="$(. /etc/os-release && echo "${ID}")"
CODENAME="$(lsb_release -sc || true)"
if [[ -z "${CODENAME}" ]]; then
    CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
fi

# Map Ubuntu codenames to CRAN suite names; Debian uses "<codename>-cran40"
if [[ "${ID}" == "ubuntu" ]]; then
    case "${CODENAME}" in
    noble | jammy | focal) CRAN_SUITE="${CODENAME}-cran40" ;;
    *)
        echo "Unsupported/unknown Ubuntu codename: ${CODENAME}"
        exit 1
        ;;
    esac
elif [[ "${ID}" == "debian" ]]; then
    case "${CODENAME}" in
    bookworm | bullseye) CRAN_SUITE="${CODENAME}-cran40" ;;
    *)
        echo "Unsupported/unknown Debian codename: ${CODENAME}"
        exit 1
        ;;
    esac
else
    echo "This script supports Debian/Ubuntu bases only (got: ${ID})"
    exit 1
fi

# Import CRAN key (Johannes Ranke). Fingerprint:
# 95C0 FA F3 8D B3 CC AD 0C 08 0A 7B DC 78 B2 DD EA BC 47 B7
if ! gpg --list-keys 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7 >/dev/null 2>&1; then
    gpg --keyserver keyserver.ubuntu.com --recv-key 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7
fi
install -d -m 0755 /etc/apt/trusted.gpg.d /etc/apt/sources.list.d
gpg --armor --export 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7 >/etc/apt/trusted.gpg.d/cran_debian_key.asc

# Add CRAN repo (same URL for Debian/Ubuntu, suite differs)
echo "deb http://cloud.r-project.org/bin/linux/${ID} ${CRAN_SUITE}/" >/etc/apt/sources.list.d/cran_r.list

# Core libs often needed to compile R packages
apt-get update
apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev libtiff5 libjpeg62-turbo \
    libxt6 locales

# Keep image lean
rm -rf /var/lib/apt/lists/*

# Install IRkernel (and optionally extras)
Rscript - <<'RS'
repos <- "https://cloud.r-project.org"
if (!requireNamespace("IRkernel", quietly=TRUE)) install.packages("IRkernel", repos=repos)
RS

# Optional extras: set INSTALL_R_EXTRAS=1 in environment to enable
if [[ "${INSTALL_R_EXTRAS:-0}" == "1" ]]; then
    Rscript - <<'RS'
repos <- "https://cloud.r-project.org"
pkgs <- c("tidyverse","data.table","arrow","DBI","RPostgres","languageserver")
to_install <- pkgs[!sapply(pkgs, function(p) requireNamespace(p, quietly=TRUE))]
if (length(to_install)) install.packages(to_install, repos=repos)
RS
fi

# Register the Jupyter kernel only if jupyter is present now
if command -v jupyter >/dev/null 2>&1; then
    Rscript - <<'RS'
IRkernel::installspec(user = FALSE)
RS
else
    echo "Note: jupyter not found; skipping IRkernel::installspec. It will run cleanly later if needed:"
    echo 'R -q -e "IRkernel::installspec(user = FALSE)"'
fi

echo "R + IRkernel setup complete."
