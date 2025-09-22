#!/usr/bin/env bash

# set -e

sudo apt-get update -y 
sudo apt-get install -y xclip zoxide direnv software-properties-common
cargo install eza
curl -fsSL https://fnm.vercel.app/install | bash

pip install -r requirements.txt 
