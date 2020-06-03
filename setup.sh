#!/bin/bash -eu
# @(#) setup heroku

mkdir -p ~/.streamlit/

cat <<EOF >~/.streamlit/config.toml
[server]
headless = true
port = $PORT
enableCORS = false
EOF
