#!/usr/bin/bash
nvflare poc stop 
docker compose stop
rm -rf /tmp/nvflare/poc
docker build -t nvflare-pt-docker . -f Dockerfile
nvflare poc prepare -d nvflare-pt-docker -n $1
docker compose up -d
nvflare poc start -gpu 0

