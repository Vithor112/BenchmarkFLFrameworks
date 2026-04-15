#!/usr/bin/bash
nvflare poc stop -ex admin@nvidia.com
docker compose down
sudo rm -rf /tmp/nvflare/poc
docker build -t nvflare-pt-docker . -f Dockerfile
python job/job.py
nvflare poc prepare -d nvflare-pt-docker -n $1
docker compose up -d
nvflare poc start -gpu 0 -ex admin@nvidia.com
nvflare job submit -j /tmp/nvflare/jobs/job_config/job_test/
