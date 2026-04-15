#!/usr/bin/bash
nvflare poc stop -ex admin@nvidia.com
docker compose down
sudo rm -rf /tmp/nvflare/poc
docker build -t nvflare-pt-docker . -f Dockerfile
python job/job.py
nvflare poc prepare -d nvflare-pt-docker -n $1
docker compose up -d
export GPU2USE='--gpus=all' # nvflare poc start -gpu flag does not work with docker :)
nvflare poc start -ex admin@nvidia.com
nvflare job submit -j /tmp/nvflare/jobs/job_config/job_test/
