#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd ${SCRIPT_DIR}
source env/bin/activate

if [ -f workspace/mednist_project/prod_00/compose.yaml ]; then
    export IMAGE_NAME=nvflare-deploy
    export PYTHON_EXECUTABLE=python
    export WORKSPACE=/workspace
    docker compose -f workspace/mednist_project/prod_00/compose.yaml down
fi

docker compose down
sudo rm -rf workspace/
rm -f project.yml