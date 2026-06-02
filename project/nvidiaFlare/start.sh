#!/bin/bash


CLIENTS=2
ROUNDS=3
EPOCHS=1
BATCH_SIZE=32
SEED=42
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--clients) 
            CLIENTS="$2"
            shift
            ;;
        -r|--rounds) 
            ROUNDS="$2"
            shift 
            ;;
        -e|--epochs) 
            EPOCHS="$2"
            shift 
            ;;
        -b|--batch-size) 
            BATCH_SIZE="$2"
            shift 
            ;;
        -s|--seed)
            SEED="$2"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -c, --clients     Number of clients (default: 2)"
            echo "  -r, --rounds      Number of rounds (default: 3)"
            echo "  -e, --epochs      Number of epochs (default: 1)"
            echo "  -b, --batch-size  Batch size (default: 32)"
            echo "  -s, --seed        Random seed for reproducibility (default: 42)"
            exit 0
            ;;
        *) 
            echo "Unknown parameter passed: $1" >&2
            exit 1 
            ;;
    esac
    shift
done
echo "Starting NVIDIA FLARE in DEPLOY mode with the following parameters:"
echo "Clients:    $CLIENTS"
echo "Rounds:     $ROUNDS"
echo "Epochs:     $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Seed:      $SEED"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd ${SCRIPT_DIR}
source env/bin/activate
./stop.sh

docker build -t nvflare-pt-docker . -f Dockerfile

cat <<EOF > project.yml
api_version: 3
name: mednist_project
description: MedNIST project in deploy mode

participants:
  - name: server
    type: server
    org: nvidia
    fed_learn_port: 8002
    admin_port: 8003
EOF

for i in $(seq 1 $CLIENTS); do
cat <<EOF >> project.yml
  - name: site-$i
    type: client
    org: nvidia
EOF
done

cat <<EOF >> project.yml
  - name: admin@nvidia.com
    type: admin
    org: nvidia
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file: master_template.yml
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
  - path: nvflare.lighter.impl.docker.DockerBuilder
    args:
      base_image: nvflare-pt-docker
EOF

nvflare provision -p project.yml

python add_gpu_to_compose.py workspace/mednist_project/prod_00/compose.yaml

export IMAGE_NAME=nvflare-deploy
export PYTHON_EXECUTABLE=python
export WORKSPACE=/workspace
docker compose -f workspace/mednist_project/prod_00/compose.yaml up -d

nvflare config -d workspace/mednist_project/prod_00/admin@nvidia.com/startup

echo "Waiting for NVFlare server to start..."
sleep 15

python job/job.py --batch_size $BATCH_SIZE --num_of_clients $CLIENTS --epochs $EPOCHS --num_of_rounds $ROUNDS --seed $SEED

nvflare job submit -j /tmp/nvflare/jobs/job_config/job_test/
