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
echo "Starting Flower with the following parameters:"
echo "Clients:    $CLIENTS"
echo "Rounds:     $ROUNDS"
echo "Epochs:     $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Seed:       $SEED"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd ${SCRIPT_DIR}
source ../env/bin/activate
./stop.sh
python generate_compose.py  --num_of_clients $CLIENTS;
docker compose up -d --build;
flwr run . local-deployment  --run-config batch_size=$BATCH_SIZE --run-config epochs=$EPOCHS --run-config num_of_rounds=$ROUNDS --run-config seed=$SEED;

