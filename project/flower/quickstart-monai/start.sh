#!/bin/bash

if [ -f docker compose.yml ]; then
    docker compose down 
fi

if [ "$#" -eq 0 ]; then
    echo "No number of nodes passed, defaulting to 2 clients."
    python generate_compose.py;
else
    python generate_compose.py $1;
fi

docker compose up -d --build;
flwr run . local-deployment --stream;

