#!/bin/bash

docker build -t fbm-node:6.2.0 -f dockerFileNode .
docker build -t fbm-researcher:6.2.0 -f dockerFileResearcher .