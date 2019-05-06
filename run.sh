#!/bin/sh

docker stop gcl-container
docker rm gcl-container
docker build . -t gcl:v1
# non-interactive
docker run --name gcl-container gcl:v1 /bin/bash /root/run_examples.py

# attach if necessary: docker  exec -it gcl-container /bin/bash
