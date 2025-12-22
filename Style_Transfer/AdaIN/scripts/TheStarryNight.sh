#!/bin/bash

DATA='TheStarryNight'

./AdaIN \
    --generate true \
    --iterations 5000 \
    --dataset ${DATA} \
    --content "content.png" \
    --style "style.png" \
    --gpu_id 0
