#!/bin/bash

DATA='TheStarryNight'

./NST \
    --generate true \
    --iterations 1000 \
    --dataset ${DATA} \
    --content "content.png" \
    --style "style.png" \
    --gpu_id 0
