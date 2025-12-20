#!/bin/bash

DATA='Neckarfront_TheStarryNight'

./NST \
    --generate true \
    --iterations 1000 \
    --dataset ${DATA} \
    --content "content.png" \
    --style "style.png" \
    --gpu_id 0
