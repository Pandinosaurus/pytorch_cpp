#!/bin/bash

DATA='celebA'

./LDM-v \
    --test true \
    --dataset ${DATA} \
    --test_in_dir "test" \
    --test_out_dir "test" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
