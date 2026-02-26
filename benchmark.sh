#!/bin/bash

i=30
while true; do
    python main_with_random_start.py --env Hockey-Strong --policy $1 --eval_freq 50000 --eval_episodes 100 --iteration $i #--pink_noise
    ((i++))
done
