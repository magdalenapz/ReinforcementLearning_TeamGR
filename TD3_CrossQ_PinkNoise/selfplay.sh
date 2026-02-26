#!/bin/bash

name=heat=$1
cp base_policy_buffer models/$name
while true; do
    python -u main.py --env Hockey-SelfPlay --policy CrossQ --heat $1 --policy_buffer $name --pink_noise --eval_freq 2000000 #--max_timesteps 300000
done