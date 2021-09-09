#!/bin/sh

export DISPLAY=localhost:0.0

python3 src/main_wandb.py --config=lomaq,qmix --env-config=multi_cart
python3 src/plot.py --config=lomaq,qmix --env-config=multi_cart
#python3 src/plot.py --config=iql_local,lomaq,qmix --env-config=multi_cart

