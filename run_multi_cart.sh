export DISPLAY=localhost:0.0

python3 src/main.py --config=iql,qmix,local_qmix --env-config=multi_cart
python3 src/plot.py -env multi_cart
