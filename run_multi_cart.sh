export DISPLAY=localhost:0.0

python3 src/main.py --config=local_qmix,qmix,iql --env-config=multi_cart
python3 src/plot.py -env multi_cart