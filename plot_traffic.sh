export DISPLAY=localhost:0.0

export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH

python3 src/plot.py -env traffic
