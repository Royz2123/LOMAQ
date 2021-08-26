#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

env_name="multi_particle";
alg_name="local_qmix";

if [ "$#" -eq 2 ]; then
  env_name=$1;
  alg_name=$2;
fi
if [ "$#" -gt 2 ] || [ "$#" -eq 1 ]; then
  echo "Usage: run_multi_bash.sh [env-name] [alg-name]";
  echo "Usage: If no parameters are provided, multi_particle and local_qmix are assumed.";
  exit 1;
fi

echo "Running test for enviroment $env_name, with algorithm $alg_name"

xvfb-run -a -e /dev/stdout python3 src/main.py --env-name=$env_name --alg-name=$alg_name

