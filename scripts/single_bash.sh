#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

if [ "$#" -ne 2 ]; then
  echo "Usage: sh scripts/single_bash.sh [test-num] [run-num]";
  exit 1;
fi

echo "Running test for test num $1, with run index $2"

python3 src/single_main.py --test-num=$1 --run-num=$2 --human-mode=True