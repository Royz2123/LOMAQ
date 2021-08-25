#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

test_num=1;
if [ "$#" -eq 1 ]; then
  test_num=$1;
fi
if [ "$#" -gt 1 ]; then
  echo "Usage: run_multi_bash.sh [test_num=1]"
  exit 1
fi
echo "Running test number $test_num"

python3 src/multi_main.py --test-num=$test_num