#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

test_num=0;
iteration_num=0;
run_num=0;
if [ "$#" -eq 2 ]; then
  test_num=$1;
  run_num=$2;
fi
if [ "$#" -eq 3 ]; then
  test_num=$1;
  iteration_num=$2;
  run_num=$3;
fi
if [ "$#" -gt 3 ] || [ "$#" -lt 2 ]; then
  echo "Usage: sh scripts/single_server.sh [test-num] [optional: iteration_num (default=0)] [run-num]";
  exit 1;
fi

echo "Running test for test num $test_num, with run index $run_num, with iteration num $iteration_num"

xvfb-run -a -e /dev/stdout python3 src/single_main.py --test-num=$test_num --iteration-num=$iteration_num --run-num=$run_num