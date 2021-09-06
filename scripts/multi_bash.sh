#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

if [ "$#" -eq 1 ]; then
  python3 src/multi_main.py --test-num=$1 --platform=bash
fi
if [ "$#" -eq 2 ]; then
  python3 src/multi_main.py --test-num=$1 --iteration_num=$2 --platform=bash
fi
if [ "$#" -gt 2 ]; then
  echo "Usage: run_multi_technion.sh [test_num=1] [optional: iteration_num]"
  exit 1
fi


