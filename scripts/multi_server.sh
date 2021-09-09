#!/bin/sh

export DISPLAY=localhost:0.0
LANG=en_US

if [ "$#" -eq 1 ]; then
  xvfb-run -a -e /dev/stdout python3 src/multi_main.py --test-num=$1 --platform=server
fi
if [ "$#" -eq 2 ]; then
  xvfb-run -a -e /dev/stdout python3 src/multi_main.py --test-num=$1 --iteration-num=$2 --platform=server
fi
if [ "$#" -gt 2 ]; then
  echo "Usage: sh scripts/multi_server.sh [test_num=1] [optional: iteration_num]"
  exit 1
fi