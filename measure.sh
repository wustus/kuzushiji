#!/bin/zsh

EPOCHS=30

RUN_FILE="runs/run-$(date -Iminutes)"

for i in {1..10}; do 
    { time bin/kuzushiji > /dev/null; } 2>> "${RUN_FILE}"
done

TIMES=$(awk '{ print $(NF-1) }' "${RUN_FILE}")
# DOES NOT WORK IF TIME < 1min
AVG_TIME=$(echo "${TIMES}" | awk -F: '{ sum += $1*60 + $2 } END { print sum / 10 }')
TIME_PER_EPOCH=$( echo "scale=2; $AVG_TIME / $EPOCHS" | bc -l)

echo "Average Time: ${AVG_TIME}s" >&1 >> "${RUN_FILE}"
echo "Time / Epoch: ${TIME_PER_EPOCH}s" >&1 >> "${RUN_FILE}"
