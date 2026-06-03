#!/bin/bash
set -e
while true; do
    if ls outputs/ckpt_sweep_refine/summary.json 2>/dev/null; then
        echo "Summary found, exiting."
        exit 0
    fi
    echo "Waiting..."
    sleep 30
    if [[ $(ps aux | grep -v grep | grep "sweep_eval_ckpts.py" | wc -l) -eq 0 ]]; then
        echo "Process finished, exiting."
        exit 0
    fi
done
