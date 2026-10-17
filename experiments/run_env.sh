#!/usr/bin/env bash
# Unified runtime environment for SHREC2025 dual-version experiments.
# Usage:
#   source experiments/run_env.sh        # then run commands
#   experiments/run_env.sh python3 ...   # or exec directly
#
# Why: system /usr/local/cuda (11.8) carries cuDNN 8.6 in LD_LIBRARY_PATH,
# incompatible with torch 2.3.0+cu121 (built against cuDNN 8.9.2). Stripping
# LD_LIBRARY_PATH makes the loader use the bundled nvidia/cudnn pip package.

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -E '^/usr/local/cuda(/lib64)?$' | paste -sd:)"
  [ -z "$LD_LIBRARY_PATH" ] && unset LD_LIBRARY_PATH
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
# Avoid thread oversubscription (data loaders spawn their own workers)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

if [ "$#" -gt 0 ]; then
  exec "$@"
fi
