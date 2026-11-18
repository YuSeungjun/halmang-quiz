#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
if [ ! -s runtime/translations.sqlite3 ]; then
    python3 translation.py
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
exec ../training/envs/voxcpm/bin/python serve.py "$@"
