#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
if [ ! -s runtime/translations.sqlite3 ]; then
    python3 translation.py
fi
server_python="../training/envs/voxcpm/bin/python"
"$server_python" retrieval.py
if [ -z "${HANARABONG_TRANSLATION_MODEL:-}" ] && [ -s runtime/translation-model.txt ]; then
    IFS= read -r HANARABONG_TRANSLATION_MODEL < runtime/translation-model.txt
    export HANARABONG_TRANSLATION_MODEL
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
exec "$server_python" serve.py "$@"
