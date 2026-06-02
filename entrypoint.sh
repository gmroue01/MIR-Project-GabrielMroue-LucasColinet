#!/bin/sh
set -e

export HF_HOME=/app/data
export TRANSFORMERS_CACHE=/app/data

DATA_DIR="/app/data"
SENTINEL="$DATA_DIR/.download_complete"

if [ ! -f "$SENTINEL" ]; then
    echo "First boot — downloading data from HuggingFace..."
    for i in 1 2 3 4 5; do
        echo "Attempt $i..."
        timeout 600 python -c "
import os
os.environ['HF_HOME'] = '/app/data'
from huggingface_hub import snapshot_download
snapshot_download(
    'GabrielMroue/MirProject',
    local_dir='/app/data',
    repo_type='dataset',
    token=os.environ.get('HF_TOKEN'),
    max_workers=2
)
" && touch "$SENTINEL" && break || echo "Attempt $i failed, retrying..."
        sleep 10
    done
else
    echo "Data already present at /app/data, skipping download."
fi

echo "Creating symlinks..."
ln -sfn /app/data/Flickr8k      /app/Flickr8K      2>/dev/null || true
ln -sfn /app/data/Flickr8k      /app/Flickr8k      2>/dev/null || true
ln -sfn /app/data/indexes_faiss  /app/indexes_faiss 2>/dev/null || true
ln -sfn /app/data/dataset        /app/dataset       2>/dev/null || true

echo "Syncing indexes from image to volume..."
INDEX_VOL="$DATA_DIR/indexes"
mkdir -p "$INDEX_VOL"
for f in /app/indexes_image/*; do
    [ -f "$f" ] || continue
    fn=$(basename "$f")
    # Always overwrite JSON/metadata files; only copy binary indexes if missing
    case "$fn" in
        *.json|*.npy)
            echo "  ~ $fn (overwrite)"
            cp "$f" "$INDEX_VOL/$fn"
            ;;
        *)
            if [ ! -f "$INDEX_VOL/$fn" ]; then
                echo "  + $fn"
                cp "$f" "$INDEX_VOL/$fn"
            fi
            ;;
    esac
done
ln -sfn "$INDEX_VOL" /app/indexes 2>/dev/null || true

echo "Starting app..."
exec python run.py