#!/usr/bin/env bash
set -euo pipefail
# Example: pass a local path that contains Cricsheet zips or extracted files mounted into the container later
docker compose --profile ops run --rm etl python -m etl.cli ingest-cricsheet /data/cricsheet
