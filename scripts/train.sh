#!/usr/bin/env bash
set -euo pipefail
docker compose --profile ml run --rm trainer python -m ml.train
