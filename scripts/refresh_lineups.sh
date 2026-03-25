#!/usr/bin/env bash
set -euo pipefail
docker compose --profile ops run --rm etl python -m etl.cli refresh-lineups-cmd
