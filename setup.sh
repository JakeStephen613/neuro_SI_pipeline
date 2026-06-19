#!/bin/bash
# One-time environment setup for neuro_SI_pipeline.
#
# Required env vars:
#   GRAPHMERT_UMLS_ROOT  - path to graphmert_umls repo checkout
#
# Usage:
#   export GRAPHMERT_UMLS_ROOT=/path/to/graphmert_umls
#   bash setup.sh

set -euo pipefail

: "${GRAPHMERT_UMLS_ROOT:?GRAPHMERT_UMLS_ROOT must be set (path to graphmert_umls repo checkout)}"

echo "==> Installing graphrag (editable)..."
uv pip install -e "${GRAPHMERT_UMLS_ROOT}/graphrag"

echo ""
echo "Setup complete. graphrag will stay in sync with git pull in ${GRAPHMERT_UMLS_ROOT}."
echo "The GraphMERT model code (2_graphmert/graphmert_model/) is vendored directly in this repo."
