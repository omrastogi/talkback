#!/usr/bin/env bash
# env runner: activate conda `voice`, cd to project, exec the given command.
set -eo pipefail
export HOME=/home/omras
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate voice
cd /mnt/e/PARCS/server/voice-server
exec "$@"
