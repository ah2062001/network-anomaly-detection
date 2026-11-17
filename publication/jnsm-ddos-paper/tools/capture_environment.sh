#!/usr/bin/env bash
set -euo pipefail
# User action; run in the original activated local virtual environment.
python3 --version
python3 -m pip freeze > environment_freeze_local.txt
shasum -a 256 environment_freeze_local.txt
printf 'Created environment_freeze_local.txt. Review before adding it to release.\n'
