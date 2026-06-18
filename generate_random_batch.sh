# Генерация N случайных винтов (STEP) через FreeCADCmd.
# Пример: SCREW_RANDOM_COUNT=1000 ./generate_random_batch.sh

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export SCREW_RANDOM_COUNT="${SCREW_RANDOM_COUNT:-1000}"
export SCREW_RANDOM_SEED="${SCREW_RANDOM_SEED:-}"
export SCREW_OUT="${SCREW_OUT:-$DIR/models/generated_random_${SCREW_RANDOM_COUNT}}"

echo "[INFO] SCREW_RANDOM_COUNT=$SCREW_RANDOM_COUNT seed=${SCREW_RANDOM_SEED:-<none>} out=$SCREW_OUT"
exec freecadcmd "$DIR/generate_screws.py"
