#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

set -e

echo "Building JunQi..."
make -C "$SCRIPT_DIR"

ENGINE_BIN="$SCRIPT_DIR/legacy_engine/bin/JunQiEngine"
GUI_DIR="$SCRIPT_DIR/legacy_gui/bin"

if [ ! -x "$ENGINE_BIN" ] || [ ! -x "$GUI_DIR/JunQiGUI" ]; then
    echo "Error: build did not produce the JunQi binaries." >&2
    exit 1
fi

# The legacy GUI talks to two engine seats by default:
#   seat 0 -> UDP 6678, seat 1 -> UDP 5678; both reply to GUI UDP 1234.
# Keep the PIDs local to this script instead of using pkill, which could
# terminate an unrelated JunQi process owned by the user.
ENGINE_PIDS=()
cleanup() {
    for pid in "${ENGINE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

"$ENGINE_BIN" --seat 0 --local-port 6678 --remote-port 1234 &
ENGINE_PIDS+=("$!")
"$ENGINE_BIN" --seat 1 --local-port 5678 --remote-port 1234 &
ENGINE_PIDS+=("$!")

sleep 0.5
echo "Starting GTK client. Close the window to stop both engines."
cd "$GUI_DIR"
./JunQiGUI
