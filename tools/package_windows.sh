#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/dist/JunQi-windows}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$ROOT_DIR/legacy_engine/bin/windows/JunQiEngine.exe" "$OUT_DIR/"
cp "$ROOT_DIR/legacy_gui/bin/windows/JunQiGUI.exe" "$OUT_DIR/"
cp "$ROOT_DIR/run_windows.ps1" "$ROOT_DIR/run_windows.bat" "$OUT_DIR/"
cp -R "$ROOT_DIR/legacy_gui/bin/windows/res" "$OUT_DIR/"
cp -R "$ROOT_DIR/legacy_gui/bin/windows/sound" "$OUT_DIR/"

# A fresh MSYS2 installation keeps GTK/MinGW DLLs outside the repository.
# Collect the complete dependency closure so the uploaded ZIP can run from a
# normal PowerShell prompt without requiring users to modify PATH manually.
if command -v ldd >/dev/null 2>&1; then
    queue=("$OUT_DIR/JunQiEngine.exe" "$OUT_DIR/JunQiGUI.exe")
    declare -A seen
    while ((${#queue[@]})); do
        current="${queue[0]}"
        queue=("${queue[@]:1}")
        while read -r dep; do
            [[ -z "$dep" || "$dep" == "not-found" ]] && continue
            case "$dep" in
                /c/Windows/*|/Windows/*|C:/Windows/*|/mingw*/bin/msvcrt.dll|/ucrt*/bin/api-ms-*)
                    continue
                    ;;
            esac
            [[ -f "$dep" ]] || continue
            name="$(basename "$dep")"
            [[ -n "${seen[$name]+x}" ]] && continue
            seen[$name]=1
            cp "$dep" "$OUT_DIR/$name"
            queue+=("$OUT_DIR/$name")
        done < <(ldd "$current" 2>/dev/null | awk '{ if ($3 ~ /^\//) print $3; else if ($1 ~ /^\//) print $1 }')
    done
fi

{
    echo "JunQi Windows client package"
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Engine: JunQiEngine.exe"
    echo "GUI: JunQiGUI.exe"
    echo "Start: run_windows.bat"
    echo
    sha256sum "$OUT_DIR"/* 2>/dev/null || true
} > "$OUT_DIR/manifest.txt"

echo "Packaged Windows client: $OUT_DIR"
