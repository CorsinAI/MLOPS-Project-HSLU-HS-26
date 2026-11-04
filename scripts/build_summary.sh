#!/usr/bin/env bash
# Builds docs/ms2_summary.pdf from docs/ms2_summary.md.
# Needs pandoc and a Chromium browser. Set BROWSER to use a specific one.
set -euo pipefail

cd "$(dirname "$0")/../docs"

browser="${BROWSER:-}"
if [ -z "$browser" ]; then
    for candidate in \
        "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
        "/c/Program Files/Google/Chrome/Application/chrome.exe" \
        google-chrome chromium; do
        if command -v "$candidate" >/dev/null 2>&1; then
            browser="$candidate"
            break
        fi
    done
fi
if [ -z "$browser" ]; then
    echo "No browser found. Set BROWSER." >&2
    exit 1
fi

page="ms2_summary.html"
trap 'rm -f "$page"' EXIT
pandoc ms2_summary.md --standalone --columns=1000 --css ms2_summary.css \
    --metadata pagetitle="NegativeHours MS2 summary" --output "$page"

if command -v cygpath >/dev/null 2>&1; then
    here="$(cygpath -m "$PWD")"
    url="file:///$here/$page"
    output="$here/ms2_summary.pdf"
else
    url="file://$PWD/$page"
    output="$PWD/ms2_summary.pdf"
fi

"$browser" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$output" "$url"
