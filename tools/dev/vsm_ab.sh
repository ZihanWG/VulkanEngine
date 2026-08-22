#!/usr/bin/env bash
#
# Virtual-shadow-map A/B: cascades versus the page pool, on a frame that
# reproduces byte for byte.
#
# This is deliberately NOT a CI job. Pixel determinism holds on MoltenVK and
# does not hold on lavapipe, where the same commit rendered three times produced
# 0, then 433, then 0 differing pixels (docs/headless_ci.md). A gate that flaky
# is worse than no gate, so this runs on the development machine and reports.
#
# The control is the point. Every configuration is captured twice and the two
# captures must be byte identical before any cascade-versus-VSM number is
# believed -- without that, a difference could just as easily be the animated
# light swarm at a different phase.
#
# Usage: tools/dev/vsm_ab.sh [output-directory] [frame]

set -euo pipefail

OUTPUT_DIR="${1:-build/measurements/vsm_ab}"
CAPTURE_FRAME="${2:-60}"
EXIT_FRAME=$((CAPTURE_FRAME + 30))

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENGINE="build/ci-debug/VulkanEngine"
COMPARE="build/ci-debug/compare_images"

if [[ ! -x "$ENGINE" ]]; then
    echo "error: $ENGINE not found. Run tools/dev/verify_renderer.sh fast first." >&2
    exit 1
fi
if [[ ! -x "$COMPARE" ]]; then
    echo "error: $COMPARE not found. Build it with:" >&2
    echo "  cmake --build --preset ci-debug --target compare_images" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --vsm overrides the three stage toggles for one run and writes nothing, so
# this never disturbs config/runtime_settings.json.
capture() {
    local mode="$1" out="$2" log="$3"
    "$ENGINE" --deterministic --vsm "$mode" \
        --exit-after-frames "$EXIT_FRAME" \
        --capture-frame "$CAPTURE_FRAME" \
        --capture-output "$out" >"$log" 2>&1
}

control() {
    local mode="$1" label="$2"
    local first="$OUTPUT_DIR/${label}_1.png" second="$OUTPUT_DIR/${label}_2.png"
    capture "$mode" "$first" "$OUTPUT_DIR/${label}_1.log"
    capture "$mode" "$second" "$OUTPUT_DIR/${label}_2.log"
    if "$COMPARE" "$first" "$second" >/dev/null; then
        echo "control  $label: two captures of the same configuration are identical"
    else
        echo "control  $label: FAILED -- the same configuration did not reproduce." >&2
        echo "         Nothing below this line means anything until that is fixed." >&2
        "$COMPARE" "$first" "$second" >&2 || true
        exit 1
    fi
}

echo "VSM A/B at frame $CAPTURE_FRAME -> $OUTPUT_DIR"
control off cascades
control shadows vsm

echo
echo "cascades vs VSM, at three tolerances:"
for tolerance in 0 1 3; do
    printf '  tolerance %d: ' "$tolerance"
    "$COMPARE" "$OUTPUT_DIR/vsm_1.png" "$OUTPUT_DIR/cascades_1.png" \
        --channel-tolerance "$tolerance" \
        --diff-output "$OUTPUT_DIR/diff_tolerance${tolerance}.png" 2>/dev/null |
        head -1 || true
done

echo
echo "Read the tolerance-3 diff, not the tolerance-0 one: a shadow change moves"
echo "scene luminance, auto-exposure follows it, and the whole frame lands one"
echo "quantization step away. Those pixels are real and mean nothing."
echo
grep -E "requested pages|resident:|directional shadows" "$OUTPUT_DIR/vsm_1.log" || true
