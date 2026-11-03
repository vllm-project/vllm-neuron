#!/usr/bin/env bash
# Apply (or check, or revert) the plugin-core change this model needs for
# speculative decoding. See README.md in this folder for why it exists.
#
#   ./apply_plugin_core.sh [--check | --revert]
#
# Safe to run twice: if the change is already present — because the upstream PR
# merged, or because this script ran before — it reports that and exits 0
# without touching anything.
set -euo pipefail

PATCH_NAME="deepseek-v41-spec-decode.patch"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/$PATCH_NAME"

#: Generated against this commit. Recorded so a failure can say "the tree has
#: moved" rather than only "the patch did not apply".
BASE_COMMIT="f32079d84c2330602e68c58db98e9c28fa0c44cc"

#: The two things the change adds. Both present means applied; one without the
#: other means a half-applied tree, which is worth refusing rather than
#: compounding.
SENTINEL_FILE="vllm_neuron/vllm/patches/in_model_proposer_patch.py"
SENTINEL_SYMBOL="def register_proposer"
SENTINEL_SYMBOL_FILE="vllm_neuron/vllm/spec_decode/__init__.py"

MODE="apply"
case "${1:-}" in
    --check)  MODE="check" ;;
    --revert) MODE="revert" ;;
    "")       ;;
    *) echo "usage: $(basename "$0") [--check | --revert]" >&2; exit 2 ;;
esac

die() { echo "error: $*" >&2; exit 1; }

[ -f "$PATCH" ] || die "$PATCH_NAME is missing from $HERE"

ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" \
    || die "not inside a git checkout; this script applies the patch with \`git apply\`"
cd "$ROOT"

[ -f "$SENTINEL_SYMBOL_FILE" ] || die \
    "$SENTINEL_SYMBOL_FILE not found. Run this from a vllm-neuron checkout."

has_file=false; has_symbol=false
[ -f "$SENTINEL_FILE" ] && has_file=true
grep -q "$SENTINEL_SYMBOL" "$SENTINEL_SYMBOL_FILE" && has_symbol=true

if $has_file && $has_symbol; then
    applied=true
elif ! $has_file && ! $has_symbol; then
    applied=false
else
    die "the tree is half-patched: $SENTINEL_FILE $($has_file && echo exists || echo missing), \
and '$SENTINEL_SYMBOL' $($has_symbol && echo present || echo absent) in \
$SENTINEL_SYMBOL_FILE. Resolve that before running this script."
fi

if [ "$MODE" = "revert" ]; then
    $applied || { echo "not applied; nothing to revert."; exit 0; }
    git apply --reverse "$PATCH" \
        || die "reverse apply failed. If the upstream PR merged, the plugin
  carries these changes permanently and reverting them is not what you want."
    echo "reverted $PATCH_NAME."
    exit 0
fi

if $applied; then
    echo "The proposer registry is already in this tree:"
    echo "  - $SENTINEL_FILE exists"
    echo "  - '$SENTINEL_SYMBOL' is in $SENTINEL_SYMBOL_FILE"
    echo
    echo "Either the upstream change merged or this script has already run."
    echo "Nothing to do; speculative decoding will be available."
    exit 0
fi

echo "The proposer registry is not in this tree, so DSpark speculative decoding"
echo "would be unavailable without $PATCH_NAME."
if [ "$(git rev-parse HEAD)" != "$BASE_COMMIT" ]; then
    echo
    echo "note: the patch was generated against $BASE_COMMIT and HEAD is"
    echo "      $(git rev-parse --short HEAD). A three-way merge will be used if the"
    echo "      context has moved."
fi
echo

if [ "$MODE" = "check" ]; then
    if git apply --check "$PATCH" 2>/dev/null; then
        echo "would apply cleanly."
    elif git apply --3way --check "$PATCH" 2>/dev/null; then
        echo "would apply with a three-way merge."
    else
        die "would NOT apply. The plugin has moved too far from $BASE_COMMIT;
  take the change from the upstream pull request instead of this patch."
    fi
    exit 0
fi

echo "applying ..."
if git apply "$PATCH" 2>/dev/null; then
    echo "applied cleanly."
elif git apply --3way "$PATCH"; then
    # --3way can leave conflict markers and still exit 0, so this is checked
    # rather than assumed.
    if git ls-files --unmerged | grep -q .; then
        echo >&2
        die "applied with conflicts. Resolve the files listed by
  \`git status\`, or take the change from the upstream pull request."
    fi
    echo "applied with a three-way merge."
else
    die "the patch did not apply. The plugin has moved too far from
  $BASE_COMMIT; take the change from the upstream pull request instead."
fi

echo
echo "Done. Verify with:"
echo "  python -c \"from vllm_neuron.vllm.spec_decode import register_proposer; print('ok')\""
