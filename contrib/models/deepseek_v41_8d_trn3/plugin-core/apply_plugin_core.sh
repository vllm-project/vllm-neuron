#!/usr/bin/env bash
# Apply, check or revert the vllm-neuron plugin-core changes this model needs.
# See README.md in this folder for what the patch contains and why.
#
#   ./apply_plugin_core.sh [--check | --revert] [--target DIR]
#
# DIR is the directory that holds the `vllm_neuron/` package to patch. It defaults to
# the root of the vllm-neuron checkout this script lives in. Pass an installed copy's
# parent (e.g. a venv's site-packages) to patch that copy instead.
#
# Safe to run twice: when the patch is already applied it says so and exits 0.
set -euo pipefail

PATCH_NAME="deepseek-v41-plugin-core.patch"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/$PATCH_NAME"

#: The vllm-project/vllm-neuron `main` commit the patch was generated against
#: (vllm-neuron 0.29.0.1.2.0; its `vllm_neuron/` is byte-identical to the package the
#: DLAMI venv `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` installs).
BASE_COMMIT="fe7bf43c9ae9d0e2731a0765496186c11e6152bd"

usage() { echo "usage: $(basename "$0") [--check | --revert] [--target DIR]" >&2; exit 2; }
die() { echo "error: $*" >&2; exit 1; }

MODE="apply"
TARGET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check)  MODE="check" ;;
        --revert) MODE="revert" ;;
        --target) [ $# -ge 2 ] || usage; TARGET="$2"; shift ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
    shift
done

[ -f "$PATCH" ] || die "$PATCH_NAME is missing from $HERE"

if [ -z "$TARGET" ]; then
    TARGET="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" \
        || die "not inside a git checkout; pass --target DIR (the directory holding vllm_neuron/)"
fi
TARGET="$(cd "$TARGET" && pwd)" || die "no such directory: $TARGET"
[ -d "$TARGET/vllm_neuron" ] || die "$TARGET has no vllm_neuron/ package to patch"
cd "$TARGET"

# Inside a git checkout, `git apply` resolves paths from the repository root, so the
# target has to be that root. Outside one, it behaves like `patch -p1`.
IN_REPO=false
if TOP="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    [ "$TOP" = "$TARGET" ] || die "$TARGET is inside the git checkout $TOP but is not its root"
    IN_REPO=true
fi

if git apply --check "$PATCH" 2>/dev/null; then
    STATE="unapplied"
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
    STATE="applied"
else
    STATE="neither"
fi

echo "target: $TARGET/vllm_neuron"
if $IN_REPO && [ "$STATE" = "neither" ] && git cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null \
    && ! git diff --quiet "$BASE_COMMIT" -- vllm_neuron; then
    echo "note: vllm_neuron/ differs from the patch's base $BASE_COMMIT."
fi

case "$MODE:$STATE" in
    check:unapplied) echo "not applied; would apply cleanly."; exit 0 ;;
    check:applied)   echo "already applied."; exit 0 ;;
    check:neither)
        if $IN_REPO && git apply --3way --check "$PATCH" 2>/dev/null; then
            echo "not applied; would apply only with a three-way merge."
            exit 0
        fi
        die "the patch neither applies nor reverses cleanly: vllm_neuron/ has moved away from $BASE_COMMIT or is partly patched." ;;

    revert:applied)
        git apply --reverse "$PATCH"
        echo "reverted $PATCH_NAME."; exit 0 ;;
    revert:unapplied) echo "not applied; nothing to revert."; exit 0 ;;
    revert:neither)   die "the patch does not reverse cleanly; revert the files by hand (git checkout -- vllm_neuron)." ;;

    apply:applied)    echo "already applied; nothing to do."; exit 0 ;;
    apply:unapplied)
        git apply "$PATCH"
        echo "applied $PATCH_NAME." ;;
    apply:neither)
        $IN_REPO || die "the patch does not apply to $TARGET/vllm_neuron; it was generated against vllm-neuron $BASE_COMMIT (0.29.0.1.2.0)."
        echo "the patch does not apply cleanly; trying a three-way merge ..."
        git apply --3way "$PATCH" || die "three-way merge failed; resolve the conflicts listed by \`git status\`."
        # --3way can leave conflict markers and still exit 0.
        if git ls-files --unmerged | grep -q .; then
            die "applied with conflicts; resolve the files listed by \`git status\`."
        fi
        echo "applied $PATCH_NAME with a three-way merge." ;;
esac

echo
echo "Verify (from the directory that should provide vllm_neuron):"
echo "  python -c \"import vllm_neuron.vllm.spec_decode as s; print(s.__file__, hasattr(s, 'register_proposer'))\""
