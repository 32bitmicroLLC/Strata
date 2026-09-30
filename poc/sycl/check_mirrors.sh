#!/usr/bin/env bash
# poc/sycl/check_mirrors.sh -- fixture-drift guard for the SYCL PoC.
#
# Phase A rule (plans/sycl-phase-a.md): parity drivers REUSE the main tree's
# CUDA parity fixtures/host references -- copy or factored, never re-derived,
# because touching src/ is out of scope. Copied code must be verbatim and must
# stay verbatim. Convention:
#
#   // SYCL-MIRROR-BEGIN <repo-relative-path>:<first-line>:<last-line> @ <sha>
#   ... verbatim copy ...
#   // SYCL-MIRROR-END
#
# This script re-extracts each original line range and diffs it against the
# mirrored copy. Any drift means the PoC is testing something the source test
# no longer tests -- exactly the silent weakening Phase A forbids.
#
# The worked example lives in drivers/smoke_parity.cpp (mirrors the init
# pattern from poc/sycl/t0/bw_spike.cpp).
set -uo pipefail

POC_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$POC_ROOT/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail=0

check_file() {
  local f="$1"
  local in_block=0 meta="" blockfile=""
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      *SYCL-MIRROR-BEGIN*)
        in_block=1
        meta="${line#*SYCL-MIRROR-BEGIN }"
        blockfile="$tmp/mirror_$$"
        : > "$blockfile"
        ;;
      *SYCL-MIRROR-END*)
        if [ "$in_block" = 1 ]; then
          in_block=0
          # blockfile holds exactly the mirrored lines (markers were never appended)
          cp "$blockfile" "$tmp/mirrored"
          local path rest from to orig
          meta="${meta%%@*}"                       # drop " @ <sha>"
          path="${meta%%:*}"; rest="${meta#*:}"
          from="${rest%%:*}"; to="${rest#*:}"
          from="${from//[^0-9]/}"; to="${to//[^0-9]/}"
          if [ -z "$path" ] || [ -z "$from" ] || [ -z "$to" ]; then
            echo "check_mirrors: unparseable marker in $f: $meta" >&2
            fail=1
            continue
          fi
          orig="$REPO_ROOT/$path"
          if [ ! -f "$orig" ]; then
            echo "check_mirrors: source missing: $orig (marker in $f)" >&2
            fail=1
            continue
          fi
          sed -n "${from},${to}p" "$orig" > "$tmp/original"
          if diff -u "$tmp/original" "$tmp/mirrored" > "$tmp/diff"; then
            echo "check_mirrors: OK   $path:$from-$to  (in ${f#"$POC_ROOT"/})"
          else
            echo "check_mirrors: DRIFT $path:$from-$to  (in ${f#"$POC_ROOT"/})" >&2
            sed -n '1,40p' "$tmp/diff" >&2
            fail=1
          fi
        else
          echo "check_mirrors: unbalanced SYCL-MIRROR-END in $f" >&2
          fail=1
        fi
        ;;
      *)
        if [ "$in_block" = 1 ]; then
          printf '%s\n' "$line" >> "$blockfile"
        fi
        ;;
    esac
  done < "$f"
  if [ "$in_block" = 1 ]; then
    echo "check_mirrors: unbalanced SYCL-MIRROR-BEGIN in $f" >&2
    fail=1
  fi
}

while IFS= read -r f; do
  check_file "$f"
done < <(find "$POC_ROOT" -name '*.cpp' -o -name '*.hpp' | sort)

exit "$fail"
