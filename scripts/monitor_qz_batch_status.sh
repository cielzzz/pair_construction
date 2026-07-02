#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/monitor_qz_batch_status.sh <submitted_jobs.tsv>" >&2
  exit 1
fi

SUMMARY_PATH="$1"
QZCLI_BIN="${QZCLI_BIN:-/usr/local/bin/qzcli}"
QUERY_SLEEP_SEC="${QUERY_SLEEP_SEC:-1}"

if [ ! -f "$SUMMARY_PATH" ]; then
  echo "submitted_jobs.tsv not found: $SUMMARY_PATH" >&2
  exit 1
fi

tmp_jobs=$(mktemp)
trap 'rm -f "$tmp_jobs"' EXIT

awk -F '\t' '{ latest[$1] = $0 } END { for (name in latest) print latest[name] }' "$SUMMARY_PATH" | sort > "$tmp_jobs"

printf 'timestamp_utc\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'job_name\tjob_id\tstatus\tcompute_group\n'

while IFS=$'\t' read -r job_name job_id compute_group _rest; do
  [ -n "$job_name" ] || continue
  output="$("$QZCLI_BIN" status --json "$job_id" 2>&1 || true)"
  status=$(printf '%s\n' "$output" | sed -n 's/.*Status: \(.*\)$/\1/p' | head -n 1)
  if [ -z "$status" ]; then
    status=$(printf '%s\n' "$output" | sed -n 's/.*"status": "\([^"]*\)".*/\1/p' | head -n 1)
  fi
  if [ -z "$status" ]; then
    status=$(printf '%s\n' "$output" | tail -n 1 | tr '\t' ' ')
  fi
  status=${status%%│*}
  status=$(printf '%s' "$status" | sed 's/[[:space:]]*$//')
  printf '%s\t%s\t%s\t%s\n' "$job_name" "$job_id" "$status" "$compute_group"
  sleep "$QUERY_SLEEP_SEC"
done < "$tmp_jobs"
