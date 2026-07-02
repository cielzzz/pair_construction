#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


QZCLI_TOOL_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/xyzhang/qzcli_tool"
)
QZCLI_BIN = Path("/usr/local/bin/qzcli")

if str(QZCLI_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(QZCLI_TOOL_ROOT))

from qzcli.api import QzAPIError, get_api  # type: ignore  # noqa: E402
from qzcli.config import (  # type: ignore  # noqa: E402
    find_workspace_by_name,
    get_cookie,
    list_cached_workspaces,
)


def bypass_proxy_env() -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        os.environ.pop(key, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query running high-priority jobs in the shared H200 pool. "
            "Defaults: compute-group='共享H200资源池', priority='HIGH', status='job_running'."
        )
    )
    parser.add_argument(
        "--node",
        default=None,
        help="Filter by exact node name, e.g. qb-prod-gpu1761.",
    )
    parser.add_argument(
        "--compute-group",
        default="共享H200资源池",
        help="Compute group name filter. Default: 共享H200资源池.",
    )
    parser.add_argument(
        "--priority",
        default="HIGH",
        help="Priority level filter. Default: HIGH.",
    )
    parser.add_argument(
        "--status",
        default="job_running",
        help="Job status filter. Default: job_running.",
    )
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help=(
            "Workspace name or ID. Repeatable. "
            "Default: search all cached workspaces, or all live-accessible workspaces "
            "when --live-workspaces is enabled."
        ),
    )
    parser.add_argument(
        "--live-workspaces",
        action="store_true",
        help=(
            "Discover workspaces live from the platform instead of using only the local "
            "cache. Useful when you want all currently accessible workspaces."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Page size for the job list API. Default: 200.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a plain-text summary.",
    )
    parser.add_argument(
        "--no-auto-login",
        action="store_true",
        help="Do not auto-refresh the qzcli cookie with `qzcli login`.",
    )
    return parser.parse_args()


def cached_workspace_options() -> list[dict[str, str]]:
    return [
        {"id": str(item["id"]), "name": str(item.get("name", "") or item["id"])}
        for item in list_cached_workspaces()
        if item.get("id")
    ]


def fetch_live_workspaces(cookie: str) -> list[dict[str, str]]:
    api = get_api()
    live = []
    seen: set[str] = set()
    for item in api.list_workspaces(cookie):
        workspace_id = str(item.get("id", "") or "")
        if not workspace_id or workspace_id in seen:
            continue
        seen.add(workspace_id)
        live.append(
            {
                "id": workspace_id,
                "name": str(item.get("name", "") or workspace_id),
            }
        )
    return live


def resolve_workspaces(
    workspace_args: list[str],
    *,
    workspace_options: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not workspace_args:
        return workspace_options

    if not workspace_options:
        raise SystemExit("No workspace options available to resolve from.")

    by_id = {item["id"]: item for item in workspace_options}
    by_name = {item["name"]: item for item in workspace_options if item.get("name")}

    if not workspace_args:
        return [
            {"id": str(item["id"]), "name": str(item.get("name", "") or item["id"])}
            for item in workspace_options
            if item.get("id")
        ]

    resolved: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw in workspace_args:
        workspace_id = ""
        workspace_name = raw
        if raw.startswith("ws-"):
            workspace_id = raw
            workspace_name = by_id.get(workspace_id, {}).get("name", raw)
        else:
            matched = by_name.get(raw)
            if matched is None:
                workspace_id = find_workspace_by_name(raw) or ""
                matched = by_id.get(workspace_id) if workspace_id else None
            if matched is None:
                for item in workspace_options:
                    if raw.lower() in item["name"].lower():
                        matched = item
                        break
            if matched is None:
                raise SystemExit(f"Workspace not found: {raw}")
            workspace_id = matched["id"]
            workspace_name = matched["name"]

        if workspace_id in seen:
            continue
        seen.add(workspace_id)
        resolved.append({"id": workspace_id, "name": workspace_name})

    return resolved


def fetch_workspace_jobs(
    workspace_id: str,
    cookie: str,
    page_size: int,
) -> list[dict[str, Any]]:
    api = get_api()
    page_num = 1
    jobs: list[dict[str, Any]] = []

    while True:
        result = api.list_jobs_with_cookie(
            workspace_id,
            cookie,
            page_num=page_num,
            page_size=page_size,
        )
        batch = result.get("jobs", [])
        if not isinstance(batch, list):
            break
        jobs.extend(batch)

        total = result.get("total")
        if len(batch) < page_size:
            break
        if isinstance(total, int) and len(jobs) >= total:
            break
        page_num += 1

    return jobs


def is_cookie_error(exc: Exception) -> bool:
    message = str(exc)
    return "Cookie 已过期" in message or "Cookie 已过期或无效" in message or "无效" in message


def refresh_cookie_via_qzcli_login() -> None:
    completed = subprocess.run(
        [str(QZCLI_BIN), "login"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "qzcli login failed"
        raise RuntimeError(detail)


def job_nodes(job: dict[str, Any]) -> list[str]:
    nodes = []
    for item in job.get("node_infos", []) or []:
        if isinstance(item, dict) and item.get("node_name"):
            nodes.append(str(item["node_name"]))
    return nodes


def filter_jobs(
    jobs: list[dict[str, Any]],
    *,
    workspace_name: str,
    compute_group: str,
    priority: str,
    status: str,
    node: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for job in jobs:
        if job.get("status") != status:
            continue
        if job.get("priority_level") != priority:
            continue
        if job.get("logic_compute_group_name") != compute_group:
            continue

        nodes = job_nodes(job)
        if node and node not in nodes:
            continue

        rows.append(
            {
                "job_id": job.get("job_id", ""),
                "name": job.get("name", ""),
                "workspace_id": job.get("workspace_id", ""),
                "workspace_name": workspace_name,
                "status": job.get("status", ""),
                "priority_level": job.get("priority_level", ""),
                "compute_group_name": job.get("logic_compute_group_name", ""),
                "gpu_count": job.get("gpu_count", 0),
                "instance_count": job.get("node_count", job.get("instance_count", 0)),
                "created_at": job.get("created_at", ""),
                "running_time_ms": job.get("running_time_ms", ""),
                "nodes": nodes,
                "url": (
                    f"https://qz.sii.edu.cn/jobs/distributedTrainingDetail/"
                    f"{job.get('job_id')}?spaceId={job.get('workspace_id', '')}"
                ),
            }
        )

    rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return rows


def print_plain(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No matching jobs.")
        return

    print(f"Matched {len(rows)} job(s)")
    print("")
    for idx, row in enumerate(rows, start=1):
        gpu_count = row.get("gpu_count", 0)
        instance_count = row.get("instance_count", 0)
        nodes = ", ".join(row.get("nodes", []))
        print(f"[{idx}] {row['name']}")
        print(f"    job_id: {row['job_id']}")
        print(f"    workspace: {row['workspace_name']} ({row['workspace_id']})")
        print(
            "    status/priority/group: "
            f"{row['status']} / {row['priority_level']} / {row['compute_group_name']}"
        )
        print(f"    resources: {gpu_count} GPU, {instance_count} node(s)")
        print(f"    nodes: {nodes or '-'}")
        print(f"    created_at: {row['created_at']}")
        print(f"    url: {row['url']}")
        print("")


def main() -> int:
    args = parse_args()
    bypass_proxy_env()
    cookie_data = get_cookie() or {}
    cookie = str(cookie_data.get("cookie", "") or "")
    login_refreshed = False
    if not cookie:
        if args.no_auto_login:
            print("Missing qzcli cookie. Run: qzcli login", file=sys.stderr)
            return 1
        refresh_cookie_via_qzcli_login()
        login_refreshed = True
        cookie_data = get_cookie() or {}
        cookie = str(cookie_data.get("cookie", "") or "")
        if not cookie:
            print("qzcli login completed but no cookie was saved.", file=sys.stderr)
            return 1

    workspace_options = cached_workspace_options()
    if args.live_workspaces:
        try:
            workspace_options = fetch_live_workspaces(cookie)
        except (QzAPIError, Exception) as exc:
            if not args.no_auto_login and not login_refreshed and is_cookie_error(exc):
                refresh_cookie_via_qzcli_login()
                login_refreshed = True
                cookie_data = get_cookie() or {}
                cookie = str(cookie_data.get("cookie", "") or "")
                workspace_options = fetch_live_workspaces(cookie)
            else:
                print(f"Failed to fetch live workspaces: {exc}", file=sys.stderr)
                return 1

    workspaces = resolve_workspaces(
        args.workspace,
        workspace_options=workspace_options,
    )
    if not workspaces:
        if args.live_workspaces:
            print(
                "No accessible workspaces found from the live API. Run: qzcli login",
                file=sys.stderr,
            )
        else:
            print("No cached workspaces found. Run: qzcli res -u", file=sys.stderr)
        return 1

    matched_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for workspace in workspaces:
        try:
            jobs = fetch_workspace_jobs(
                workspace_id=workspace["id"],
                cookie=cookie,
                page_size=args.page_size,
            )
        except (QzAPIError, Exception) as exc:
            if not args.no_auto_login and not login_refreshed and is_cookie_error(exc):
                refresh_cookie_via_qzcli_login()
                login_refreshed = True
                cookie_data = get_cookie() or {}
                cookie = str(cookie_data.get("cookie", "") or "")
                try:
                    jobs = fetch_workspace_jobs(
                        workspace_id=workspace["id"],
                        cookie=cookie,
                        page_size=args.page_size,
                    )
                except (QzAPIError, Exception) as retry_exc:
                    errors.append(
                        f"{workspace['name']} ({workspace['id']}): {retry_exc}"
                    )
                    continue
            else:
                errors.append(f"{workspace['name']} ({workspace['id']}): {exc}")
                continue

        matched_rows.extend(
            filter_jobs(
                jobs,
                workspace_name=workspace["name"],
                compute_group=args.compute_group,
                priority=args.priority,
                status=args.status,
                node=args.node,
            )
        )

    if args.json:
        payload = {
            "filters": {
                "node": args.node,
                "compute_group": args.compute_group,
                "priority": args.priority,
                "status": args.status,
                "live_workspaces": args.live_workspaces,
                "workspaces": workspaces,
            },
            "jobs": matched_rows,
            "errors": errors,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_plain(matched_rows)
        if errors:
            print("Workspace query errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)

    return 0 if matched_rows or not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
