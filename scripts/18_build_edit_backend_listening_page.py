#!/usr/bin/env python3
"""Build a static listening page for speech-editing benchmark results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_path(path: str | Path | None, base_dir: Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def copy_file(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.name)


def safe_name(value: str | None) -> str:
    keep = []
    for ch in value or "item":
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(
    results_path: Path,
    summary_path: Path,
    generation_summary_path: Path,
    cases_path: Path,
    page_dir: Path,
    title: str,
    backend_label: str,
    base_dir: Path,
) -> dict[str, Any]:
    results = list(iter_jsonl(results_path))
    summary_obj = load_json(summary_path)
    page_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = page_dir / "assets"
    data_dir = page_dir / "data"
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for row in results:
        prefix = f"{int(row['index']):02d}_{safe_name(row.get('task'))}_{safe_name(row.get('language'))}_{safe_name(row.get('file_name'))}"
        source_path = resolve_path(row.get("source_wav"), base_dir)
        target_path = resolve_path(row.get("output_wav"), base_dir)
        source_asset = copy_file(source_path, assets_dir / f"{prefix}_source.wav")
        target_asset = copy_file(target_path, assets_dir / f"{prefix}_target.wav")
        run_metrics_link = None
        run_metrics_path = resolve_path(row.get("run_metrics_json"), base_dir)
        if run_metrics_path and run_metrics_path.exists():
            dst = data_dir / "run_metrics" / safe_name(row.get("task")) / safe_name(row.get("language")) / run_metrics_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_metrics_path, dst)
            run_metrics_link = str(dst.relative_to(page_dir))
        sample = dict(row)
        sample["source_audio"] = f"assets/{source_asset}" if source_asset else None
        sample["target_audio"] = f"assets/{target_asset}" if target_asset else None
        sample["source_path"] = row.get("source_wav")
        sample["target_path"] = row.get("output_wav")
        sample["run_metrics_json"] = run_metrics_link
        samples.append(sample)

    data_files = {
        "results_jsonl": results_path,
        "results_csv": results_path.with_suffix(".csv"),
        "summary_json": summary_path,
        "report_md": summary_path.with_name("report.md"),
        "asr_manifest_jsonl": summary_path.with_name("asr_manifest.jsonl"),
        "asr_results_jsonl": summary_path.with_name("asr_results.jsonl"),
        "wavlm_sim_jsonl": summary_path.with_name("wavlm_sim.jsonl"),
        "generation_summary_jsonl": generation_summary_path,
        "cases_jsonl": cases_path,
    }
    downloads = {}
    for key, src in data_files.items():
        if src.exists():
            dst = data_dir / src.name
            shutil.copy2(src, dst)
            downloads[key] = str(dst.relative_to(page_dir))

    return {
        "title": title,
        "backend": backend_label,
        "thresholds": summary_obj.get("thresholds", {}),
        "totals": summary_obj.get("totals", {}),
        "summary": summary_obj.get("summary", []),
        "downloads": downloads,
        "samples": samples,
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <script src="samples.js"></script>
  <style>
    :root { color-scheme: light; --bg:#f5f7f8; --panel:#fff; --ink:#121820; --muted:#58636d; --line:#d8dee4; --soft:#eef2f4; --ok:#0f766e; --bad:#b91c1c; --warn:#b45309; --accent:#1d4ed8; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { position:sticky; top:0; z-index:5; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:saturate(160%) blur(10px); }
    .bar { max-width:1540px; margin:0 auto; padding:18px 24px; display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    h1 { margin:0; font-size:22px; letter-spacing:0; line-height:1.2; }
    .sub { color:var(--muted); font-size:13px; margin-top:5px; line-height:1.45; }
    .controls { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; max-width:760px; }
    button, .download { height:34px; border:1px solid var(--line); background:white; color:var(--ink); border-radius:8px; padding:0 12px; font-weight:700; cursor:pointer; text-decoration:none; display:inline-flex; align-items:center; }
    button.active { background:#172026; color:white; border-color:#172026; }
    main { max-width:1540px; margin:0 auto; padding:18px 24px 56px; }
    .downloads { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 14px; }
    .summary { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; margin-bottom:16px; }
    .summary-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
    .summary-card strong { display:block; font-size:13px; margin-bottom:6px; }
    .summary-card span { display:block; color:var(--muted); font-size:12px; line-height:1.45; }
    .sample { background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:16px; overflow:hidden; }
    .sample-head { padding:14px 16px; background:#fbfcfd; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    .title { display:flex; align-items:center; flex-wrap:wrap; gap:8px; min-width:0; }
    .idx { font-weight:850; }
    .pill { min-height:24px; display:inline-flex; align-items:center; border-radius:999px; background:var(--soft); color:var(--muted); font-size:12px; font-weight:800; padding:2px 9px; white-space:nowrap; }
    .pill.zh { color:#0f766e; background:#dff4ef; } .pill.en { color:#7c2d12; background:#f7e5d8; }
    .pill.pass { color:var(--ok); background:#dcfce7; } .pill.fail { color:var(--bad); background:#fee2e2; } .pill.warn { color:var(--warn); background:#ffedd5; }
    .ids { color:var(--muted); font-size:12px; line-height:1.45; text-align:right; overflow-wrap:anywhere; }
    .body-grid { display:grid; grid-template-columns:360px 1fr 1fr; gap:0; }
    .panel { padding:14px; border-right:1px solid var(--line); min-width:0; }
    .panel:last-child { border-right:0; }
    .label { font-size:12px; font-weight:850; color:#26323b; margin-bottom:8px; display:flex; justify-content:space-between; gap:8px; }
    audio { width:100%; height:34px; display:block; margin-bottom:10px; }
    .text { border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:white; font-size:13px; line-height:1.5; white-space:pre-wrap; overflow-wrap:anywhere; max-height:150px; overflow:auto; }
    .metric-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; margin-top:10px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:8px; background:#fbfcfd; min-width:0; }
    .metric b { display:block; font-size:12px; margin-bottom:3px; }
    .metric span { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    details { margin-top:10px; }
    summary { cursor:pointer; color:var(--muted); font-size:12px; font-weight:750; }
    code { display:block; margin-top:6px; color:#28323a; background:#f1f4f6; border:1px solid var(--line); border-radius:6px; padding:7px; font-size:11px; line-height:1.35; white-space:pre-wrap; overflow-wrap:anywhere; }
    @media (max-width: 1180px) { .summary { grid-template-columns:repeat(2, minmax(0, 1fr)); } .body-grid { grid-template-columns:1fr; } .panel { border-right:0; border-bottom:1px solid var(--line); } .panel:last-child { border-bottom:0; } }
    @media (max-width: 720px) { .bar,.sample-head { flex-direction:column; } .controls { justify-content:flex-start; } .ids { text-align:left; } .summary { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>__TITLE__</h1>
        <div class="sub">40 official benchmark samples: deletion / insertion / substitution / multi_substitution, zh/en each 5. Strict text pass is CER/WER <= 0.2.</div>
      </div>
      <div class="controls" id="filters"></div>
    </div>
  </header>
  <main>
    <div class="downloads" id="downloads"></div>
    <section class="summary" id="summary"></section>
    <section id="app"></section>
  </main>
<script>
const payload = window.EDIT_BACKEND_SAMPLES || {samples: [], summary: [], downloads: {}, thresholds: {}, totals: {}};
const filterButtons = [
  ["all", "All"], ["pass", "CER/WER <= 0.2"], ["fail", "CER/WER > 0.2"], ["warn", "Audio warn"],
  ["zh", "ZH"], ["en", "EN"],
  ["task:deletion", "Deletion"], ["task:insertion", "Insertion"], ["task:substitution", "Substitution"], ["task:multi_substitution", "Multi Sub"]
];
function esc(value) { return String(value ?? "").replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[ch] || ch)); }
function fmt(value, digits = 3) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "N/A"; }
function renderDownloads() {
  const labels = {report_md:"report.md", results_csv:"results.csv", results_jsonl:"results.jsonl", summary_json:"summary.json", cases_jsonl:"cases.jsonl", asr_results_jsonl:"asr_results.jsonl", wavlm_sim_jsonl:"wavlm_sim.jsonl", generation_summary_jsonl:"generation_summary.jsonl"};
  document.getElementById("downloads").innerHTML = Object.entries(labels).filter(([key]) => payload.downloads[key]).map(([key,label]) => `<a class="download" href="${esc(payload.downloads[key])}">${esc(label)}</a>`).join("");
}
function renderSummary() {
  const total = payload.totals || {};
  const totalCard = `<div class="summary-card"><strong>Total / ${esc(payload.backend)}</strong><span>N=${total.n ?? 0}, generated=${total.gen_ok ?? 0}, strict pass=${total.strict_text_pass ?? 0}</span><span>audio warn=${total.audio_warn ?? 0}, avg CER/WER=${fmt(total.avg_text_error,4)}, avg sim=${fmt(total.avg_speaker_sim_wavlm,4)}</span></div>`;
  const cards = payload.summary.map(s => `<div class="summary-card"><strong>${esc(s.task)} / ${esc(s.language)}</strong><span>N=${s.n}, generated=${s.gen_ok}, strict pass=${s.strict_text_pass}, exact=${s.text_exact}</span><span>avg CER/WER=${fmt(s.avg_text_error,4)}, avg sim=${fmt(s.avg_speaker_sim_wavlm,4)}</span></div>`).join("");
  document.getElementById("summary").innerHTML = totalCard + cards;
}
function passesFilter(s, filter) {
  if (filter === "all") return true;
  if (filter === "pass") return s.strict_text_pass;
  if (filter === "fail") return !s.strict_text_pass;
  if (filter === "warn") return s.audio_warn;
  if (filter === "zh" || filter === "en") return s.language === filter;
  if (filter.startsWith("task:")) return s.task === filter.slice(5);
  return true;
}
function renderFilters(active) {
  const el = document.getElementById("filters");
  el.innerHTML = filterButtons.map(([key,label]) => `<button data-filter="${esc(key)}" class="${active === key ? "active" : ""}">${esc(label)}</button>`).join("");
  el.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => render(btn.dataset.filter)));
}
function render(filter = "all") {
  renderFilters(filter);
  const samples = payload.samples.filter(s => passesFilter(s, filter));
  document.getElementById("app").innerHTML = samples.map(s => {
    const textClass = s.strict_text_pass ? "pass" : "fail";
    const warn = s.audio_warn ? `<span class="pill warn">audio warn</span>` : "";
    const metricJson = s.run_metrics_json ? `<a href="${esc(s.run_metrics_json)}">run json</a>` : "N/A";
    return `<article class="sample">
      <div class="sample-head">
        <div class="title"><span class="idx">#${s.index}</span><span class="pill ${s.language}">${esc(String(s.language).toUpperCase())}</span><span class="pill">${esc(s.task)}</span><span class="pill ${textClass}">${s.strict_text_pass ? "CER/WER pass" : "CER/WER fail"}</span>${warn}</div>
        <div class="ids">${esc(s.case_id)}<br>${esc(s.file_name)}</div>
      </div>
      <div class="body-grid">
        <section class="panel">
          <div class="label"><span>Instruction</span></div><div class="text">${esc(s.instruction)}</div>
          <div class="metric-grid">
            <div class="metric"><b>${esc(s.text_metric)}</b><span>${fmt(s.text_error,4)} · exact=${s.text_exact ? "Y" : "N"}</span></div>
            <div class="metric"><b>WavLM sim</b><span>${fmt(s.speaker_sim_wavlm,4)}</span></div>
            <div class="metric"><b>Duration ratio</b><span>${fmt(s.duration_ratio,2)}${s.duration_warn ? " · warn" : ""}</span></div>
            <div class="metric"><b>Tail silence</b><span>${fmt(s.target_audio_stats.tail_silence_sec,2)}s / ${fmt(s.target_audio_stats.tail_silence_ratio,2)}${s.tail_silence_warn ? " · warn" : ""}</span></div>
          </div>
          <details><summary>JSON / Paths</summary><code>metrics: ${metricJson}\\nsource: ${esc(s.source_path)}\\ntarget: ${esc(s.target_path)}</code></details>
        </section>
        <section class="panel">
          <div class="label"><span>Official Source Audio</span><span>${fmt(s.source_audio_stats.duration_sec,2)}s</span></div>
          <audio controls preload="metadata" src="${esc(s.source_audio)}"></audio>
          <div class="label"><span>Source Text</span></div><div class="text">${esc(s.source_text)}</div>
          <div class="label" style="margin-top:10px"><span>Target Text</span></div><div class="text">${esc(s.target_text)}</div>
        </section>
        <section class="panel">
          <div class="label"><span>${esc(payload.backend)} Target Audio</span><span>${fmt(s.target_audio_stats.duration_sec,2)}s</span></div>
          <audio controls preload="metadata" src="${esc(s.target_audio)}"></audio>
          <div class="label"><span>Qwen-ASR</span></div><div class="text">${esc(s.asr_text)}</div>
          <details><summary>Generation status</summary><code>ok=${s.gen_ok}\\nelapsed_s=${s.elapsed_s}\\nerror=${esc(s.gen_error)}</code></details>
        </section>
      </div>
    </article>`;
  }).join("");
}
renderDownloads();
renderSummary();
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--generation-summary", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--page-dir", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--backend-label", required=True)
    args = ap.parse_args()

    base_dir = Path.cwd().resolve()
    results_path = resolve_path(args.results, base_dir)
    summary_path = resolve_path(args.summary, base_dir)
    generation_summary_path = resolve_path(args.generation_summary, base_dir)
    cases_path = resolve_path(args.cases, base_dir)
    page_dir = resolve_path(args.page_dir, base_dir)
    assert results_path and summary_path and generation_summary_path and cases_path and page_dir

    payload = build_payload(
        results_path,
        summary_path,
        generation_summary_path,
        cases_path,
        page_dir,
        args.title,
        args.backend_label,
        base_dir,
    )
    samples_js = "window.EDIT_BACKEND_SAMPLES = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    (page_dir / "samples.js").write_text(samples_js, encoding="utf-8")
    (page_dir / "index.html").write_text(HTML.replace("__TITLE__", args.title), encoding="utf-8")
    print(f"wrote page: {page_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
