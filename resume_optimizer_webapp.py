#!/usr/bin/env python3
"""
Resume Optimizer — Interactive Web Demo

Paste a job posting URL (or the job description text), upload a resume as .pdf
or .docx, and watch the LangGraph multi-agent review run live over SSE.

Usage:
  pip install -r requirements-resume-optimizer.txt
  export ANTHROPIC_API_KEY=sk-ant-...
  python resume_optimizer_webapp.py
  Open http://localhost:8030
"""

import json
import os
import queue
import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse

from resume_optimizer_agent import (
    AVAILABLE_MODELS, DEFAULT_MODEL, NODES, OUTPUT_FORMATS, Progress,
    generate_updated_resume, resolve_model, resolve_output_format, review_resume,
)

load_dotenv()

app = FastAPI()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

# Finished reports stay in memory so the download link keeps working, so the
# registry is bounded: the oldest entries are dropped once it exceeds this.
MAX_RETAINED_JOBS = 50

# job_id → {queue, resume_path, resume_name, job_url, job_text, report, started}
_JOBS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_JOBS_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Event Emitter — bridges the sync LangGraph run to the SSE stream
# ─────────────────────────────────────────────────────────────────────────────

class EventEmitter:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def emit(self, payload: dict):
        self._q.put(payload)

    def close(self):
        self._q.put(None)

    def stream(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield f"data: {json.dumps(item)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/review")
async def create_review(
    resume: Optional[UploadFile] = File(None),
    resume_text: str = Form(""),
    job_url: str = Form(""),
    job_text: str = Form(""),
    model: str = Form(""),
):
    """Accept the inputs, stage the resume on disk (or keep pasted text), hand back a job id."""
    job_url, job_text = job_url.strip(), job_text.strip()
    resume_text = resume_text.strip()
    model = resolve_model(model)
    if not job_url and len(job_text) < 200:
        raise HTTPException(
            400,
            "Provide a job posting URL, or paste at least a couple of paragraphs "
            "of the job description.",
        )

    has_file = resume is not None and bool((resume.filename or "").strip())
    if not has_file and len(resume_text) < 200:
        raise HTTPException(
            400,
            "Attach a resume file, or paste at least a couple of paragraphs of "
            "resume text.",
        )

    # A substantial paste wins over a file, mirroring the job-posting intake —
    # so the pasted-text path below runs even if a (too-short) file is also present.
    path: Optional[str] = None
    name = "Pasted resume"
    if has_file and len(resume_text) < 200:
        name = os.path.basename(resume.filename or "resume")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400, f"Unsupported resume format '{ext or 'none'}'. "
                     f"Use {', '.join(ALLOWED_EXTENSIONS)}."
            )

        data = await resume.read()
        if not data:
            raise HTTPException(400, "The uploaded resume is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"Resume is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

        fd, path = tempfile.mkstemp(suffix=ext, prefix="resume_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)

    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "queue": EventEmitter(),
            "resume_path": path,
            "resume_text": resume_text if path is None else "",
            "resume_name": name,
            "job_url": job_url,
            "job_text": job_text,
            "model": model,
            "review_result": None,
            "started": False,
        }
        while len(_JOBS) > MAX_RETAINED_JOBS:
            _, dropped = _JOBS.popitem(last=False)
            if dropped.get("resume_path"):
                try:
                    os.unlink(dropped["resume_path"])
                except OSError:
                    pass
    return {"job_id": job_id, "resume_name": name, "model": model}


@app.get("/api/stream")
def stream_review(job: str):
    """Run the graph in a worker thread and stream its progress as SSE."""
    with _JOBS_LOCK:
        record = _JOBS.get(job)
        if record is None:
            raise HTTPException(404, "Unknown or expired review job.")
        if record["started"]:
            raise HTTPException(409, "This review is already streaming.")
        record["started"] = True

    emitter: EventEmitter = record["queue"]

    def run():
        try:
            result = review_resume(
                resume_path=record["resume_path"],
                resume_text=record.get("resume_text") or None,
                job_url=record["job_url"] or None,
                job_text=record["job_text"] or None,
                resume_name=record["resume_name"],
                model=record["model"],
                progress=Progress(sink=emitter.emit),
                save_report=False,
            )
            record["review_result"] = result
            emitter.emit({
                "type": "result",
                "error": result.get("error"),
                "warnings": result.get("warnings") or [],
                "job_requirements": result.get("job_requirements"),
                "resume_profile": result.get("resume_profile"),
                "fit_assessment": result.get("fit_assessment"),
                "ats_audit": result.get("ats_audit"),
                "bullet_rewrites": result.get("bullet_rewrites"),
                "review_report": result.get("review_report"),
            })
        except Exception as exc:
            emitter.emit({"type": "result", "error": f"{exc.__class__.__name__}: {exc}"})
        finally:
            emitter.emit({"type": "done"})
            emitter.close()
            if record.get("resume_path"):
                try:
                    os.unlink(record["resume_path"])
                except OSError:
                    pass

    threading.Thread(target=run, daemon=True).start()

    return StreamingResponse(
        emitter.stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/report", response_class=PlainTextResponse)
def download_report(job: str):
    """Serve the finished review as a markdown file."""
    with _JOBS_LOCK:
        record = _JOBS.get(job)
    result = record.get("review_result") if record else None
    if not result or not result.get("review_report"):
        raise HTTPException(404, "No finished report for this job.")
    stem = os.path.splitext(record["resume_name"])[0][:60] or "resume"
    return PlainTextResponse(
        result["review_report"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="review_{stem}.md"'},
    )


_MEDIA_TYPE_BY_EXT = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@app.get("/api/update-resume")
def update_resume(job: str, format: str = "pdf"):
    """
    Apply the review's approved edits and return the enhanced resume in the
    requested format (pdf, docx, or google_doc — the latter downloads the same
    .docx a Google Doc would; Google Docs opens it natively from Drive).

    Only reachable once /api/stream has produced a successful review_result —
    the "Update my resume" button stays disabled client-side until then, and
    this is the server-side version of that same check.
    """
    with _JOBS_LOCK:
        record = _JOBS.get(job)
    if record is None:
        raise HTTPException(404, "Unknown or expired review job.")

    result = record.get("review_result")
    if not result:
        raise HTTPException(409, "The review has not finished yet.")
    if result.get("error"):
        raise HTTPException(400, "The review did not complete successfully, so there is nothing to enhance.")

    output_format = resolve_output_format(format)
    update = generate_updated_resume(
        resume_doc=result["resume_doc"],
        job_requirements=result["job_requirements"],
        ats_audit=result["ats_audit"],
        bullet_rewrites=result["bullet_rewrites"],
        model=record["model"],
        output_format=output_format,
    )
    if update.get("error"):
        raise HTTPException(502, update["error"])

    try:
        with open(update["file_path"], "rb") as fh:
            file_bytes = fh.read()
    finally:
        try:
            os.unlink(update["file_path"])
        except OSError:
            pass

    role = (result.get("job_requirements") or {}).get("role_title") or "resume"
    slug = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")[:40] or "resume"
    ext = os.path.splitext(update["file_path"])[1].lstrip(".")
    return Response(
        content=file_bytes,
        media_type=_MEDIA_TYPE_BY_EXT.get(ext, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="updated_{slug}.{ext}"'},
    )


@app.get("/api/update-formats")
def update_formats():
    """The format dropdown for 'Update my resume' — PDF, Word, or Google Doc."""
    return OUTPUT_FORMATS


@app.get("/api/models")
def available_models():
    """The curated model dropdown, plus which one is selected by default."""
    return {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL}


@app.get("/api/nodes")
def pipeline_nodes():
    """The pipeline shape, so the UI renders the same graph the agent runs."""
    return [
        {"node": node, "index": idx, "title": title, "detail": detail}
        for node, (idx, title, detail) in sorted(NODES.items(), key=lambda kv: kv[1][0])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Optimizer</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ee;
  --text:#0b0b0b; --text-2:#52514e; --muted:#898781;
  --border:rgba(11,11,11,.10); --hairline:#e1e0d9;
  --accent:#2a78d6; --accent-soft:#cde2fb; --accent-ink:#184f95;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --radius:10px;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --surface-2:#242423;
    --text:#fff; --text-2:#c3c2b7; --muted:#898781;
    --border:rgba(255,255,255,.10); --hairline:#2c2c2a;
    --accent:#3987e5; --accent-soft:#184f95; --accent-ink:#9ec5f4;
  }
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--page); color:var(--text);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
}
.wrap { max-width:1060px; margin:0 auto; padding-block:28px; padding-inline:16px; }
h1 { font-size:21px; margin:0 0 4px; letter-spacing:-.01em; }
h2 { font-size:15px; margin:0 0 14px; letter-spacing:-.005em; }
.sub { color:var(--text-2); margin:0 0 24px; font-size:13px; }
.sub code { font-family:var(--mono); font-size:12px; background:var(--surface-2); padding:1px 5px; border-radius:4px; }

.card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius); padding:18px; margin-bottom:16px;
}
label { display:block; font-weight:600; font-size:12.5px; margin-bottom:6px; }
.hint { color:var(--muted); font-size:12px; font-weight:400; margin-top:5px; }
input[type=url], textarea, select {
  width:100%; padding:9px 11px; font:inherit; color:var(--text);
  background:var(--page); border:1px solid var(--hairline); border-radius:7px;
}
input[type=url]:focus, textarea:focus, select:focus { outline:2px solid var(--accent); outline-offset:-1px; border-color:var(--accent); }
textarea { min-height:120px; resize:vertical; font-size:13px; }
select { appearance:none; cursor:pointer;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath fill='%23898781' d='M4 6l4 4 4-4H4z'/%3E%3C/svg%3E");
  background-repeat:no-repeat; background-position:right 10px center; padding-right:30px; }
select:disabled { opacity:.6; cursor:default; }
.field { margin-bottom:16px; }
.field:last-of-type { margin-bottom:0; }

.drop {
  border:1.5px dashed var(--hairline); border-radius:8px; padding:20px;
  text-align:center; color:var(--text-2); cursor:pointer; background:var(--page);
  transition:border-color .12s, background .12s;
}
.drop:hover, .drop.over { border-color:var(--accent); background:var(--accent-soft); }
.drop strong { color:var(--text); }
.drop.has-file { border-style:solid; border-color:var(--good); text-align:left; }

details.paste { margin-top:10px; }
details.paste summary { cursor:pointer; color:var(--accent-ink); font-size:12.5px; font-weight:600; }

button {
  font:inherit; font-weight:600; border-radius:7px; cursor:pointer;
  padding:9px 18px; border:1px solid transparent;
}
button.primary { background:var(--accent); color:#fff; }
button:disabled { opacity:.45; cursor:not-allowed; }
button.ghost { background:transparent; color:var(--accent-ink); border-color:var(--hairline); }
.actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:18px; }
.actions select { width:auto; min-width:132px; padding:8px 28px 8px 11px; }

/* pipeline */
.pipeline { display:grid; gap:7px; }
.step {
  display:grid; grid-template-columns:22px 1fr auto; gap:11px; align-items:baseline;
  padding:8px 10px; border-radius:7px; border:1px solid transparent; background:var(--page);
}
.step .n {
  font:600 11px/18px var(--mono); text-align:center; border-radius:50%;
  background:var(--surface-2); color:var(--muted); height:18px;
}
.step .t { font-weight:600; font-size:13px; color:var(--text-2); }
.step .s { font-size:12px; color:var(--muted); text-align:right; }
.step.active { border-color:var(--accent); background:var(--accent-soft); }
.step.active .n { background:var(--accent); color:#fff; }
.step.active .t { color:var(--text); }
.step.done .n { background:var(--good); color:#fff; }
.step.done .t { color:var(--text); }
.step.failed .n { background:var(--critical); color:#fff; }

.note { font-size:12.5px; padding:9px 11px; border-radius:7px; margin-top:10px; display:flex; gap:8px; }
.note b { flex:none; }
.note.warn { background:color-mix(in srgb,var(--warning) 14%,var(--surface)); color:var(--text); }
.note.error { background:color-mix(in srgb,var(--critical) 12%,var(--surface)); color:var(--text); }
.note.info { background:var(--surface-2); color:var(--text-2); }

/* scores */
.scores { display:flex; gap:14px; flex-wrap:wrap; }
.tile {
  flex:1 1 150px; background:var(--page); border:1px solid var(--border);
  border-radius:8px; padding:14px 16px;
}
.tile .k { font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.tile .v { font-size:30px; font-weight:650; line-height:1.15; margin-top:3px; }
.tile .v small { font-size:14px; font-weight:500; color:var(--muted); }
.tile .d { font-size:12px; color:var(--text-2); margin-top:3px; }
.headline { margin-top:14px; font-size:14px; line-height:1.5; }
.headline b { color:var(--accent-ink); }

/* coverage bar: 2px surface gaps between segments, 4px rounded ends */
.covbar { display:flex; gap:2px; height:14px; margin:4px 0 10px; }
.covbar span { display:block; }
.covbar span:first-child { border-radius:4px 0 0 4px; }
.covbar span:last-child { border-radius:0 4px 4px 0; }
.covbar span:only-child { border-radius:4px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12.5px; color:var(--text-2); }
.legend i { width:9px; height:9px; border-radius:2px; display:inline-block; margin-right:5px; }

table { width:100%; border-collapse:collapse; font-size:13px; }
.scroll { overflow-x:auto; }
th { text-align:left; font-size:11.5px; text-transform:uppercase; letter-spacing:.04em;
     color:var(--muted); font-weight:600; padding:0 10px 7px 0; border-bottom:1px solid var(--hairline); }
td { padding:9px 10px 9px 0; border-bottom:1px solid var(--hairline); vertical-align:top; }
tr:last-child td { border-bottom:none; }
td .ev { color:var(--text-2); font-size:12.5px; margin-top:3px; }

.chip {
  display:inline-flex; align-items:center; gap:5px; white-space:nowrap;
  font-size:11.5px; font-weight:600; padding:2px 8px; border-radius:20px;
  border:1px solid var(--border); background:var(--page);
}
.chip .dot { width:8px; height:8px; border-radius:50%; flex:none; }
.chip.must { color:var(--text-2); }
.kw { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
.kw span {
  font:12px/1.5 var(--mono); padding:2px 8px; border-radius:5px;
  background:var(--surface-2); color:var(--text-2); border:1px solid var(--border);
}

.rw { border:1px solid var(--border); border-radius:8px; padding:13px; margin-bottom:11px; background:var(--page); }
.rw .meta { font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
.rw .before { color:var(--text-2); font-size:13px; border-left:3px solid var(--hairline); padding-left:10px; }
.rw .after { font-weight:600; font-size:13.5px; border-left:3px solid var(--good); padding-left:10px; margin-top:8px; }
.rw .why { font-size:12.5px; color:var(--text-2); margin-top:8px; }
pre.block {
  background:var(--surface-2); border:1px solid var(--border); border-radius:7px;
  padding:12px; font:13px/1.6 var(--mono); white-space:pre-wrap; margin:0;
}

.report { font-size:14px; line-height:1.65; }
.report h1 { font-size:19px; margin:0 0 12px; }
.report h2 { font-size:15px; margin:22px 0 8px; padding-top:14px; border-top:1px solid var(--hairline); }
.report h2:first-of-type { border-top:none; padding-top:0; }
.report table { margin:12px 0; }
.report td, .report th { padding:7px 10px 7px 0; }
.report blockquote { margin:8px 0; padding-left:11px; border-left:3px solid var(--hairline); color:var(--text-2); }
.report code { font:12.5px var(--mono); background:var(--surface-2); padding:1px 5px; border-radius:4px; }
.report li { margin-bottom:5px; }
.hidden { display:none !important; }
.spin {
  width:13px; height:13px; border:2px solid var(--accent-soft); border-top-color:var(--accent);
  border-radius:50%; display:inline-block; animation:spin .7s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
.disclaimer { font-size:12px; color:var(--muted); margin-top:18px; }
@media (max-width:620px) { .tile { flex:1 1 100%; } }
</style>
</head>
<body>
<div class="wrap">

  <h1>Resume Optimizer</h1>
  <p class="sub">
    Eight LangGraph agents read the job posting, screen your resume against it, and
    propose edits built only from what your resume already evidences.
  </p>

  <!-- INPUT -->
  <div class="card" id="inputCard">
    <h2>1 · The job</h2>
    <div class="field">
      <label for="url">Job posting URL</label>
      <input type="url" id="url" placeholder="https://company.com/careers/senior-data-engineer">
      <div class="hint">
        LinkedIn, Indeed and Workday render postings in the browser or behind a login,
        so fetching them often returns nothing. Paste the text below when that happens.
      </div>
      <details class="paste">
        <summary>Paste the job description instead (or as a fallback)</summary>
        <textarea id="jd" placeholder="Paste the full job description here..." style="margin-top:9px"></textarea>
      </details>
    </div>

    <h2 style="margin-top:22px">2 · Your resume</h2>
    <div class="field">
      <div class="drop" id="drop" tabindex="0" role="button">
        <strong>Choose a file</strong> or drop it here<br>
        <span class="hint" style="margin:0">PDF, DOCX, TXT or MD · up to 10 MB · text-based PDFs only</span>
      </div>
      <input type="file" id="file" class="hidden" accept=".pdf,.docx,.txt,.md">
      <details class="paste">
        <summary>Paste resume text instead (or as a fallback)</summary>
        <textarea id="resumeText" placeholder="Paste the full resume text here..." style="margin-top:9px"></textarea>
        <div class="hint">If you add both a file and pasted text, the pasted text is used.</div>
      </details>
    </div>

    <h2 style="margin-top:22px">3 · Model</h2>
    <div class="field">
      <label for="model">Which model should run the review?</label>
      <select id="model" disabled><option>Loading models...</option></select>
      <div class="hint" id="modelNote"></div>
    </div>

    <div class="actions">
      <button class="primary" id="go" disabled>Review my resume</button>
      <select id="updateFormat" disabled title="Format for the updated resume"><option>...</option></select>
      <button class="ghost" id="updateResume" disabled title="Finish a review first">Update my resume</button>
      <span id="status" class="hint" style="margin:0"></span>
    </div>
  </div>

  <!-- PIPELINE -->
  <div class="card hidden" id="pipeCard">
    <h2>Agent pipeline</h2>
    <div class="pipeline" id="pipe"></div>
    <div id="notes"></div>
  </div>

  <!-- SCORES -->
  <div class="card hidden" id="scoreCard">
    <h2>Screening result</h2>
    <div class="scores" id="tiles"></div>
    <div class="headline" id="headline"></div>
  </div>

  <!-- COVERAGE -->
  <div class="card hidden" id="covCard">
    <h2>Requirement coverage</h2>
    <div class="covbar" id="covbar"></div>
    <div class="legend" id="covlegend"></div>
    <div class="scroll" style="margin-top:16px"><table id="covtable"></table></div>
  </div>

  <!-- KEYWORDS / ATS -->
  <div class="card hidden" id="atsCard">
    <h2>Keywords & ATS</h2>
    <div id="atsBody"></div>
  </div>

  <!-- REWRITES -->
  <div class="card hidden" id="rwCard">
    <h2>Proposed rewrites</h2>
    <div id="rwBody"></div>
  </div>

  <!-- REPORT -->
  <div class="card hidden" id="repCard">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
      <h2 style="margin:0">Full review</h2>
      <button class="ghost" id="dl">Download .md</button>
    </div>
    <div class="report" id="report" style="margin-top:16px"></div>
    <p class="disclaimer">
      Generated by an AI pipeline. Check every suggested line against your own record
      before you send it — you are the one who has to defend it in the interview.
    </p>
  </div>

</div>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const STATUS = {
  covered: { label:"Covered", icon:"✓", color:"var(--good)" },
  partial: { label:"Partial", icon:"~", color:"var(--warning)" },
  missing: { label:"Missing", icon:"✕", color:"var(--critical)" },
};
const SEVERITY = {
  critical:{ label:"Critical", icon:"✕", color:"var(--critical)" },
  warning: { label:"Warning",  icon:"!", color:"var(--warning)" },
  minor:   { label:"Minor",    icon:"·", color:"var(--muted)" },
  useful:  { label:"Useful",   icon:"+", color:"var(--accent)" },
};

/* A status never speaks through color alone: every chip carries icon + word. */
function chip(map, key, fallback) {
  const s = map[key] || { label: fallback || key || "—", icon:"·", color:"var(--muted)" };
  return `<span class="chip"><i class="dot" style="background:${s.color}"></i>${s.icon} ${esc(s.label)}</span>`;
}

/* ── model dropdown ────────────────────────────────────────────────────── */
let MODELS = [];

async function loadModels() {
  try {
    const { models, default: def } = await fetch("/api/models").then(r => r.json());
    MODELS = models;
    $("model").innerHTML = models.map(m =>
      `<option value="${esc(m.id)}" ${m.id === def ? "selected" : ""}>${esc(m.label)}</option>`
    ).join("");
    $("model").disabled = false;
    updateModelNote();
  } catch (err) {
    $("model").innerHTML = `<option value="">Default model</option>`;
    $("modelNote").textContent = "Could not load the model list — using the server default.";
  }
}

function updateModelNote() {
  const chosen = MODELS.find(m => m.id === $("model").value);
  $("modelNote").textContent = chosen ? chosen.note : "";
}

/* ── update-format dropdown ───────────────────────────────────────────────── */
let UPDATE_FORMATS = [];

async function loadUpdateFormats() {
  try {
    UPDATE_FORMATS = await fetch("/api/update-formats").then(r => r.json());
    $("updateFormat").innerHTML = UPDATE_FORMATS.map(f =>
      `<option value="${esc(f.id)}" title="${esc(f.note)}">${esc(f.label)}</option>`
    ).join("");
  } catch (err) {
    $("updateFormat").innerHTML = `<option value="pdf">PDF</option>`;
  }
}

$("model").addEventListener("change", updateModelNote);

/* ── input wiring ───────────────────────────────────────────────────────── */
let picked = null;

$("drop").onclick = () => $("file").click();
$("drop").onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
$("file").onchange = (e) => setFile(e.target.files[0]);
["dragenter","dragover"].forEach(ev => $("drop").addEventListener(ev, e => {
  e.preventDefault(); $("drop").classList.add("over");
}));
["dragleave","drop"].forEach(ev => $("drop").addEventListener(ev, e => {
  e.preventDefault(); $("drop").classList.remove("over");
}));
$("drop").addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

function setFile(f) {
  if (!f) return;
  picked = f;
  $("drop").classList.add("has-file");
  $("drop").innerHTML = `<strong>${esc(f.name)}</strong>
    <span class="hint" style="margin:0"> · ${(f.size/1024).toFixed(0)} KB · click to replace</span>`;
  refresh();
}
["input","change"].forEach(ev => {
  $("url").addEventListener(ev, refresh);
  $("jd").addEventListener(ev, refresh);
  $("resumeText").addEventListener(ev, refresh);
});

function refresh() {
  const haveJob = $("url").value.trim() || $("jd").value.trim().length >= 200;
  const haveResume = !!picked || $("resumeText").value.trim().length >= 200;
  $("go").disabled = !(haveResume && haveJob);
  $("status").textContent = !haveResume ? "Add a resume — upload a file or paste its text."
    : !haveJob ? "Add a job posting URL, or paste the job description."
    : "";
}

/* ── run ────────────────────────────────────────────────────────────────── */
let jobId = null;
let reviewSucceeded = false;

$("go").onclick = async () => {
  $("go").disabled = true;
  $("model").disabled = true;
  $("updateResume").disabled = true;
  $("updateFormat").disabled = true;
  reviewSucceeded = false;
  $("status").innerHTML = '<span class="spin"></span> Uploading...';
  $("pipeCard").classList.remove("hidden");
  ["scoreCard","covCard","atsCard","rwCard","repCard"].forEach(id => $(id).classList.add("hidden"));
  $("notes").innerHTML = "";
  await buildPipeline();

  const body = new FormData();
  if (picked) body.append("resume", picked);
  body.append("resume_text", $("resumeText").value.trim());
  body.append("job_url", $("url").value.trim());
  body.append("job_text", $("jd").value.trim());
  body.append("model", $("model").value);

  let res;
  try {
    res = await fetch("/api/review", { method:"POST", body });
  } catch (err) {
    return fail(`Upload failed: ${err.message}`);
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    return fail(detail.detail || `Upload rejected (HTTP ${res.status}).`);
  }
  const created = await res.json();
  jobId = created.job_id;
  const modelLabel = MODELS.find(m => m.id === created.model)?.label || created.model;
  $("status").innerHTML =
    `<span class="spin"></span> Reviewing with ${esc(modelLabel)} — eight agents, usually 3-6 minutes.`;
  listen(jobId);
};

function fail(message) {
  $("go").disabled = false;
  $("model").disabled = false;
  $("status").textContent = "";
  note("error", message);
}

async function buildPipeline() {
  const nodes = await fetch("/api/nodes").then(r => r.json());
  $("pipe").innerHTML = nodes.map(n => `
    <div class="step" id="step-${n.node}">
      <span class="n">${n.index}</span>
      <span class="t">${esc(n.title)}<span class="hint" style="font-weight:400;display:block">${esc(n.detail)}</span></span>
      <span class="s"></span>
    </div>`).join("");
}

function note(level, message) {
  const tag = { warn:"Heads up", error:"Problem", info:"Note" }[level] || "Note";
  $("notes").insertAdjacentHTML("beforeend",
    `<div class="note ${level}"><b>${tag}:</b><span>${esc(message)}</span></div>`);
}

function listen(id) {
  let finished = false;
  const es = new EventSource(`/api/stream?job=${encodeURIComponent(id)}`);
  es.onmessage = (ev) => {
    const e = JSON.parse(ev.data);
    if (e.type === "node_start") {
      $(`step-${e.node}`)?.classList.add("active");
    } else if (e.type === "node_done") {
      const el = $(`step-${e.node}`);
      if (el) { el.classList.remove("active"); el.classList.add("done");
                el.querySelector(".s").textContent = e.summary || ""; }
    } else if (e.type === "note") {
      note(e.level, e.message);
      if (e.level === "error") {
        document.querySelectorAll(".step.active").forEach(s => {
          s.classList.remove("active"); s.classList.add("failed");
        });
      }
    } else if (e.type === "result") {
      render(e);
    } else if (e.type === "done") {
      finished = true;
      es.close();
      $("go").disabled = false;
      $("model").disabled = false;
      $("updateResume").disabled = !reviewSucceeded;
      $("updateFormat").disabled = !reviewSucceeded;
      $("status").textContent = "";
    }
  };
  /* The server ends the stream after "done"; only an early close is a real failure. */
  es.onerror = () => {
    es.close();
    if (!finished) fail("Lost the connection to the server.");
  };
}

/* ── render ─────────────────────────────────────────────────────────────── */
function render(r) {
  reviewSucceeded = !r.error;
  (r.warnings || []).forEach(w => note("warn", w));
  if (r.error) { note("error", r.error); return; }

  const fit = r.fit_assessment || {};
  const ats = r.ats_audit || {};
  const req = r.job_requirements || {};
  const rw  = r.bullet_rewrites || {};
  const coverage = fit.coverage || [];

  /* scores — a hero figure and three supporting tiles, no chart needed */
  const counts = { covered:0, partial:0, missing:0 };
  coverage.forEach(c => { if (counts[c.status] !== undefined) counts[c.status]++; });
  const musts = coverage.filter(c => c.importance === "must_have");
  const mustsCovered = musts.filter(c => c.status === "covered").length;

  $("tiles").innerHTML = `
    <div class="tile">
      <div class="k">Fit score</div>
      <div class="v">${esc(fit.fit_score ?? "—")}<small>/100</small></div>
      <div class="d">${esc(fit.verdict || "")}</div>
    </div>
    <div class="tile">
      <div class="k">ATS score</div>
      <div class="v">${esc(ats.ats_score ?? "—")}<small>/100</small></div>
      <div class="d">keyword &amp; parse health</div>
    </div>
    <div class="tile">
      <div class="k">Must-haves covered</div>
      <div class="v">${mustsCovered}<small>/${musts.length}</small></div>
      <div class="d">${esc(req.role_title || "")}</div>
    </div>
    <div class="tile">
      <div class="k">Experience</div>
      <div class="v" style="font-size:19px;padding-top:7px">${esc(fit.years_evidenced || "—")}</div>
      <div class="d">role asks for ${esc(fit.years_required || "—")}</div>
    </div>`;

  $("headline").innerHTML =
    (fit.headline ? `<b>Biggest lever:</b> ${esc(fit.headline)}<br>` : "") +
    (fit.interview_odds ? esc(fit.interview_odds) : "");
  $("scoreCard").classList.remove("hidden");

  /* coverage */
  if (coverage.length) {
    const total = coverage.length;
    $("covbar").innerHTML = ["covered","partial","missing"]
      .filter(k => counts[k])
      .map(k => `<span style="width:${(counts[k]/total*100).toFixed(2)}%;background:${STATUS[k].color}"
                       title="${counts[k]} ${STATUS[k].label.toLowerCase()}"></span>`).join("");
    $("covlegend").innerHTML = ["covered","partial","missing"].map(k =>
      `<span><i style="background:${STATUS[k].color}"></i>${STATUS[k].icon} ${STATUS[k].label} — ${counts[k]}</span>`
    ).join("");
    $("covtable").innerHTML = `
      <thead><tr><th>Requirement</th><th>Weight</th><th>Status</th><th>What to do</th></tr></thead>
      <tbody>${coverage.map(c => `
        <tr>
          <td>${esc(c.requirement)}${c.evidence ? `<div class="ev">“${esc(c.evidence)}”</div>` : ""}</td>
          <td><span class="chip must">${c.importance === "must_have" ? "Must have" : "Nice to have"}</span></td>
          <td>${chip(STATUS, c.status)}</td>
          <td>${esc(c.action)}</td>
        </tr>`).join("")}</tbody>`;
    $("covCard").classList.remove("hidden");
  }

  /* ATS */
  const missing = ats.missing_keywords || [];
  const issues  = ats.formatting_issues || [];
  const order   = { critical:0, warning:1, useful:1, minor:2 };
  issues.sort((a,b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3));

  let atsHtml = "";
  if (ats.parse_quality) atsHtml += `<p style="margin:0 0 8px">${esc(ats.parse_quality)}</p>`;
  if (ats.length_verdict) atsHtml += `<p style="margin:0 0 14px;color:var(--text-2)">${esc(ats.length_verdict)}</p>`;
  if (missing.length) {
    atsHtml += `<div class="scroll"><table>
      <thead><tr><th>Missing keyword</th><th>Weight</th><th>Where it belongs</th><th>Only if</th></tr></thead>
      <tbody>${missing.map(k => `<tr>
        <td><code>${esc(k.keyword)}</code></td>
        <td>${chip(SEVERITY, k.importance)}</td>
        <td>${esc(k.where_to_add)}</td>
        <td>${esc(k.truthful_if)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  if (issues.length) {
    atsHtml += `<h2 style="margin:22px 0 10px">Formatting</h2><div class="scroll"><table>
      <thead><tr><th>Issue</th><th>Severity</th><th>Fix</th></tr></thead>
      <tbody>${issues.map(i => `<tr>
        <td>${esc(i.issue)}</td><td>${chip(SEVERITY, i.severity)}</td><td>${esc(i.fix)}</td></tr>`).join("")}</tbody>
      </table></div>`;
  }
  if ((ats.present_keywords || []).length) {
    atsHtml += `<h2 style="margin:22px 0 4px">Already present</h2>
      <div class="kw">${ats.present_keywords.map(k => `<span>${esc(k)}</span>`).join("")}</div>`;
  }
  if (ats.summary_rewrite) {
    atsHtml += `<h2 style="margin:22px 0 10px">Suggested summary</h2>
      <pre class="block">${esc(ats.summary_rewrite)}</pre>`;
  }
  if (atsHtml) { $("atsBody").innerHTML = atsHtml; $("atsCard").classList.remove("hidden"); }

  /* rewrites */
  let rwHtml = (rw.rewrites || []).map(x => `
    <div class="rw">
      <div class="meta">${esc(x.section || "")}${x.requirement_served ? " · serves: " + esc(x.requirement_served) : ""}</div>
      <div class="before">${esc(x.before)}</div>
      <div class="after">${esc(x.after)}</div>
      ${x.why ? `<div class="why">${esc(x.why)}</div>` : ""}
    </div>`).join("");
  if ((rw.new_bullets_if_true || []).length) {
    rwHtml += `<h2 style="margin:22px 0 10px">Add these — but only if true</h2>` +
      rw.new_bullets_if_true.map(b => `
        <div class="rw">
          <div class="after" style="border-left-color:var(--warning)">${esc(b.bullet)}</div>
          <div class="why">Only if: ${esc(b.only_if)}</div>
        </div>`).join("");
  }
  if (rw.skills_section) {
    rwHtml += `<h2 style="margin:22px 0 10px">Suggested skills section</h2>
      <pre class="block">${esc(rw.skills_section)}</pre>`;
  }
  if ((rw.upskilling || []).length) {
    rwHtml += `<h2 style="margin:22px 0 10px">What editing cannot fix</h2><div class="scroll"><table>
      <thead><tr><th>Gap</th><th>Effort</th><th>How to close it</th></tr></thead>
      <tbody>${rw.upskilling.map(u => `<tr>
        <td>${esc(u.gap)}</td><td>${esc(u.closes_in)}</td><td>${esc(u.how)}</td></tr>`).join("")}</tbody>
      </table></div>`;
  }
  if (rwHtml) { $("rwBody").innerHTML = rwHtml; $("rwCard").classList.remove("hidden"); }

  /* full report */
  if (r.review_report) {
    $("report").innerHTML = (typeof marked !== "undefined")
      ? marked.parse(r.review_report)
      : `<pre class="block">${esc(r.review_report)}</pre>`;
    $("repCard").classList.remove("hidden");
  }
}

$("dl").onclick = () => {
  if (jobId) window.location = `/api/report?job=${encodeURIComponent(jobId)}`;
};

$("updateResume").onclick = async () => {
  if (!jobId) return;
  const format = $("updateFormat").value || "pdf";
  const chosen = UPDATE_FORMATS.find(f => f.id === format);
  const original = $("updateResume").textContent;
  $("updateResume").disabled = true;
  $("updateFormat").disabled = true;
  $("updateResume").textContent = "Enhancing...";
  $("status").innerHTML = `<span class="spin"></span> Applying the review's edits and rendering ${esc(chosen?.label || format)} — usually 15-30s.`;

  try {
    const res = await fetch(`/api/update-resume?job=${encodeURIComponent(jobId)}&format=${encodeURIComponent(format)}`);
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (HTTP ${res.status}).`);
    }
    const blob = await res.blob();
    const match = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") || "");
    const filename = match ? match[1] : "updated_resume.pdf";

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    note("info", `Updated resume downloaded as ${filename}.` + (chosen?.note ? ` ${chosen.note}` : ""));
  } catch (err) {
    note("error", `Could not generate the updated resume: ${err.message}`);
  } finally {
    $("updateResume").disabled = false;
    $("updateFormat").disabled = false;
    $("updateResume").textContent = original;
    $("status").textContent = "";
  }
};

refresh();
loadModels();
loadUpdateFormats();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.\n"
              "Run:  export ANTHROPIC_API_KEY='sk-ant-...'")
        raise SystemExit(1)
    print("Starting Resume Optimizer → http://localhost:8030")
    uvicorn.run(app, host="0.0.0.0", port=8030, log_level="warning")
