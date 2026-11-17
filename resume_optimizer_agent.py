#!/usr/bin/env python3
"""
Resume Optimizer — Agentic AI System using LangGraph

Reads a job posting, works out what the role actually requires, parses a resume
(.pdf / .docx / .txt), then scores the fit and proposes concrete, honest edits
to raise the odds of landing an interview.

Architecture (two parallel intake branches → fan-in → two parallel auditors → writer):

   ┌─ job_scraper   → jd_analyst     ─┐                ┌─ ats_auditor     ─┐
 ──┤                                   ├→ fit_scorer ──┤                    ├→ report_writer → END
   └─ resume_parser → resume_analyst ─┘                └─ bullet_rewriter ─┘

The intake branches are independent I/O (HTTP fetch vs. file parse) so they run
concurrently; the two auditors both read the same fit assessment and also fan out.

Usage:
  export ANTHROPIC_API_KEY='sk-ant-...'
  python resume_optimizer_agent.py https://example.com/careers/123 ./resume.pdf
  python resume_optimizer_agent.py --jd-file jd.txt ./resume.docx
"""

import json
import operator
import os
import re
import shutil
import sys
import tempfile
import threading
import warnings
from datetime import datetime
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, TypedDict
from xml.sax.saxutils import escape as _xml_escape

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import requests
from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# ─────────────────────────────────────────────────────────────────────────────
# LLM — user-selectable model, one ChatAnthropic instance per model id
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Curated for the dropdown — id must be a real Anthropic model string.
# `supports_temperature`: the 5th-generation models (Opus 5, Sonnet 5, Fable 5)
# removed sampling params entirely — passing `temperature` is a 400, not a no-op —
# so it is only sent for models where it is still accepted.
AVAILABLE_MODELS: List[Dict[str, Any]] = [
    {"id": "claude-opus-5",     "label": "Claude Opus 5",     "note": "Most capable — best for a close, judgment-heavy screen",
     "supports_temperature": False},
    {"id": "claude-sonnet-5",   "label": "Claude Sonnet 5",    "note": "Strong and fast — good default",
     "supports_temperature": False},
    {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6",  "note": "Repo default",
     "supports_temperature": True},
    {"id": "claude-haiku-4-5",  "label": "Claude Haiku 4.5",   "note": "Fastest and cheapest — lighter-touch review",
     "supports_temperature": True},
]
_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}
_SUPPORTS_TEMPERATURE = {m["id"]: m["supports_temperature"] for m in AVAILABLE_MODELS}

_llm_cache: Dict[str, ChatAnthropic] = {}
_llm_cache_lock = threading.Lock()


def resolve_model(model: Optional[str]) -> str:
    """Fall back to the default for anything blank or not in the curated list."""
    model = (model or "").strip()
    return model if model in _MODEL_IDS else DEFAULT_MODEL


def _get_llm(model: str) -> ChatAnthropic:
    """One ChatAnthropic client per model id, built lazily and reused."""
    with _llm_cache_lock:
        client = _llm_cache.get(model)
        if client is None:
            kwargs: Dict[str, Any] = {"model": model, "max_tokens": 8000}
            # Default to supporting it for any model outside our curated list —
            # that matches the behaviour before this flag existed.
            if _SUPPORTS_TEMPERATURE.get(model, True):
                kwargs["temperature"] = 0.2
            client = ChatAnthropic(**kwargs)
            _llm_cache[model] = client
        return client

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE METADATA — shared by the CLI and the web UI
# ─────────────────────────────────────────────────────────────────────────────

NODES: Dict[str, Tuple[int, str, str]] = {
    "job_scraper":     (1, "Job Scraper",     "Fetching the job posting"),
    "resume_parser":   (2, "Resume Parser",   "Extracting resume text"),
    "jd_analyst":      (3, "JD Analyst",      "Structuring role requirements"),
    "resume_analyst":  (4, "Resume Analyst",  "Profiling the candidate"),
    "fit_scorer":      (5, "Fit Scorer",      "Scoring requirement coverage"),
    "ats_auditor":     (6, "ATS Auditor",     "Auditing keywords & formatting"),
    "bullet_rewriter": (7, "Bullet Rewriter", "Rewriting bullets for this role"),
    "report_writer":   (8, "Report Writer",   "Assembling the review"),
}

# The "Update my resume" pipeline — a second, much shorter pipeline triggered by
# its own button after a review completes, not part of NODES above.
UPDATE_NODES: Dict[str, Tuple[int, str, str]] = {
    "resume_builder":  (1, "Resume Builder",  "Applying the approved edits"),
    "document_writer": (2, "Document Writer", "Rendering the document"),
}

# The three download formats offered alongside "Update my resume". "google_doc"
# is not a real Google Docs API integration (that needs an OAuth-configured
# Google Cloud project) — it downloads the same .docx a Google Doc would, which
# Google Docs opens natively from Drive with no conversion step.
OUTPUT_FORMATS: List[Dict[str, str]] = [
    {"id": "pdf", "label": "PDF", "ext": "pdf",
     "note": "Ready to send as-is."},
    {"id": "docx", "label": "Word (.docx)", "ext": "docx",
     "note": "Editable in Word or any word processor."},
    {"id": "google_doc", "label": "Google Doc", "ext": "docx",
     "note": "Downloads as .docx — upload it to Google Drive and it opens "
             "directly as an editable Google Doc."},
]
_OUTPUT_FORMAT_BY_ID: Dict[str, Dict[str, str]] = {f["id"]: f for f in OUTPUT_FORMATS}


def resolve_output_format(fmt: Optional[str]) -> str:
    """Fall back to PDF for anything blank or not in OUTPUT_FORMATS."""
    fmt = (fmt or "").strip()
    return fmt if fmt in _OUTPUT_FORMAT_BY_ID else "pdf"

# Progress looks nodes up here regardless of which pipeline they belong to, with
# each node keeping its own pipeline's total (8 vs. 2) rather than a merged count.
_ALL_NODES: Dict[str, Tuple[int, str, str, int]] = {
    **{node: (*info, len(NODES)) for node, info in NODES.items()},
    **{node: (*info, len(UPDATE_NODES)) for node, info in UPDATE_NODES.items()},
}


class Progress:
    """
    Pipeline progress sink. The CLI prints to stdout; the web app hands in a
    callable that pushes each event onto an SSE queue.
    """

    def __init__(self, sink: Optional[Callable[[dict], None]] = None):
        self._sink = sink

    def start(self, node: str) -> None:
        idx, title, detail, total = _ALL_NODES[node]
        self._emit("node_start", node=node, index=idx, total=total,
                   title=title, detail=detail)

    def finish(self, node: str, summary: str = "") -> None:
        idx, title, _, _ = _ALL_NODES[node]
        self._emit("node_done", node=node, index=idx, title=title, summary=summary)

    def note(self, message: str, level: str = "info") -> None:
        self._emit("note", message=message, level=level)

    def _emit(self, kind: str, **data: Any) -> None:
        payload = {"type": kind, **data}
        if self._sink:
            self._sink(payload)
            return
        if kind == "node_start":
            print(f"[{data['index']}/{data['total']}] {data['title']} — {data['detail']} ...")
        elif kind == "node_done" and data.get("summary"):
            print(f"    ✓ {data['summary']}")
        elif kind == "note":
            mark = {"warn": "!", "error": "✗"}.get(data["level"], "·")
            print(f"    {mark} {data['message']}")


def _progress(state: "ResumeReviewState") -> Progress:
    return state.get("progress") or Progress()

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class ResumeReviewState(TypedDict):
    # inputs
    job_url:          Optional[str]
    job_text_input:   Optional[str]   # pasted JD — used when a board blocks scraping
    resume_path:      Optional[str]   # one of resume_path / resume_text_input is required
    resume_text_input: Optional[str]  # pasted resume — the copy/paste alternative to a file
    resume_name:      str
    model:            str             # Anthropic model id for every LLM call this run
    progress:         Optional[Progress]
    # intake
    job_posting:      Optional[Dict[str, Any]]   # {text, source, title}
    resume_doc:       Optional[Dict[str, Any]]   # {text, stats}
    # analysis
    job_requirements: Optional[Dict[str, Any]]
    resume_profile:   Optional[Dict[str, Any]]
    fit_assessment:   Optional[Dict[str, Any]]
    ats_audit:        Optional[Dict[str, Any]]
    bullet_rewrites:  Optional[Dict[str, Any]]
    review_report:    Optional[str]
    # bookkeeping — reducers, because the parallel branches both append
    messages:         Annotated[List[BaseMessage], operator.add]
    warnings:         Annotated[List[str], operator.add]
    error:            Optional[str]

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT & WEB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# Job boards that render the posting client-side or behind a login wall. We can
# still try, but the user needs a clear "paste it instead" signal when we fail.
_LOGIN_WALL_HINTS = (
    "sign in to continue", "please enable javascript", "enable javascript to",
    "join linkedin", "verify you are human", "access denied", "are you a robot",
    "captcha", "unusual traffic",
)

# PDF text extraction from icon fonts, two-column templates and embedded glyphs
# routinely leaves behind raw control characters and lone surrogates. Left in,
# the model may echo them verbatim into a JSON string value it returns (e.g. a
# quoted "evidence" line), which breaks JSON parsing on our end even when the
# model's structure was otherwise fine — so every text block that reaches a
# prompt goes through this first.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Strip control characters, the Unicode replacement char, and lone surrogates."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("�", "")
    text = "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)
    return text


def fetch_job_posting(url: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Fetch a job posting URL and reduce it to readable text.

    Returns {text, title, source, warning}. `text` is empty when the fetch
    produced nothing usable — the caller then falls back to pasted text.
    """
    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Prefer the structured posting if the site publishes JSON-LD (most ATS do).
    structured = _job_posting_from_jsonld(soup)

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "form", "svg", "iframe"]):
        tag.decompose()

    title = (soup.title.get_text(strip=True) if soup.title else "") or url
    body = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

    text = structured or body
    warning = ""
    lowered = body.lower()
    if len(text) < 400:
        warning = "The page returned almost no text — it is probably rendered by JavaScript."
    elif any(hint in lowered for hint in _LOGIN_WALL_HINTS):
        warning = "The page looks like a login wall or bot check, so the text may be incomplete."

    return {
        "text": text[:40_000],
        "title": title[:300],
        "source": url,
        "structured": bool(structured),
        "warning": warning,
    }


def _job_posting_from_jsonld(soup: BeautifulSoup) -> str:
    """Pull schema.org/JobPosting fields out of JSON-LD when the site provides them."""
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict):
                continue
            if "JobPosting" not in str(node.get("@type", "")):
                continue
            parts = []
            for key in ("title", "employmentType", "datePosted", "experienceRequirements",
                        "educationRequirements", "qualifications", "responsibilities",
                        "skills", "description"):
                value = node.get(key)
                if value:
                    parts.append(f"{key}: {_flatten(value)}")
            org = node.get("hiringOrganization")
            if isinstance(org, dict) and org.get("name"):
                parts.insert(1, f"company: {org['name']}")
            if parts:
                return BeautifulSoup("\n".join(parts), "html.parser").get_text("\n", strip=True)
    return ""


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={_flatten(v)}" for k, v in value.items() if v)
    if isinstance(value, list):
        return "; ".join(_flatten(v) for v in value)
    return str(value)


def _resume_stats(text: str, fmt: str, page_count: int = 0) -> Dict[str, Any]:
    """Mechanical stats shared by file-extracted and pasted resume text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return {
        "word_count":   len(text.split()),
        "line_count":   len(lines),
        "bullet_count": sum(1 for line in lines if re.match(r"^[•▪◦\-\*•]", line)),
        "page_count":   page_count,
        "has_email":    bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)),
        "has_phone":    bool(re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)),
        "has_links":    bool(re.search(r"(linkedin\.com|github\.com|https?://)", text, re.I)),
        "format":       fmt,
    }


def _clean_resume_text(text: str) -> str:
    """Sanitize and normalise whitespace — shared by both resume text sources."""
    text = _sanitize_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_resume_text_from_string(text: str) -> Dict[str, Any]:
    """Build the same {text, stats} shape as extract_resume_text, for pasted resume text."""
    text = _clean_resume_text(text)
    return {"text": text[:60_000], "stats": _resume_stats(text, "pasted")}


def extract_resume_text(path: str) -> Dict[str, Any]:
    """
    Read a resume from disk and return {text, stats}. Supports .pdf, .docx,
    .txt and .md — the formats people actually keep resumes in.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Reading .pdf resumes needs pypdf — `pip install pypdf`.") from exc
        reader = PdfReader(path)
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(pages)
        page_count = len(pages)
    elif ext == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError(
                "Reading .docx resumes needs python-docx — `pip install python-docx`."
            ) from exc
        document = docx.Document(path)
        blocks = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(blocks)
        page_count = 0
    elif ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        page_count = 0
    elif ext == ".doc":
        raise ValueError(
            "Legacy .doc files cannot be read. Re-save the resume as .docx or .pdf."
        )
    else:
        raise ValueError(f"Unsupported resume format '{ext}'. Use .pdf, .docx, .txt or .md.")

    text = _clean_resume_text(text)
    return {"text": text[:60_000], "stats": _resume_stats(text, ext.lstrip("."), page_count)}

# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Every agent that proposes resume edits gets this. The whole product is
# worthless — and harmful to the candidate in an interview — if it invents
# experience, so the rule is repeated verbatim rather than paraphrased.
_HONESTY_RULE = """NON-NEGOTIABLE RULES:
- Never invent experience, employers, dates, titles, certifications or metrics.
- Every rewrite must be traceable to something already in the resume. You may
  re-frame, re-order, sharpen verbs, surface implied skills and quantify where
  the resume already implies a number — nothing more.
- If a number would strengthen a bullet but the resume does not contain one, write
  the bullet with a clearly marked placeholder like [X%] and say what to measure.
- If a requirement is genuinely not met, say so and recommend how to close the gap.
  Do not paper over it with keyword stuffing.
- Never use an em dash (—) anywhere in your output. Use a period, comma, colon, or
  a plain hyphen instead, or just restructure the sentence."""


_EM_DASH_RE = re.compile(r"\s*—\s*")


def _strip_em_dashes(text: str) -> str:
    """
    Replace every em dash with a plain, space-padded hyphen - Claude's prose
    defaults to em dashes, and the user asked for none in the generated output.
    Normalises surrounding whitespace rather than a bare character swap, so
    "word-emdash-word" (no spaces) doesn't come out looking like a compound
    word and "word - emdash - word" (already spaced) doesn't double up.
    """
    return _EM_DASH_RE.sub(" - ", text)


def _response_text(response: Any) -> str:
    """
    Extract the assistant's visible text from a ChatAnthropic response.

    `response.content` is a plain string only when the model skipped extended
    thinking for that turn. Thinking defaults ON for Claude Opus 5 / Sonnet 5
    whenever it is not explicitly configured, and it kicks in reliably on a
    judgment-heavy prompt (exactly what fit_scorer is) — when it does,
    `response.content` becomes a list of content blocks: a thinking block
    plus a text block, not text alone. str()-ing that list stringifies the
    whole block structure (including the opaque thinking signature), which
    is not JSON and can never parse — reliably, on every attempt, which is
    why a retry on its own never recovered from this. Only the text blocks
    are ever the actual answer.

    Also strips em dashes here, once, for every caller: this runs before JSON
    parsing in `_ask_json` (a plain hyphen is just as valid inside a JSON
    string as an em dash was) and covers the markdown report in
    `report_writer_node` the same way.
    """
    content = response.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
        text = "".join(parts) if parts else str(content)
    else:
        text = str(content)
    return _strip_em_dashes(text)


def _ask_json(llm: ChatAnthropic, system: str, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke the model and parse a JSON object out of the reply.

    Structured output drives the web UI (score, coverage table, chips), so a
    parse failure degrades to `fallback` plus the raw text rather than killing
    the run — the markdown report downstream is still useful. One retry with a
    sharper nudge runs first: a single bad roll — a sentence of preamble, or the
    model tripping over messy PDF-extracted text while quoting it back — is common
    enough to be worth a second try before accepting defeat.
    """
    messages: List[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=f"{prompt}\n\nReply with JSON only — no prose, no code fence."),
    ]
    raw = ""
    for attempt in range(2):
        response = llm.invoke(messages)
        raw = _response_text(response)
        parsed = _parse_json_object(raw)
        if parsed is not None:
            return parsed
        if attempt == 0:
            messages = messages + [
                AIMessage(content=raw[:2000]),
                HumanMessage(content=(
                    "That reply could not be parsed as JSON. Reply again with a single JSON "
                    "object only — no commentary, no code fence — and make sure every quote, "
                    "newline or backslash inside a string value is properly escaped."
                )),
            ]
    return {**fallback, "_parse_failed": True, "_raw": raw[:4000]}


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON object from a model reply, tolerating fences, prose, and literal
    control characters inside string values (json.loads is strict about those by
    default; the model is quoting messy resume text, not writing a wire protocol).
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text, strict=False)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost balanced {...} span.
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:i + 1], strict=False)
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — Job Scraper (intake branch A)
# ─────────────────────────────────────────────────────────────────────────────

def job_scraper_node(state: ResumeReviewState) -> Dict:
    """Resolves the job posting text from a URL, pasted text, or both."""
    prog = _progress(state)
    prog.start("job_scraper")

    url = (state.get("job_url") or "").strip()
    pasted = (state.get("job_text_input") or "").strip()
    notes: List[str] = []

    fetched: Dict[str, Any] = {}
    if url:
        try:
            fetched = fetch_job_posting(url)
            if fetched.get("warning"):
                notes.append(f"{fetched['warning']}")
                prog.note(fetched["warning"], level="warn")
        except Exception as exc:
            notes.append(f"Could not fetch {url} ({exc.__class__.__name__}: {exc}).")
            prog.note(f"Fetch failed: {exc}", level="warn")

    scraped_text = (fetched.get("text") or "").strip()

    # Pasted text wins whenever it is substantial — it is the user's own copy of
    # the posting, and it is the escape hatch for JS-rendered boards.
    if len(pasted) >= 200:
        text, source = pasted, "pasted"
        if scraped_text:
            notes.append("Used the pasted job description; the fetched page was kept as context.")
            text = f"{pasted}\n\n--- ADDITIONAL TEXT FETCHED FROM {url} ---\n{scraped_text[:12000]}"
            source = "pasted+fetched"
    elif len(scraped_text) >= 200:
        text, source = scraped_text, "fetched"
    else:
        message = (
            "No usable job description. The URL could not be read "
            "(many boards such as LinkedIn and Indeed render postings in the browser "
            "or require a login) — paste the job description text instead."
        )
        prog.note(message, level="error")
        return {
            "error": message,
            "warnings": notes,
            "messages": [HumanMessage(content=message)],
        }

    text = _sanitize_text(text)
    title = fetched.get("title") or "Job posting"
    prog.finish("job_scraper", f"{len(text.split()):,} words from {source}")

    return {
        "job_posting": {
            "text": text,
            "title": title,
            "source": source,
            "url": url,
            "structured": bool(fetched.get("structured")),
        },
        "warnings": notes,
        "messages": [HumanMessage(content=f"Job posting acquired from {source}: {title}")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — Resume Parser (intake branch B)
# ─────────────────────────────────────────────────────────────────────────────

def resume_parser_node(state: ResumeReviewState) -> Dict:
    """Extracts raw text and mechanical statistics from the resume — pasted or a file."""
    prog = _progress(state)
    prog.start("resume_parser")

    pasted = (state.get("resume_text_input") or "").strip()
    path = (state.get("resume_path") or "").strip()

    # Pasted text wins whenever it is substantial — mirrors the job-posting intake,
    # where a paste is the deliberate, reliable path and a file is the convenience.
    if len(pasted) >= 200:
        doc = extract_resume_text_from_string(pasted)
        filename = state.get("resume_name") or "Pasted resume"
    elif path:
        try:
            doc = extract_resume_text(path)
        except Exception as exc:
            message = f"Could not read the resume '{state.get('resume_name', path)}': {exc}"
            prog.note(message, level="error")
            return {
                "error": message,
                "messages": [HumanMessage(content=message)],
            }
        filename = state.get("resume_name") or os.path.basename(path)
    else:
        message = "No resume provided — upload a file or paste the resume text."
        prog.note(message, level="error")
        return {"error": message, "messages": [HumanMessage(content=message)]}

    if doc["stats"]["word_count"] < 80:
        message = (
            f"Only {doc['stats']['word_count']} words of text came out of the resume. "
            "If it is a scanned or image-only PDF, export a text-based PDF or a .docx — "
            "or paste the resume text directly."
        )
        prog.note(message, level="error")
        return {"error": message, "messages": [HumanMessage(content=message)]}

    stats = doc["stats"]
    prog.finish("resume_parser", f"{stats['word_count']:,} words, {stats['bullet_count']} bullets")

    return {
        "resume_doc": {**doc, "filename": filename},
        "messages": [HumanMessage(content=f"Resume parsed: {stats['word_count']} words")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — JD Analyst
# ─────────────────────────────────────────────────────────────────────────────

_JD_ANALYST_SYSTEM = """You are a technical recruiter who has screened thousands of
applications. You read a job posting and separate what the role genuinely requires
from boilerplate. You distinguish hard requirements (a screener will reject without
them) from preferences, and you name the seniority the posting implies even when it
is not stated."""


def jd_analyst_node(state: ResumeReviewState) -> Dict:
    """Turns the raw posting into a structured requirement model."""
    if state.get("error"):
        return {}

    prog = _progress(state)
    prog.start("jd_analyst")
    posting = state["job_posting"]

    prompt = f"""JOB POSTING (title: {posting['title']}):
\"\"\"
{posting['text'][:25000]}
\"\"\"

Extract the role's real requirements. Return JSON with exactly this shape:

{{
  "role_title": "normalised job title",
  "company": "company name or empty string",
  "seniority": "intern | junior | mid | senior | staff | principal | manager | director | executive",
  "years_experience": "what the posting asks for, e.g. '5-8 years' or 'not stated'",
  "location_mode": "onsite | hybrid | remote | not stated",
  "domain": "industry or business domain",
  "must_have": [
    {{"requirement": "one specific capability, skill or credential",
      "why_it_matters": "what the role uses it for",
      "keywords": ["exact terms a screener or ATS would look for"]}}
  ],
  "nice_to_have": [
    {{"requirement": "...", "keywords": ["..."]}}
  ],
  "responsibilities": ["the 5-8 things this person will actually do"],
  "soft_skills": ["collaboration / communication style the posting emphasises"],
  "screening_signals": ["what a recruiter will scan the resume for in the first 10 seconds"],
  "red_flags": ["things that would get an application rejected for this role"],
  "boilerplate_ignored": ["generic phrases you deliberately discarded"]
}}

Rules:
- 5-10 entries in must_have, ordered most to least decisive.
- Keywords must be terms that literally appear in the posting or are its standard synonyms.
- If the posting is thin, say so in boilerplate_ignored rather than inventing requirements."""

    data = _ask_json(_get_llm(state["model"]), _JD_ANALYST_SYSTEM, prompt, fallback={
        "role_title": posting["title"], "must_have": [], "nice_to_have": [],
        "responsibilities": [],
    })

    prog.finish("jd_analyst",
                f"{data.get('role_title', 'role')} — {len(data.get('must_have') or [])} must-haves, "
                f"{len(data.get('nice_to_have') or [])} nice-to-haves")

    return {
        "job_requirements": data,
        "messages": [AIMessage(content=json.dumps(data)[:4000], name="jd_analyst")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — Resume Analyst
# ─────────────────────────────────────────────────────────────────────────────

_RESUME_ANALYST_SYSTEM = """You are a resume analyst. You read a resume as written —
without reference to any particular job — and produce a faithful inventory of what
it actually evidences. You never flatter and never embellish: if a bullet describes
a responsibility rather than an achievement, you record it as a responsibility."""


def resume_analyst_node(state: ResumeReviewState) -> Dict:
    """Builds a faithful inventory of what the resume evidences."""
    if state.get("error"):
        return {}

    prog = _progress(state)
    prog.start("resume_analyst")
    doc = state["resume_doc"]

    prompt = f"""RESUME ({doc['filename']}, mechanical stats: {json.dumps(doc['stats'])}):
\"\"\"
{doc['text'][:30000]}
\"\"\"

Inventory this resume exactly as written. Return JSON with exactly this shape:

{{
  "candidate_headline": "how the resume currently presents the candidate, in one line",
  "total_years_experience": "computed from the dated roles, e.g. '~9 years (2016-2025)'",
  "current_title": "most recent title",
  "career_arc": "one or two sentences on the trajectory the dates show",
  "sections_present": ["summary", "experience", "skills", "education", "..."],
  "sections_missing": ["standard sections a recruiter expects that are absent"],
  "roles": [
    {{"title": "...", "organisation": "...", "dates": "as written",
      "scope": "team size / budget / scale if stated, else empty string"}}
  ],
  "hard_skills": ["every concrete tool, language, platform or method named"],
  "domain_experience": ["industries or business domains evidenced"],
  "quantified_achievements": ["bullets that already carry a number or measurable outcome"],
  "unquantified_claims": ["bullets that assert impact with no number — rewrite candidates"],
  "responsibility_only_bullets": ["bullets that describe duties, not outcomes"],
  "credentials": ["degrees, certifications, publications, patents"],
  "writing_issues": ["passive voice, first person, inconsistent tense, dense paragraphs, etc."],
  "employment_gaps": ["unexplained gaps over 6 months, with the dates, or empty list"]
}}

Rules:
- Quote or closely paraphrase the resume. Do not add skills it does not name.
- Leave a field as an empty list or empty string when the resume does not support it."""

    data = _ask_json(_get_llm(state["model"]), _RESUME_ANALYST_SYSTEM, prompt, fallback={
        "candidate_headline": "", "hard_skills": [], "roles": [],
    })

    prog.finish("resume_analyst",
                f"{len(data.get('hard_skills') or [])} skills, "
                f"{len(data.get('roles') or [])} roles, "
                f"{len(data.get('quantified_achievements') or [])} quantified wins")

    return {
        "resume_profile": data,
        "messages": [AIMessage(content=json.dumps(data)[:4000], name="resume_analyst")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — Fit Scorer (fan-in)
# ─────────────────────────────────────────────────────────────────────────────

_FIT_SCORER_SYSTEM = """You are the hiring manager for this role doing a first-pass
screen. You decide, requirement by requirement, whether the resume as written would
get this candidate a phone screen. You are calibrated and specific: a 'covered'
verdict must point at evidence in the resume, and a generous read of a missing
requirement is a disservice to the candidate."""


def fit_scorer_node(state: ResumeReviewState) -> Dict:
    """Scores the resume against each requirement and produces the coverage table."""
    if state.get("error"):
        return {}

    prog = _progress(state)
    prog.start("fit_scorer")

    prompt = f"""ROLE REQUIREMENTS (structured):
{json.dumps(state['job_requirements'], indent=2)[:14000]}

CANDIDATE INVENTORY (structured):
{json.dumps(state['resume_profile'], indent=2)[:12000]}

RESUME TEXT (authoritative — quote from here):
\"\"\"
{state['resume_doc']['text'][:18000]}
\"\"\"

{_HONESTY_RULE}

Screen this resume against this role. Return JSON with exactly this shape:

{{
  "fit_score": 0-100,
  "verdict": "strong | moderate | weak",
  "interview_odds": "one sentence on the realistic chance of a screen as the resume stands",
  "headline": "the single most important thing to change, in one sentence",
  "years_required": "from the posting",
  "years_evidenced": "from the resume",
  "coverage": [
    {{"requirement": "requirement text, from must_have then nice_to_have",
      "importance": "must_have | nice_to_have",
      "status": "covered | partial | missing",
      "evidence": "quote or close paraphrase from the resume, or empty string if missing",
      "action": "the specific edit or step that improves this line"}}
  ],
  "strengths": ["3-5 things that genuinely land for this role, each tied to resume evidence"],
  "risks": ["3-5 reasons a screener might pass, stated plainly"],
  "deal_breakers": ["hard requirements the candidate does not meet at all, or empty list"],
  "positioning_shift": "how the resume should re-frame the candidate's story for this role"
}}

Scoring guide — anchor to these bands:
- 80-100: every must-have covered with evidence; would be screened in.
- 60-79: all must-haves at least partial, 1-2 need surfacing; screened in after edits.
- 40-59: one or two must-haves missing; edits help but the gap is real.
- 0-39: multiple must-haves missing; this is a stretch application.

Rules:
- Include EVERY must_have and nice_to_have from the requirements, in that order.
- 'covered' requires evidence text. If you cannot quote the resume, it is 'partial' or 'missing'.
- Judge the resume as written, not the candidate's potential."""

    data = _ask_json(_get_llm(state["model"]), _FIT_SCORER_SYSTEM, prompt, fallback={
        "fit_score": 0, "verdict": "unknown", "coverage": [],
        "strengths": [], "risks": [],
    })

    # This is the screening engine — everything downstream reads its output, so a
    # parse failure here (even after _ask_json's retry) is fatal rather than
    # cascading: the existing `if state.get("error"): return {}` guard on every
    # later node means ats_auditor and bullet_rewriter no-op instead of each
    # independently gambling on the same messy input, and report_writer renders
    # the clean failure template instead of a report built on empty data.
    if data.get("_parse_failed"):
        message = (
            "The screening model could not produce a structured assessment for this "
            "resume/job pairing, even after a retry. This sometimes happens with PDF "
            "resumes that have unusual formatting (icon fonts, multi-column layouts, "
            "scanned text) — try re-exporting the resume as .docx or .txt, or run the "
            "review again."
        )
        prog.note(message, level="error")
        return {
            "fit_assessment": data,
            "error": message,
            "messages": [AIMessage(content=json.dumps(data)[:4000], name="fit_scorer")],
        }

    coverage = data.get("coverage") or []
    covered = sum(1 for c in coverage if c.get("status") == "covered")
    prog.finish("fit_scorer",
                f"fit {data.get('fit_score', '?')}/100 ({data.get('verdict', '?')}) — "
                f"{covered}/{len(coverage)} requirements covered")

    return {
        "fit_assessment": data,
        "messages": [AIMessage(content=json.dumps(data)[:4000], name="fit_scorer")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — ATS Auditor (fan-out, parallel with the rewriter)
# ─────────────────────────────────────────────────────────────────────────────

_ATS_AUDITOR_SYSTEM = """You audit resumes for the mechanical filters they must survive
before a human reads them: applicant tracking system parsing, keyword matching, and a
recruiter's ten-second scan. You know that keyword stuffing is detectable and
counter-productive, so every keyword you recommend must be placeable truthfully."""


def ats_auditor_node(state: ResumeReviewState) -> Dict:
    """Audits keyword coverage, parse-ability and recruiter scan-ability."""
    if state.get("error"):
        return {}

    prog = _progress(state)
    prog.start("ats_auditor")

    prompt = f"""ROLE REQUIREMENTS:
{json.dumps(state['job_requirements'], indent=2)[:12000]}

RESUME MECHANICAL STATS: {json.dumps(state['resume_doc']['stats'])}

RESUME TEXT (exactly as extracted — extraction artefacts are themselves signal):
\"\"\"
{state['resume_doc']['text'][:20000]}
\"\"\"

{_HONESTY_RULE}

Audit this resume. Return JSON with exactly this shape:

{{
  "ats_score": 0-100,
  "parse_quality": "one sentence on how cleanly this file parsed and what that implies",
  "present_keywords": ["role keywords the resume already contains, verbatim"],
  "missing_keywords": [
    {{"keyword": "term from the posting that is absent",
      "importance": "critical | useful",
      "where_to_add": "the exact section and line where it belongs",
      "truthful_if": "the condition under which the candidate may legitimately claim it"}}
  ],
  "terminology_mismatches": [
    {{"resume_says": "...", "posting_says": "...", "recommendation": "which term to use and why"}}
  ],
  "formatting_issues": [
    {{"issue": "...", "severity": "critical | warning | minor", "fix": "the concrete change"}}
  ],
  "section_order": ["the section order you recommend for this specific role"],
  "summary_rewrite": "a 2-3 sentence professional summary for the top of the resume, built only from what the resume evidences",
  "title_alignment": "whether and how to align the headline title with the posting, truthfully",
  "length_verdict": "one sentence: is this the right length for this seniority, and what to cut or add"
}}

Rules:
- Flag anything that breaks ATS parsing: tables, columns, text in images, headers/footers,
  graphics-only skill bars, non-standard section names, unusual date formats.
- Every missing_keyword needs a truthful_if condition. If the candidate plainly cannot
  claim the term, mark importance 'critical' and put the honest condition in truthful_if.
- The summary_rewrite must contain no claim the resume does not already support."""

    data = _ask_json(_get_llm(state["model"]), _ATS_AUDITOR_SYSTEM, prompt, fallback={
        "ats_score": 0, "present_keywords": [], "missing_keywords": [],
        "formatting_issues": [],
    })

    warnings: List[str] = []
    if data.get("_parse_failed"):
        message = "The keyword and ATS audit could not be generated this run — the report will note it."
        prog.note(message, level="warn")
        warnings.append(message)
    else:
        prog.finish("ats_auditor",
                    f"ATS {data.get('ats_score', '?')}/100 — "
                    f"{len(data.get('missing_keywords') or [])} missing keywords, "
                    f"{len(data.get('formatting_issues') or [])} formatting issues")

    return {
        "ats_audit": data,
        "warnings": warnings,
        "messages": [AIMessage(content=json.dumps(data)[:4000], name="ats_auditor")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 7 — Bullet Rewriter (fan-out, parallel with the auditor)
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_REWRITER_SYSTEM = """You are a resume editor. You rewrite real bullets from a
real resume so that they land for one specific role. Your rewrites use the
candidate's own facts, lead with outcome, name the skill the posting asks for, and
stay under two lines. You would rather return six excellent rewrites than twenty
generic ones."""


def bullet_rewriter_node(state: ResumeReviewState) -> Dict:
    """Produces before/after rewrites of actual resume lines, targeted at this role."""
    if state.get("error"):
        return {}

    prog = _progress(state)
    prog.start("bullet_rewriter")

    fit = state.get("fit_assessment") or {}

    prompt = f"""ROLE REQUIREMENTS:
{json.dumps(state['job_requirements'], indent=2)[:10000]}

SCREENING RESULT (what the hiring manager flagged):
{json.dumps({k: fit.get(k) for k in ('headline', 'coverage', 'risks', 'positioning_shift')}, indent=2)[:10000]}

RESUME TEXT (rewrite only lines that appear here, verbatim in "before"):
\"\"\"
{state['resume_doc']['text'][:20000]}
\"\"\"

{_HONESTY_RULE}

Rewrite this resume for this role. Return JSON with exactly this shape:

{{
  "rewrites": [
    {{"section": "which resume section / which role the line sits under",
      "before": "the line exactly as it appears in the resume",
      "after": "the rewritten line",
      "requirement_served": "which role requirement this now evidences",
      "why": "one sentence on what the rewrite fixes"}}
  ],
  "new_bullets_if_true": [
    {{"bullet": "a bullet the candidate should add",
      "only_if": "the fact the candidate must actually have in order to use it",
      "requirement_served": "..."}}
  ],
  "skills_section": "the full skills section as it should read, grouped and ordered for this role",
  "cut_or_compress": [
    {{"text": "the line or section to remove or shorten", "why": "what it costs to keep"}}
  ],
  "cover_letter_angle": "the two or three sentences the candidate should lead a cover letter or outreach note with",
  "upskilling": [
    {{"gap": "a genuinely missing requirement", "closes_in": "realistic effort, e.g. '2-3 weeks'",
      "how": "the specific course, project or certification that closes it"}}
  ]
}}

Rules:
- 6-12 rewrites, prioritising the lines that serve the requirements the screen flagged.
- Every "before" string must be copied verbatim from the resume text above.
- Prefer sharpening an existing bullet over adding a new one; new_bullets_if_true is for
  real experience the resume currently omits, never for experience the candidate lacks.
- upskilling is for the real gaps only; return an empty list if there are none."""

    data = _ask_json(_get_llm(state["model"]), _BULLET_REWRITER_SYSTEM, prompt, fallback={
        "rewrites": [], "new_bullets_if_true": [], "cut_or_compress": [], "upskilling": [],
    })

    warnings: List[str] = []
    if data.get("_parse_failed"):
        message = "The proposed-edits generator failed to produce output this run — the report will note it."
        prog.note(message, level="warn")
        warnings.append(message)
    else:
        prog.finish("bullet_rewriter",
                    f"{len(data.get('rewrites') or [])} rewrites, "
                    f"{len(data.get('upskilling') or [])} upskilling items")

    return {
        "bullet_rewrites": data,
        "warnings": warnings,
        "messages": [AIMessage(content=json.dumps(data)[:4000], name="bullet_rewriter")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# NODE 8 — Report Writer
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_WRITER_SYSTEM = """You write the final resume review the candidate reads. You are
direct, specific and encouraging without being soft: the candidate should finish reading
knowing exactly which edits to make, in what order, and what they cannot fix with editing
alone. You write in markdown and you never pad."""


def report_writer_node(state: ResumeReviewState) -> Dict:
    """Assembles every agent's output into the candidate-facing markdown review."""
    prog = _progress(state)

    if state.get("error"):
        report = (
            f"# Resume review could not be completed\n\n"
            f"**{state['error']}**\n\n"
            "Fix the input above and run the review again.\n"
        )
        return {"review_report": report,
                "messages": [AIMessage(content="Aborted.", name="report_writer")]}

    prog.start("report_writer")

    requirements = state["job_requirements"]
    fit = state["fit_assessment"]
    ats = state["ats_audit"]
    rewrites = state["bullet_rewrites"]

    # fit_scorer failing is fatal (handled by the error branch above), so the only
    # _parse_failed blocks that can still reach here are the two independent
    # auditors — call that out explicitly rather than relying on the model to
    # infer it from a stray `_parse_failed` key buried in a JSON dump.
    degraded = [name for name, block in (("Keywords and ATS", ats), ("Rewrites", rewrites))
                if block.get("_parse_failed")]
    degraded_note = (
        f"\nDEGRADED SECTIONS: {', '.join(degraded)} could not be generated this run "
        "(the model's output did not parse as JSON). In those sections, say plainly "
        "that this part of the review did not generate and the candidate should retry "
        "the run — do not invent content to fill the section."
        if degraded else ""
    )

    prompt = f"""Write the resume review for this candidate.

ROLE: {requirements.get('role_title')} at {requirements.get('company') or 'the hiring company'}
SOURCE: {state['job_posting'].get('url') or 'pasted job description'}
RESUME FILE: {state['resume_doc']['filename']}

SCREENING RESULT:
{json.dumps(fit, indent=2)[:12000]}

ATS & KEYWORD AUDIT:
{json.dumps(ats, indent=2)[:10000]}

PROPOSED EDITS:
{json.dumps(rewrites, indent=2)[:12000]}
{degraded_note}
{_HONESTY_RULE}

Produce a markdown report with exactly these sections:

# Resume Review - {requirements.get('role_title')}

## Verdict
The fit score, what it means, and the realistic odds as the resume stands today.
Two short paragraphs maximum.

## The five edits that matter most
A numbered list. Each item: what to change, where, and the requirement it unlocks.
Order by impact on getting a screen. Be ruthless about what makes this list.

## Requirement-by-requirement
A markdown table: Requirement | Must/Nice | Status | What to do.
Use ✅ covered, ⚠️ partial, ❌ missing — with the word, not just the symbol.

## Rewrites
For each proposed rewrite, show the before line as a blockquote and the after line in
bold, with one line on why. Group by resume section.

## Keywords and ATS
What to add, where, and the condition for claiming it truthfully. Then the formatting
fixes, most severe first.

## Suggested summary section
The rewritten professional summary, ready to paste.

## Suggested skills section
The rewritten skills section, ready to paste.

## What editing cannot fix
The real gaps, and the concrete way to close each one. If there are none, say so in one line.

## Before you apply
A short checklist the candidate can tick off.

Rules:
- Every claim traces to the inputs above. Add nothing new.
- Use the candidate's own wording in before/after pairs.
- No filler, no restating the job description back at the candidate."""

    response = _get_llm(state["model"]).invoke([
        SystemMessage(content=_REPORT_WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ])
    report = _response_text(response)

    prog.finish("report_writer", f"{len(report.split()):,} word review")

    return {
        "review_report": report,
        "messages": [AIMessage(content="Resume review complete.", name="report_writer")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESUME UPDATE — on demand, after a review completes: apply the approved edits
# and render a ready-to-send PDF. Deliberately not part of the review graph
# above — it's a second, much shorter pipeline triggered by its own button,
# not a step every review run needs to pay for. (UPDATE_NODES is defined up
# near NODES, so Progress can resolve both pipelines' node metadata.)
# ─────────────────────────────────────────────────────────────────────────────

_RESUME_BUILDER_SYSTEM = """You assemble the final, ready-to-send version of a candidate's
resume by applying only the edits a prior review already approved. You never invent
content, and you never add anything the review flagged as unconfirmed — you are
assembling an already-vetted set of changes, not writing a new resume from scratch."""


def build_enhanced_resume(
    resume_doc: Dict[str, Any],
    job_requirements: Dict[str, Any],
    ats_audit: Dict[str, Any],
    bullet_rewrites: Dict[str, Any],
    model: str,
    progress: Optional[Progress] = None,
) -> Dict[str, Any]:
    """
    Produce the structured, ready-to-render resume: the original content with the
    review's approved edits applied — rewritten bullets swapped in, the suggested
    skills section, flagged lines cut — and nothing the review did not already
    approve (no new_bullets_if_true, no invented keywords).

    Returns {"sections": {...}} on success, or {"sections": ..., "error": "..."}
    if the model's output never parsed as JSON.
    """
    prog = progress or Progress()
    prog.start("resume_builder")

    prompt = f"""ORIGINAL RESUME (authoritative — every fact must come from here):
\"\"\"
{resume_doc['text'][:30000]}
\"\"\"

TARGET ROLE: {job_requirements.get('role_title')} at {job_requirements.get('company') or 'the hiring company'}

APPROVED SUMMARY REWRITE:
{ats_audit.get('summary_rewrite', '') or '(none proposed — keep the original summary if it has one)'}

APPROVED SKILLS SECTION:
{bullet_rewrites.get('skills_section', '') or '(none proposed — keep the original skills section)'}

APPROVED BULLET REWRITES (swap "before" for "after" wherever "before" appears in the
original; leave every other bullet in the original untouched):
{json.dumps(bullet_rewrites.get('rewrites', []), indent=2)[:10000]}

LINES TO CUT OR COMPRESS:
{json.dumps(bullet_rewrites.get('cut_or_compress', []), indent=2)[:4000]}

RECOMMENDED SECTION ORDER (for guidance only — do not force a section that doesn't apply):
{json.dumps(ats_audit.get('section_order', []))}

{_HONESTY_RULE}

Assemble the complete, final resume as JSON with exactly this shape:

{{
  "contact": {{"name": "...", "headline": "current title if the resume states one, else empty string",
               "email": "...", "phone": "...", "location": "...",
               "links": ["each link exactly as it appears in the original, e.g. linkedin.com/in/..."]}},
  "summary": "the approved summary, lightly polished if needed — never add a new claim",
  "experience": [
    {{"title": "job title only", "organization": "employer name only",
      "location": "City, ST if the original states one, else empty string",
      "start_date": "MM/YYYY", "end_date": "MM/YYYY or Present", "bullets": ["..."]}}
  ],
  "skills": ["flat list of skills, from the approved skills section"],
  "education": [{{"credential": "degree / certificate name only", "institution": "school name only",
                  "location": "City, ST if stated, else empty string",
                  "end_date": "MM/YYYY or YYYY graduation date if stated, else empty string"}}],
  "additional_sections": [
    {{"heading": "e.g. CERTIFICATIONS, PROJECTS, PUBLICATIONS, LANGUAGES — anything else in the original",
      "bullets": ["..."]}}
  ]
}}

Rules:
- Every fact (employer, dates, title, credential) must come from the original resume.
- Apply every approved bullet rewrite and the approved skills section exactly as given.
- Apply every cut-or-compress instruction — omit or shorten those lines.
- Carry forward every section of the original resume not covered above (projects,
  certifications, publications, volunteer work, languages) under additional_sections —
  nothing from the original should be silently dropped.
- Do NOT add any bullet, skill, or claim that is not already in the original resume or
  in one of the approved rewrites above. If in doubt, leave it out.
- List experience newest first, exactly as the original resume orders it.
- Dates feed applicant-tracking parsers (Workday, Taleo, iCIMS) that fill From/To fields:
  write them as MM/YYYY (e.g. 03/2021), "Present" for a current role, or YYYY alone
  only when the original gives no month. Never guess a month the original doesn't state.
- Keep title, organization and location in their own fields — never combine them.
- This document goes straight to the candidate as a ready-to-send PDF — it is read as a
  resume, not as a conversation with you. If the approved summary states a total-years
  figure that conflicts with the dates in the experience section, compute the correct
  figure from those dates yourself and use it silently; do not flag the conflict in the
  text. Never insert a parenthetical explanation, caveat, or meta-commentary into the
  summary or a bullet. Never output a placeholder: if an approved rewrite contains a
  bracketed token like [X%] or [X years], rephrase that part without the number (or fall
  back to the original bullet's wording) — the file is uploaded as-is to job sites.
- Never use these characters anywhere: < > [ ] " {{ }} — Workday and similar applicant-
  tracking systems reject parsed fields that contain them. Write "under"/"over" instead
  of < and >, parentheses instead of brackets, and single quotes instead of double."""

    data = _ask_json(_get_llm(model), _RESUME_BUILDER_SYSTEM, prompt, fallback={
        "contact": {}, "summary": "", "experience": [], "skills": [],
        "education": [], "additional_sections": [],
    })

    if data.get("_parse_failed"):
        message = "Could not assemble the updated resume — the model's output did not parse. Try again."
        prog.note(message, level="error")
        return {"sections": data, "error": message}

    prog.finish("resume_builder",
                f"{len(data.get('experience') or [])} roles, "
                f"{len(data.get('skills') or [])} skills")
    return {"sections": data, "error": None}


# Typographic characters that resume parsers (Workday's in particular) mis-tokenise
# or drop: a "·"-joined contact line reads as one blob, a smart-quoted name or an
# en-dashed date range can fail to match. Plain ASCII equivalents parse everywhere.
# Workday also rejects any parsed field containing < > [ ] " { } ("Contains illegal
# characters"), so those are mapped to safe equivalents as well.
_ATS_CHAR_MAP = str.maketrans({
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": "'", "\u201d": "'", '"': "'",
    "\u00b7": "|", "\u2022": "-", "\u00a0": " ", "\u2026": "...",
    "[": "(", "]": ")", "{": "(", "}": ")",
})
_ATS_ARROW_RE = re.compile(r"\s*(?:->|=>|<-|\u2192|\u2190)\s*")
_ATS_LT_GT_RE = re.compile(r"\s*([<>])=?\s*")


def _ats_text(value: Any) -> str:
    """Normalise one line of resume text into characters every ATS form accepts."""
    text = str(value or "").translate(_ATS_CHAR_MAP)
    text = _ATS_ARROW_RE.sub(" to ", text)
    text = _ATS_LT_GT_RE.sub(lambda m: " under " if m.group(1) == "<" else " over ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _date_range(entry: Dict[str, Any]) -> str:
    """'MM/YYYY - MM/YYYY' from start/end fields, falling back to a free-text 'dates'."""
    start, end = _ats_text(entry.get("start_date")), _ats_text(entry.get("end_date"))
    if start and end:
        return f"{start} - {end}"
    return start or end or _ats_text(entry.get("dates"))


def _resume_layout(sections: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    The resume as an ordered list of (kind, text) lines, shared by the PDF and
    .docx renderers so both produce the same parser-friendly structure:

    - name alone on the first line; contact fields pipe-separated, links on
      their own line — a single "·"-joined line can't be split into fields
    - standard section headings parsers key on (WORK EXPERIENCE, EDUCATION, ...)
    - each role as three lines — title / organization | location / dates —
      instead of "Title, Company - dates", which parsers can't split reliably
      (company names contain commas) and which hides the From/To dates

    kinds: name, contact, heading, entry_title, entry_line, body, bullet, gap
    """
    lines: List[Tuple[str, str]] = []
    contact = sections.get("contact") or {}

    if contact.get("name"):
        lines.append(("name", _ats_text(contact["name"])))
    if contact.get("headline"):
        lines.append(("contact", _ats_text(contact["headline"])))
    details = " | ".join(_ats_text(v) for v in
                         [contact.get("location"), contact.get("phone"), contact.get("email")] if v)
    if details:
        lines.append(("contact", details))
    links = " | ".join(_ats_text(v) for v in contact.get("links") or [] if v)
    if links:
        lines.append(("contact", links))

    if sections.get("summary"):
        lines.append(("heading", "PROFESSIONAL SUMMARY"))
        lines.append(("body", _ats_text(sections["summary"])))

    if sections.get("experience"):
        lines.append(("heading", "WORK EXPERIENCE"))
        for i, role in enumerate(sections["experience"]):
            if i:
                lines.append(("gap", ""))
            lines.append(("entry_title", _ats_text(role.get("title"))))
            org = " | ".join(_ats_text(v) for v in
                             [role.get("organization"), role.get("location")] if v)
            if org:
                lines.append(("entry_line", org))
            dates = _date_range(role)
            if dates:
                lines.append(("entry_line", dates))
            lines.extend(("bullet", _ats_text(b)) for b in role.get("bullets") or [] if b)

    if sections.get("skills"):
        lines.append(("heading", "SKILLS"))
        skills = sections["skills"]
        text = ", ".join(_ats_text(s) for s in skills) if isinstance(skills, list) else _ats_text(skills)
        lines.append(("body", text))

    if sections.get("education"):
        lines.append(("heading", "EDUCATION"))
        for i, ed in enumerate(sections["education"]):
            if i:
                lines.append(("gap", ""))
            if ed.get("credential"):
                lines.append(("entry_title", _ats_text(ed["credential"])))
            school = " | ".join(_ats_text(v) for v in
                                [ed.get("institution"), ed.get("location")] if v)
            if school:
                lines.append(("entry_line", school))
            dates = _date_range(ed)
            if dates:
                lines.append(("entry_line", dates))

    for extra in sections.get("additional_sections") or []:
        lines.append(("heading", _ats_text(extra.get("heading") or "Additional").upper()))
        lines.extend(("bullet", _ats_text(b)) for b in extra.get("bullets") or [] if b)

    return lines


def _resume_doc_title(sections: Dict[str, Any]) -> Tuple[str, str]:
    """(author, title) for file metadata — parsers fall back to these for the name."""
    name = _ats_text((sections.get("contact") or {}).get("name"))
    return name, (f"{name} Resume" if name else "Resume")


def render_resume_pdf(sections: Dict[str, Any], out_path: str) -> str:
    """
    Render the structured resume into a clean, single-column, ATS-friendly PDF —
    plain text and real bullet lists, no tables, columns or graphics, so the
    formatting issues ats_auditor flags can't recur in the rebuilt document.
    """
    try:
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (HRFlowable, ListFlowable, ListItem, Paragraph,
                                         SimpleDocTemplate, Spacer)
    except ImportError as exc:
        raise RuntimeError(
            "Rendering the updated resume needs reportlab — `pip install reportlab` "
            "(or `pip install -r requirements-resume-optimizer.txt`)."
        ) from exc

    styles = getSampleStyleSheet()
    para_styles = {
        "name": ParagraphStyle("Name", parent=styles["Normal"], fontName="Helvetica-Bold",
                               fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=4),
        "contact": ParagraphStyle("Contact", parent=styles["Normal"], fontSize=9.5,
                                  leading=12, alignment=TA_CENTER, textColor="#444444"),
        "heading": ParagraphStyle("SectionHeading", parent=styles["Normal"],
                                  fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                                  spaceAfter=4, textColor="#1a1a1a"),
        "entry_title": ParagraphStyle("EntryTitle", parent=styles["Normal"],
                                      fontName="Helvetica-Bold", fontSize=10.5, leading=13),
        "entry_line": ParagraphStyle("EntryLine", parent=styles["Normal"], fontSize=10, leading=13),
        "body": ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=13),
        "bullet": ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=10,
                                 leading=13, spaceAfter=2),
    }

    story: List[Any] = []
    pending_bullets: List[str] = []

    def flush_bullets() -> None:
        if pending_bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(_xml_escape(b), para_styles["bullet"])) for b in pending_bullets],
                bulletType="bullet", start="•", leftIndent=14,
            ))
            pending_bullets.clear()

    for kind, text in _resume_layout(sections):
        if kind == "bullet":
            pending_bullets.append(text)
            continue
        flush_bullets()
        if kind == "gap":
            story.append(Spacer(1, 6))
        elif kind == "heading":
            story.append(Spacer(1, 8))
            story.append(HRFlowable(width="100%", thickness=0.6, color="#cccccc",
                                     spaceBefore=2, spaceAfter=6))
            story.append(Paragraph(_xml_escape(text), para_styles["heading"]))
        else:
            story.append(Paragraph(_xml_escape(text), para_styles[kind]))
    flush_bullets()

    author, title = _resume_doc_title(sections)
    doc = SimpleDocTemplate(out_path, pagesize=LETTER,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                             title=title, author=author, subject="Resume",
                             creator=author, keywords="")
    doc.build(story)
    return out_path


def render_resume_docx(sections: Dict[str, Any], out_path: str) -> str:
    """
    Render the structured resume into a .docx — the same layout as
    render_resume_pdf, laid out with python-docx instead of reportlab. This is
    also what the "Google Doc" option downloads: Google Docs opens a .docx
    natively from Drive, no separate API integration needed.

    Section headings use Word's built-in Heading 1 style (restyled to plain
    black) so parsers that read paragraph styles see real section boundaries.
    """
    try:
        import docx
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt, RGBColor
    except ImportError as exc:
        raise RuntimeError(
            "Rendering a .docx resume needs python-docx — `pip install python-docx` "
            "(or `pip install -r requirements-resume-optimizer.txt`)."
        ) from exc

    document = docx.Document()
    black = RGBColor(0, 0, 0)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(0)

    heading_style = document.styles["Heading 1"]
    heading_style.font.name = "Calibri"
    heading_style.font.size = Pt(12)
    heading_style.font.bold = True
    heading_style.font.color.rgb = black
    heading_style.paragraph_format.space_before = Pt(10)
    heading_style.paragraph_format.space_after = Pt(3)

    for kind, text in _resume_layout(sections):
        if kind == "heading":
            document.add_paragraph(text, style="Heading 1")
        elif kind == "bullet":
            document.add_paragraph(text, style="List Bullet")
        elif kind == "gap":
            document.add_paragraph()
        else:
            p = document.add_paragraph()
            run = p.add_run(text)
            if kind == "name":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run.bold = True
                run.font.size = Pt(18)
            elif kind == "contact":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            elif kind == "entry_title":
                run.bold = True

    # python-docx's template ships author "python-docx" and comment "generated by
    # python-docx" — parsers that read core properties would take those as the name.
    author, title = _resume_doc_title(sections)
    props = document.core_properties
    props.author = author
    props.last_modified_by = author
    props.title = title
    props.subject = "Resume"
    props.comments = ""
    props.keywords = ""

    document.save(out_path)
    return out_path


def generate_updated_resume(
    resume_doc: Dict[str, Any],
    job_requirements: Dict[str, Any],
    ats_audit: Dict[str, Any],
    bullet_rewrites: Dict[str, Any],
    model: Optional[str] = None,
    output_format: str = "pdf",
    out_path: Optional[str] = None,
    progress: Optional[Progress] = None,
) -> Dict[str, Any]:
    """
    Build the enhanced, ready-to-send resume from an already-completed review and
    render it in the requested format.

    Args:
        output_format: one of OUTPUT_FORMATS' ids ("pdf", "docx", "google_doc");
                        anything else falls back to "pdf"

    Returns {"sections": ..., "file_path": ... | None, "format": ..., "error": ... | None}.
    The caller owns the file at `file_path` — move or read-then-delete it.
    """
    prog = progress or Progress()
    model = resolve_model(model)
    output_format = resolve_output_format(output_format)
    ext = _OUTPUT_FORMAT_BY_ID[output_format]["ext"]

    built = build_enhanced_resume(resume_doc, job_requirements, ats_audit,
                                   bullet_rewrites, model, prog)
    if built.get("error"):
        return {"sections": built.get("sections"), "file_path": None,
                "format": output_format, "error": built["error"]}

    sections = built["sections"]
    prog.start("document_writer")

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=f".{ext}", prefix="updated_resume_")
        os.close(fd)
    try:
        (render_resume_pdf if ext == "pdf" else render_resume_docx)(sections, out_path)
    except Exception as exc:
        message = f"Could not render the {ext.upper()}: {exc.__class__.__name__}: {exc}"
        prog.note(message, level="error")
        return {"sections": sections, "file_path": None, "format": output_format, "error": message}

    prog.finish("document_writer", f"{ext.upper()} written → {os.path.basename(out_path)}")
    return {"sections": sections, "file_path": out_path, "format": output_format, "error": None}

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Build the review graph. There is no error-routing edge: every downstream node
    returns `{}` when `state["error"]` is set, so a failed intake costs no LLM
    calls and `report_writer` renders the failure instead of the review.
    """
    g = StateGraph(ResumeReviewState)

    g.add_node("job_scraper",     job_scraper_node)
    g.add_node("jd_analyst",      jd_analyst_node)
    g.add_node("resume_parser",   resume_parser_node)
    g.add_node("resume_analyst",  resume_analyst_node)
    g.add_node("fit_scorer",      fit_scorer_node)
    g.add_node("ats_auditor",     ats_auditor_node)
    g.add_node("bullet_rewriter", bullet_rewriter_node)
    g.add_node("report_writer",   report_writer_node)

    # Two independent intake branches — an HTTP fetch and a file parse — run concurrently.
    g.add_edge(START, "job_scraper")
    g.add_edge(START, "resume_parser")
    g.add_edge("job_scraper",   "jd_analyst")
    g.add_edge("resume_parser", "resume_analyst")

    # Fan-in: the screen needs both the role model and the candidate inventory.
    g.add_edge("jd_analyst",     "fit_scorer")
    g.add_edge("resume_analyst", "fit_scorer")

    # Fan-out: both auditors read the same screening result.
    g.add_edge("fit_scorer", "ats_auditor")
    g.add_edge("fit_scorer", "bullet_rewriter")

    # Fan-in: the report waits for both auditors.
    g.add_edge("ats_auditor",     "report_writer")
    g.add_edge("bullet_rewriter", "report_writer")
    g.add_edge("report_writer",   END)

    return g.compile()

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def review_resume(
    resume_path: Optional[str] = None,
    job_url: Optional[str] = None,
    job_text: Optional[str] = None,
    resume_text: Optional[str] = None,
    resume_name: Optional[str] = None,
    model: Optional[str] = None,
    progress: Optional[Progress] = None,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Run the full resume review pipeline.

    Args:
        resume_path: Path to the resume (.pdf, .docx, .txt or .md)
        job_url:     URL of the job posting
        job_text:    Pasted job description — required when the URL cannot be scraped
        resume_text: Pasted resume text — the copy/paste alternative to resume_path;
                     wins over resume_path whenever it is substantial (≥200 chars)
        resume_name: Original filename, for display (defaults to the basename, or
                     "Pasted resume" when there is no file)
        model:       Anthropic model id for every LLM call this run (see AVAILABLE_MODELS);
                     anything blank or unrecognised falls back to DEFAULT_MODEL
        progress:    Progress sink; defaults to console output
        save_report: Write the markdown review next to this script

    Returns:
        The final graph state: fit_assessment, ats_audit, bullet_rewrites,
        review_report, warnings and (on failure) error.
    """
    if not (job_url or job_text):
        raise ValueError("Provide job_url, job_text, or both.")
    if not (resume_path or resume_text):
        raise ValueError("Provide resume_path, resume_text, or both.")

    app = build_graph()

    initial_state: ResumeReviewState = {
        "job_url":          job_url,
        "job_text_input":   job_text,
        "resume_path":      resume_path,
        "resume_text_input": resume_text,
        "resume_name":      resume_name or (os.path.basename(resume_path) if resume_path else "Pasted resume"),
        "model":            resolve_model(model),
        "progress":         progress or Progress(),
        "job_posting":      None,
        "resume_doc":       None,
        "job_requirements": None,
        "resume_profile":   None,
        "fit_assessment":   None,
        "ats_audit":        None,
        "bullet_rewrites":  None,
        "review_report":    None,
        "messages":         [],
        "warnings":         [],
        "error":            None,
    }

    result = dict(app.invoke(initial_state))
    result.pop("progress", None)

    if save_report and result.get("review_report") and not result.get("error"):
        role = (result.get("job_requirements") or {}).get("role_title") or "role"
        slug = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")[:40] or "role"
        filename = f"resume_review_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(result["review_report"])
        result["report_path"] = filepath

    return result

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_USAGE = """Usage:
  python resume_optimizer_agent.py <job-posting-url> <resume.pdf|.docx|.txt> [--model MODEL_ID] [--update-resume] [--format FORMAT]
  python resume_optimizer_agent.py --jd-file <job-description.txt> <resume.pdf> [--model MODEL_ID] [--update-resume] [--format FORMAT]

Many job boards (LinkedIn, Indeed, Workday) render postings in the browser or
require a login. When a URL cannot be read, save the job description to a text
file and pass it with --jd-file.

--update-resume also applies the review's approved edits and saves a ready-to-send
document next to the markdown report (only if the review itself succeeded).
--format picks that document's format (default: pdf); "google_doc" downloads a
.docx — Google Docs opens it natively from Drive, no separate API integration.

Available --model values: {models} (default: {default})
Available --format values: {formats} (default: pdf)""".format(
    models=", ".join(m["id"] for m in AVAILABLE_MODELS),
    default=DEFAULT_MODEL,
    formats=", ".join(f["id"] for f in OUTPUT_FORMATS),
)


def _main(argv: List[str]) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.\n"
              "Run:  export ANTHROPIC_API_KEY='sk-ant-...'")
        return 1

    job_url: Optional[str] = None
    job_text: Optional[str] = None
    model: Optional[str] = None
    output_format: Optional[str] = None
    update_resume_flag = False
    positional: List[str] = []

    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg in ("-h", "--help"):
            print(_USAGE)
            return 0
        if arg == "--jd-file":
            if not args:
                print("Error: --jd-file needs a path.\n\n" + _USAGE)
                return 1
            with open(args.pop(0), "r", encoding="utf-8", errors="replace") as fh:
                job_text = fh.read()
        elif arg == "--model":
            if not args:
                print("Error: --model needs a value.\n\n" + _USAGE)
                return 1
            model = args.pop(0)
        elif arg == "--format":
            if not args:
                print("Error: --format needs a value.\n\n" + _USAGE)
                return 1
            output_format = args.pop(0)
        elif arg == "--update-resume":
            update_resume_flag = True
        else:
            positional.append(arg)

    job_url = next((a for a in positional if a.startswith(("http://", "https://"))), None)
    resume  = next((a for a in positional if not a.startswith(("http://", "https://"))), None)

    if not resume or not (job_url or job_text):
        print(_USAGE)
        return 1
    if not os.path.isfile(resume):
        print(f"Error: resume not found: {resume}")
        return 1

    model = resolve_model(model)
    sep = "=" * 68
    print(f"\n{sep}\n  RESUME OPTIMIZER — LangGraph multi-agent review\n"
          f"  Job:    {job_url or '(pasted job description)'}\n"
          f"  Resume: {resume}\n"
          f"  Model:  {model}\n{sep}\n")

    result = review_resume(resume_path=resume, job_url=job_url, job_text=job_text, model=model)

    if result.get("error"):
        print(f"\n✗ {result['error']}\n")
        return 1

    fit = result.get("fit_assessment") or {}
    print(f"\n{sep}\n  FIT SCORE: {fit.get('fit_score', '?')}/100 "
          f"({fit.get('verdict', 'unknown')})\n{sep}\n")
    print(result["review_report"])
    if result.get("report_path"):
        print(f"\nReview saved → {result['report_path']}")

    if update_resume_flag:
        resolved_format = resolve_output_format(output_format)
        print(f"\n{sep}\n  UPDATING RESUME ({resolved_format})\n{sep}\n")
        update = generate_updated_resume(
            resume_doc=result["resume_doc"],
            job_requirements=result["job_requirements"],
            ats_audit=result["ats_audit"],
            bullet_rewrites=result["bullet_rewrites"],
            model=model,
            output_format=resolved_format,
        )
        if update.get("error"):
            print(f"✗ {update['error']}")
        else:
            role = (result.get("job_requirements") or {}).get("role_title") or "role"
            slug = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")[:40] or "role"
            ext = _OUTPUT_FORMAT_BY_ID[update["format"]]["ext"]
            final_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"updated_resume_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}",
            )
            shutil.move(update["file_path"], final_path)
            print(f"Updated resume saved → {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
