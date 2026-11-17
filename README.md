# Resume Optimizer: Web App

Check your resume against a specific job posting and get it ready to send.

You give the app a job posting (a URL, or the description pasted in) and your resume. It then runs a multi-agent review built on Claude and LangGraph:

- **Fit score** and a requirement-by-requirement coverage table
- **ATS audit**: keyword gaps and formatting problems that trip up applicant tracking systems (Workday, Taleo, iCIMS)
- **Bullet rewrites**: before/after versions of your existing bullets, tailored to the role
- **Full report**: downloadable as Markdown
- **Update my resume**: applies the approved edits and downloads a new resume as PDF, Word (.docx) or a Google Docs–ready .docx

> **Honesty guardrail:** the app never invents experience. Every rewrite has to trace back to something already in your resume. Skills you don't have are listed as upskilling suggestions. They are never stuffed in as keywords.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Check with `python3 --version`. |
| **Git** | Used to clone the repository. |
| **Anthropic API key** | Create one at <https://console.anthropic.com/> under **API Keys**. Running a review uses API credits, which are billed to the account that owns the key. |
| **Internet access** | The app calls the Claude API, fetches job postings, and loads a small Markdown renderer from a CDN. |

---

## 2. Get the code

```bash
git clone https://github.com/sidman76/resume_builder.git
cd resume_builder
```

All commands below run from the repository root.

---

## 3. Create a virtual environment and install dependencies

A virtual environment keeps this app's packages separate from the rest of your system.

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-resume-optimizer.txt
```

**Windows (PowerShell)**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-resume-optimizer.txt
```

> If PowerShell blocks the activate script, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Activate the virtual environment again (`source venv/bin/activate` or `.\venv\Scripts\Activate.ps1`) each time you open a new terminal.

---

## 4. Set your API key

**Option A: `.env` file (recommended).** Create a file named `.env` in the repository root:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: the model preselected in the web UI's dropdown (default: claude-sonnet-4-6)
# ANTHROPIC_MODEL=claude-sonnet-4-6
```

The app loads this file automatically on startup. `.env` is already listed in `.gitignore`. **Never commit your API key.**

**Option B: environment variable** (lasts for the current terminal session only)

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here        # macOS / Linux
```
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"      # Windows PowerShell
```

---

## 5. Start the web app

```bash
python resume_optimizer_webapp.py
```

Open **<http://localhost:8030>** in your browser.

If the key is missing, the app prints `ANTHROPIC_API_KEY is not set` and exits. Go back to step 4.

Stop the server with **Ctrl+C**.

---

## 6. Using the app

1. **Job posting.** Do one of the following:
   - Paste the posting's **URL**, or
   - Expand the text option and **paste the full job description** (at least a couple of paragraphs).

   Pasted text takes priority over the URL whenever it's substantial. Many job boards (LinkedIn, Indeed, Workday) load postings with JavaScript or behind a login, so the app can't read them. If the review reports that it couldn't read the posting, paste the description instead.

2. **Resume.** Do one of the following:
   - **Upload** a file: `.pdf`, `.docx`, `.txt` or `.md`, up to 10 MB, or
   - **Paste** your resume text (at least ~200 characters). A substantial paste takes priority over an uploaded file.

   Scanned or image-only PDFs have no text the app can extract. Upload the `.docx` or paste the text instead.

3. **Model.** Pick a Claude model:
   - Opus: most thorough
   - Sonnet: balanced; the default is fine for most people
   - Haiku: fastest and cheapest

4. Click **Review my resume**. Each step of the pipeline lights up as it runs. A review usually takes one to a few minutes.

5. **Read the results:** fit score, requirement coverage, keyword gaps, bullet rewrites and the full report. Click **Download .md** to save the report.

6. **Update my resume** (optional). Once a review finishes successfully, choose a format and click **Update my resume**:
   - **PDF**
   - **Word (.docx)**
   - **Google Doc**: downloads a `.docx`. Upload it to Google Drive and open it with Google Docs; there is no separate Google sign-in.

   The updated resume uses a plain layout that resume parsers handle well. **Read it before you send it.** Fill in anything vague, and check that every claim is true.

---

## 7. Optional: command-line version

The same review runs without the browser:

```bash
# Job posting URL + resume file
python resume_optimizer_agent.py https://company.com/careers/123 ./resume.pdf

# Job description saved to a text file (use this when the URL can't be read)
python resume_optimizer_agent.py --jd-file jd.txt ./resume.docx

# Also generate the updated resume (pdf | docx | google_doc)
python resume_optimizer_agent.py --jd-file jd.txt ./resume.docx --update-resume --format docx

# See every option
python resume_optimizer_agent.py --help
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | The virtual environment isn't active, or the install step was skipped. Activate it and rerun `pip install -r requirements-resume-optimizer.txt`. |
| `ANTHROPIC_API_KEY is not set` | Create `.env` in the repository root or export the variable (step 4). Make sure you start the app from the repository root. |
| `401` / authentication error | The key is wrong or has been revoked. Create a new one in the Anthropic Console. |
| `credit balance is too low` | Add credits under **Billing** in the Anthropic Console. |
| Model not found / 404 from the API | Your account may not have access to that model. Choose another one in the dropdown. |
| "Couldn't read the job posting" | The site needs a login or renders with JavaScript. Paste the job description text instead. |
| Resume parsed as empty | The PDF is probably scanned or image-only. Upload a `.docx` or paste the text. |
| `Address already in use` (port 8030) | Another process is using the port. Stop it (`lsof -i :8030` on macOS/Linux), or change `port=8030` at the bottom of `resume_optimizer_webapp.py`. |
| Report shows as raw text instead of formatted | The Markdown renderer CDN couldn't load (offline or firewalled). The content is unchanged; only the formatting is affected. |

---

## 9. Privacy and limits

- The app runs **on your machine**. Your resume and the job description are sent only to the Anthropic API for analysis.
- Uploaded files go to a temporary file. Results stay in memory, and only the 50 most recent reviews are kept. Everything is gone when the server stops.
- The app has **no login or authentication**. It is meant for local, personal use. The server listens on all network interfaces (`0.0.0.0`), so other devices on your network can reach it on port 8030. Run it only on a network you trust, or change `host="0.0.0.0"` to `host="127.0.0.1"` at the bottom of `resume_optimizer_webapp.py`. Don't expose it to the internet as is.

---

## Files

| File | Purpose |
|---|---|
| `resume_optimizer_webapp.py` | FastAPI web server and browser UI (port 8030) |
| `resume_optimizer_agent.py` | LangGraph review pipeline, resume builder and CLI |
| `requirements-resume-optimizer.txt` | Python dependencies |
