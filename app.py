"""
ResearchPaper Builder
======================
Convert research materials into a journal-formatted LaTeX paper and an
Overleaf-ready ZIP package.

This is an AI-ASSISTED ASSEMBLER, not an AI paper writer. The user supplies
all real research content (problem statement, methodology, results,
references, figures, etc). Gemini 2.5 Flash-Lite is used only to organize,
polish, and structure that content into individual sections (abstract,
introduction, refined methodology, discussion, conclusion). The app never
fabricates references, data, or findings, and LaTeX is generated directly
from Python/Jinja2 templates -- never by asking the model to write LaTeX.

Deployment (Streamlit Cloud) -- requirements.txt should contain:
    streamlit
    google-generativeai
    pymupdf
    pandas
    jinja2
    bibtexparser

Set your Gemini API key as a Streamlit secret named GEMINI_API_KEY, or paste
it into the sidebar field for the current session only.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import pandas as pd
import jinja2
import bibtexparser
import zipfile
import json
import io
import re
import os
import time
from pathlib import Path
from datetime import datetime
import base64


# ============================================================================
# PAGE CONFIG -- must be the first Streamlit command executed
# ============================================================================
st.set_page_config(
    page_title="ResearchPaper Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================
GEMINI_MODEL = "gemini-2.5-flash-lite"

CITATION_STYLES = ["IEEE", "ACM", "Springer", "Elsevier", "MDPI", "Generic"]

BIB_STYLE_MAP = {
    "IEEE": "IEEEtran",
    "ACM": "ACM-Reference-Format",
    "Springer": "spmpsci",
    "Elsevier": "elsarticle-num",
    "MDPI": "mdpi",
    "Generic": "plain",
}

SYSTEM_INSTRUCTION_BASE = (
    "You are an academic writing assistant. You organize, refine, and format "
    "real research content provided by the user. You NEVER invent facts, "
    "data, citations, or findings that were not provided. If information is "
    "missing, say so explicitly instead of making it up. Respond in plain "
    "text unless explicitly asked for JSON."
)

WORKFLOW_LABELS = [
    "Paper Info",
    "Authors",
    "References",
    "Figures",
    "Research Content",
    "AI Review",
    "Generate Paper",
]


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Create every session_state key the app relies on, exactly once."""
    defaults = {
        "paper_info": {
            "title": "",
            "keywords": "",
            "target_journal": "",
            "page_limit": 8,
            "char_limit": 0,
            "citation_style": "IEEE",
        },
        "authors": [],
        "references": [],
        "figures": [],
        "tables": [],
        "case_studies": [],
        "experiment": {
            "dataset": "",
            "tools": "",
            "environment": "",
            "hardware": "",
            "metrics": "",
            "procedure": "",
        },
        "content": {
            "problem_statement": "",
            "objectives": "",
            "methodology": "",
            "results": "",
            "discussion_notes": "",
            "conclusion_notes": "",
            "future_work": "",
            "limitations": "",
        },
        "generated": {
            "research_analysis": None,
            "abstract": "",
            "introduction": "",
            "methodology_refined": "",
            "discussion": "",
            "conclusion": "",
        },
        "api_calls_made": 0,
        "api_calls_saved": 0,
        "api_key": "",
        "review_viewed": False,
        "zip_generated": False,
        "_zip_bytes": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# GENERIC HELPERS
# ============================================================================
def word_count(text):
    """Cheap word counter used for live word counts across the app."""
    if not text:
        return 0
    return len(text.split())


def get_api_key():
    """Resolve the Gemini API key from session state, then Streamlit secrets,
    then environment variables, in that priority order."""
    if st.session_state.get("api_key"):
        return st.session_state["api_key"]
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def clean_json_text(text):
    """Strip markdown code fences and isolate the JSON object/array so that
    minor formatting quirks from the model don't break json.loads()."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if not text.startswith("{") and not text.startswith("["):
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(0)
    return text


def latex_escape(text):
    """Escape LaTeX special characters in user/AI-supplied text so the
    generated .tex file compiles even when notes contain %, &, $, # etc."""
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


# ============================================================================
# GEMINI API WRAPPER -- the ONE reusable function for every model call
# ============================================================================
def call_gemini(prompt, system_instruction=None, json_mode=False, max_retries=2):
    """
    Single reusable entry point for all Gemini 2.5 Flash-Lite calls.

    Handles: authentication, retries with backoff, request timeout, and
    JSON validation when json_mode=True. Increments the session's API call
    counter on every successful call so the sidebar can report usage.

    Returns: (result, error_message). Exactly one of the two is non-None.
    """
    api_key = get_api_key()
    if not api_key:
        return None, "Missing Gemini API key. Add it in the sidebar or set GEMINI_API_KEY in Streamlit secrets."

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return None, f"Failed to configure Gemini client: {e}"

    generation_config = {"temperature": 0.3, "max_output_tokens": 2048}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_instruction,
                generation_config=generation_config,
            )
            response = model.generate_content(prompt, request_options={"timeout": 30})
            text = (getattr(response, "text", "") or "").strip()

            if not text:
                last_error = "Gemini returned an empty response."
                time.sleep(1.0 * (attempt + 1))
                continue

            if json_mode:
                cleaned = clean_json_text(text)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    last_error = "Gemini returned invalid JSON."
                    time.sleep(1.0 * (attempt + 1))
                    continue
                st.session_state.api_calls_made += 1
                return data, None

            st.session_state.api_calls_made += 1
            return text, None

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue

    return None, last_error or "Unknown error calling Gemini."


# ============================================================================
# AI MODULES -- each is a small, independent, single-purpose API call
# ============================================================================
def ai_research_analyzer(content):
    """Module 1: flags missing/weak sections in the user's raw notes."""
    prompt = f"""Analyze the research notes below and return ONLY valid JSON with keys:
problem, objectives, methodology, findings, missing_sections (array of strings).
Do not invent anything not present below. If a field is missing, set it to an
empty string and list it in missing_sections.

PROBLEM STATEMENT:
{content.get('problem_statement', '')[:1500]}

OBJECTIVES:
{content.get('objectives', '')[:800]}

METHODOLOGY:
{content.get('methodology', '')[:1500]}

RESULTS:
{content.get('results', '')[:1500]}
"""
    return call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=True)


def ai_generate_abstract(content, paper_info):
    """Module 2: abstract only, capped at 250 words."""
    prompt = f"""Write an academic abstract (maximum 250 words) for a paper titled
"{paper_info.get('title', 'Untitled')}" using ONLY the information below. Do not
add results, numbers, or claims that are not stated. Plain text only.

PROBLEM: {content.get('problem_statement', '')[:800]}
OBJECTIVES: {content.get('objectives', '')[:500]}
METHODOLOGY: {content.get('methodology', '')[:800]}
RESULTS: {content.get('results', '')[:800]}
CONCLUSION NOTES: {content.get('conclusion_notes', '')[:400]}
"""
    return call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE)


def ai_generate_introduction(content, paper_info, analysis=None):
    """Module 3: introduction only."""
    extra = ""
    if analysis and analysis.get("missing_sections"):
        extra = f"\nKnown gaps in the notes (do not fabricate to fill them): {analysis.get('missing_sections')}"
    prompt = f"""Write an academic Introduction section for a paper titled
"{paper_info.get('title', 'Untitled')}" (keywords: {paper_info.get('keywords', '')}).
Use ONLY the content below; do not fabricate background facts, statistics, or
citations. Plain text only.

PROBLEM STATEMENT: {content.get('problem_statement', '')[:1200]}
OBJECTIVES: {content.get('objectives', '')[:600]}{extra}
"""
    return call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE)


def ai_refine_methodology(methodology_text):
    """Module 4: improves clarity/structure without adding new steps."""
    prompt = f"""Improve the clarity, structure, and academic tone of the
following methodology section. Do NOT add new steps, tools, or data not
already mentioned. Plain text only.

METHODOLOGY:
{methodology_text[:2500]}
"""
    return call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE)


def ai_generate_discussion(results_text, discussion_notes):
    """Module 5: discussion based strictly on provided results/notes."""
    prompt = f"""Write an academic Discussion section based ONLY on the
results and notes below. Do not introduce new findings or numbers. Plain
text only.

RESULTS: {results_text[:1500]}
DISCUSSION NOTES: {discussion_notes[:1200]}
"""
    return call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE)


def ai_generate_conclusion(content):
    """Module 6: conclusion based strictly on provided notes."""
    prompt = f"""Write an academic Conclusion section using ONLY the
information below. Do not introduce new claims. Plain text only.

OBJECTIVES: {content.get('objectives', '')[:500]}
RESULTS: {content.get('results', '')[:800]}
CONCLUSION NOTES: {content.get('conclusion_notes', '')[:600]}
FUTURE WORK: {content.get('future_work', '')[:400]}
LIMITATIONS: {content.get('limitations', '')[:400]}
"""
    return call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE)


# ============================================================================
# PDF / REFERENCE HELPERS (no AI involved -- pure extraction, never invented)
# ============================================================================
def extract_pdf_text(file_bytes):
    """Extract raw text + whatever metadata PyMuPDF can read from a PDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text() for page in doc]
        full_text = "\n".join(text_parts).strip()
        metadata = doc.metadata or {}
        page_count = doc.page_count
        doc.close()
        return {
            "text": full_text,
            "title": metadata.get("title", "") or "",
            "author": metadata.get("author", "") or "",
            "page_count": page_count,
        }, None
    except Exception as e:
        return None, f"Could not read PDF: {e}"


@st.cache_data(show_spinner=False)
def cached_pdf_extract(file_bytes):
    """Cache PDF extraction so re-rendering the page never re-parses the
    same file -- this is the 'cache extracted PDFs' cost-optimization rule."""
    return extract_pdf_text(file_bytes)


def parse_bibtex_file(file_bytes):
    """Parse an uploaded .bib file into reference dicts. Pure parsing, no AI,
    so nothing is ever invented."""
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        bib_db = bibtexparser.loads(text)
        entries = []
        for entry in bib_db.entries:
            entries.append({
                "key": entry.get("ID", f"ref{len(entries) + 1}"),
                "entry_type": entry.get("ENTRYTYPE", "misc"),
                "title": entry.get("title", ""),
                "authors": entry.get("author", ""),
                "year": entry.get("year", ""),
                "venue": entry.get("journal", entry.get("booktitle", "")),
                "doi": entry.get("doi", ""),
                "url": entry.get("url", ""),
                "source_type": "bibtex",
            })
        return entries, None
    except Exception as e:
        return None, f"Could not parse BibTeX file: {e}"


def reference_to_bibtex_entry(ref):
    """Render one reference dict as a clean BibTeX entry block."""
    entry_type = ref.get("entry_type") or "misc"
    key = ref.get("key") or "ref"
    field_map = [
        ("title", "title"),
        ("author", "authors"),
        ("year", "year"),
        ("journal", "venue"),
        ("doi", "doi"),
        ("url", "url"),
    ]
    lines = []
    for bib_field, ref_field in field_map:
        value = ref.get(ref_field, "")
        if value:
            lines.append(f"  {bib_field} = {{{value}}}")
    body = ",\n".join(lines)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def generate_references_bib():
    """Concatenate every stored reference into a references.bib string."""
    refs = st.session_state.references
    if not refs:
        return "% No references added yet.\n"
    return "\n\n".join(reference_to_bibtex_entry(r) for r in refs)


# ============================================================================
# FIGURE HELPERS
# ============================================================================
def get_image_mime(filename):
    ext = Path(filename).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")


def render_image_preview(file_bytes, filename):
    """Preview an uploaded figure. SVGs are embedded as base64 <img> since
    st.image() does not render vector graphics directly."""
    mime = get_image_mime(filename)
    if mime == "image/svg+xml":
        b64 = base64.b64encode(file_bytes).decode()
        st.markdown(
            f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:100%;" />',
            unsafe_allow_html=True,
        )
    else:
        st.image(file_bytes, use_container_width=True)


# ============================================================================
# TABLE HELPERS
# ============================================================================
def df_to_latex_table(df, caption, label):
    """Convert an editable dataframe into a LaTeX table block."""
    if df is None or df.empty:
        return "% Empty table skipped.\n"
    col_spec = "l" * len(df.columns)
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    body_rows = [" & ".join(str(v) for v in row.values) + r" \\" for _, row in df.iterrows()]
    body = "\n".join(body_rows)
    safe_label = re.sub(r"\s+", "_", (label or "table").lower())
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{latex_escape(caption)}}}
\\label{{tab:{safe_label}}}
\\begin{{tabular}}{{{col_spec}}}
\\hline
{header}
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}"""


# ============================================================================
# LATEX TEMPLATES (Jinja2, custom delimiters so LaTeX braces never collide
# with Jinja2 syntax). Generated directly in Python -- never by Gemini.
# ============================================================================
latex_jinja_env = jinja2.Environment(
    block_start_string="<%",
    block_end_string="%>",
    variable_start_string="<<",
    variable_end_string=">>",
    comment_start_string="<#",
    comment_end_string="#>",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.BaseLoader(),
)

GENERIC_TEMPLATE = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{<< title >>}
\author{
<% for author in authors %>
<< author.name >>\\
\small << author.affiliation >>, << author.department >>\\
\small \texttt{<< author.email >>}<% if not loop.last %>\\[1em]<% endif %>
<% endfor %>
}
\date{}

\begin{document}
\maketitle

\begin{abstract}
<< abstract >>
\end{abstract}

\noindent\textbf{Keywords:} << keywords >>

\section{Introduction}
<< introduction >>

\section{Methodology}
<< methodology >>

\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>

\section{Discussion}
<< discussion >>

\section{Conclusion}
<< conclusion >>
<% if future_work %>

\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>

\section{Limitations}
<< limitations >>
<% endif %>

\bibliographystyle{<< bib_style >>}
\bibliography{references}

\end{document}
"""

IEEE_TEMPLATE = r"""
\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}

\begin{document}

\title{<< title >>}

\author{
<% for author in authors %>
\IEEEauthorblockN{<< author.name >>}
\IEEEauthorblockA{<< author.department >>, << author.affiliation >>\\
Email: << author.email >>}<% if not loop.last %>
\and
<% endif %>
<% endfor %>
}

\maketitle

\begin{abstract}
<< abstract >>
\end{abstract}

\begin{IEEEkeywords}
<< keywords >>
\end{IEEEkeywords}

\section{Introduction}
<< introduction >>

\section{Methodology}
<< methodology >>

\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\columnwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>

\section{Discussion}
<< discussion >>

\section{Conclusion}
<< conclusion >>
<% if future_work %>

\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>

\section{Limitations}
<< limitations >>
<% endif %>

\bibliographystyle{<< bib_style >>}
\bibliography{references}

\end{document}
"""

ACM_TEMPLATE = r"""
\documentclass[sigconf]{acmart}
\usepackage{graphicx}

\begin{document}

\title{<< title >>}

<% for author in authors %>
\author{<< author.name >>}
\affiliation{
  \institution{<< author.affiliation >>}
  \department{<< author.department >>}
}
\email{<< author.email >>}
<% endfor %>

\begin{abstract}
<< abstract >>
\end{abstract}

\keywords{<< keywords >>}

\maketitle

\section{Introduction}
<< introduction >>

\section{Methodology}
<< methodology >>

\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\linewidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>

\section{Discussion}
<< discussion >>

\section{Conclusion}
<< conclusion >>
<% if future_work %>

\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>

\section{Limitations}
<< limitations >>
<% endif %>

\bibliographystyle{<< bib_style >>}
\bibliography{references}

\end{document}
"""

SPRINGER_TEMPLATE = r"""
\documentclass[smallcondensed]{svjour3}
\usepackage{graphicx}
\usepackage{amsmath}

\journalname{<< target_journal >>}

\begin{document}

\title{<< title >>}

\author{
<% for author in authors %>
<< author.name >><% if not loop.last %> \and <% endif %>
<% endfor %>
}

\institute{
<% for author in authors %>
<< author.affiliation >>, << author.department >> \email{<< author.email >>}\\
<% endfor %>
}

\date{}

\maketitle

\begin{abstract}
<< abstract >>
\keywords{<< keywords >>}
\end{abstract}

\section{Introduction}
<< introduction >>

\section{Methodology}
<< methodology >>

\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>

\section{Discussion}
<< discussion >>

\section{Conclusion}
<< conclusion >>
<% if future_work %>

\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>

\section{Limitations}
<< limitations >>
<% endif %>

\bibliographystyle{<< bib_style >>}
\bibliography{references}

\end{document}
"""

ELSEVIER_TEMPLATE = r"""
\documentclass[preprint,12pt]{elsarticle}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{lineno}

\journal{<< target_journal >>}

\begin{document}

\begin{frontmatter}

\title{<< title >>}

<% for author in authors %>
\author<% if author.corresponding %>[cor]<% endif %>{<< author.name >>}
\address{<< author.affiliation >>, << author.department >>}
<% endfor %>
<% for author in authors %>
<% if author.corresponding %>
\cortext[cor]{Corresponding author. Email: << author.email >>}
<% endif %>
<% endfor %>

\begin{abstract}
<< abstract >>
\end{abstract}

\begin{keyword}
<< keywords >>
\end{keyword}

\end{frontmatter}

\section{Introduction}
<< introduction >>

\section{Methodology}
<< methodology >>

\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>

\section{Discussion}
<< discussion >>

\section{Conclusion}
<< conclusion >>
<% if future_work %>

\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>

\section{Limitations}
<< limitations >>
<% endif %>

\bibliographystyle{<< bib_style >>}
\bibliography{references}

\end{document}
"""

TEMPLATE_MAP = {
    "IEEE": IEEE_TEMPLATE,
    "ACM": ACM_TEMPLATE,
    "Springer": SPRINGER_TEMPLATE,
    "Elsevier": ELSEVIER_TEMPLATE,
    "Generic": GENERIC_TEMPLATE,
}


def build_latex_context():
    """Gather everything from session_state into the dict the active
    journal template needs, escaping text fields along the way."""
    pi = st.session_state.paper_info
    citation_style = pi.get("citation_style", "Generic")
    journal_key = citation_style if citation_style in TEMPLATE_MAP else "Generic"

    authors_ctx = [{
        "name": latex_escape(a.get("name", "")),
        "affiliation": latex_escape(a.get("affiliation", "")),
        "department": latex_escape(a.get("department", "")),
        "email": latex_escape(a.get("email", "")),
        "corresponding": a.get("corresponding", False),
    } for a in st.session_state.authors]

    figures_ctx = [{
        "filename": fig.get("filename", ""),
        "caption": latex_escape(fig.get("caption", "")),
        "number": fig.get("number", ""),
    } for fig in st.session_state.figures]

    tables_ctx = [{
        "latex": df_to_latex_table(t.get("df"), t.get("title", "Table"), t.get("label", t.get("title", "table"))),
    } for t in st.session_state.tables]

    content = st.session_state.content
    generated = st.session_state.generated

    context = {
        "title": latex_escape(pi.get("title", "Untitled Paper")),
        "keywords": latex_escape(pi.get("keywords", "")),
        "target_journal": latex_escape(pi.get("target_journal", "")),
        "authors": authors_ctx,
        "abstract": latex_escape(generated.get("abstract") or "Abstract not yet generated."),
        "introduction": latex_escape(generated.get("introduction") or content.get("problem_statement", "")),
        "methodology": latex_escape(generated.get("methodology_refined") or content.get("methodology", "")),
        "results": latex_escape(content.get("results", "")),
        "discussion": latex_escape(generated.get("discussion") or content.get("discussion_notes", "")),
        "conclusion": latex_escape(generated.get("conclusion") or content.get("conclusion_notes", "")),
        "future_work": latex_escape(content.get("future_work", "")),
        "limitations": latex_escape(content.get("limitations", "")),
        "figures": figures_ctx,
        "tables": tables_ctx,
        "bib_style": BIB_STYLE_MAP.get(citation_style, "plain"),
    }
    return context, journal_key


def generate_main_tex():
    context, journal_key = build_latex_context()
    template = latex_jinja_env.from_string(TEMPLATE_MAP[journal_key])
    return template.render(**context)


def generate_metadata_json():
    data = {
        "paper_info": st.session_state.paper_info,
        "authors": st.session_state.authors,
        "references_count": len(st.session_state.references),
        "figures_count": len(st.session_state.figures),
        "tables_count": len(st.session_state.tables),
        "generated_on": datetime.now().isoformat(),
        "generated_sections": {k: bool(v) for k, v in st.session_state.generated.items()},
    }
    return json.dumps(data, indent=2)


def generate_readme():
    style = st.session_state.paper_info.get("citation_style", "Generic")
    return f"""ResearchPaper Builder Export
=============================

This package contains:
- main.tex          LaTeX source for your paper ({style} template)
- references.bib    Bibliography file
- figures/          All uploaded figure files
- metadata.json     Paper metadata snapshot

HOW TO USE WITH OVERLEAF
1. Go to https://www.overleaf.com and create a New Project.
2. Choose "Upload Project" and select this ZIP file.
3. Overleaf detects main.tex automatically -- click Recompile.
4. Overleaf's full TeX Live distribution already includes the document
   classes used by these templates (IEEEtran, acmart, elsarticle, svjour3),
   so no extra package installation should be needed.

Generated by ResearchPaper Builder.
"""


def build_zip_package():
    """Assemble the final downloadable Overleaf-ready ZIP in memory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", generate_main_tex())
        zf.writestr("references.bib", generate_references_bib())
        zf.writestr("metadata.json", generate_metadata_json())
        zf.writestr("README.txt", generate_readme())
        for fig in st.session_state.figures:
            zf.writestr(f"figures/{fig['filename']}", fig["bytes"])
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# SIDEBAR
# ============================================================================
def compute_progress():
    pi = st.session_state.paper_info
    content = st.session_state.content
    return {
        "Paper Info": bool(pi.get("title")) and bool(pi.get("keywords")),
        "Authors": len(st.session_state.authors) > 0,
        "References": len(st.session_state.references) > 0,
        "Figures": len(st.session_state.figures) > 0,
        "Research Content": bool(content.get("problem_statement")) and bool(content.get("methodology")),
        "AI Review": st.session_state.get("review_viewed", False),
        "Generate Paper": st.session_state.get("zip_generated", False),
    }


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📄 ResearchPaper Builder")
        st.caption("Research materials → journal-formatted LaTeX + Overleaf ZIP.")

        st.divider()
        st.markdown("### Workflow Progress")
        steps = compute_progress()
        st.progress(sum(steps.values()) / len(steps))
        for name in WORKFLOW_LABELS:
            icon = "✅" if steps.get(name) else "⬜"
            st.markdown(f"{icon} {name}")

        st.divider()
        st.markdown("### Gemini API")
        st.session_state.api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            help="Stored only for this session. Alternatively set GEMINI_API_KEY in Streamlit secrets.",
        )
        if get_api_key():
            st.success("API key detected", icon="✅")
        else:
            st.warning("No API key set", icon="⚠️")

        st.caption(f"Gemini calls made this session: **{st.session_state.api_calls_made}**")
        st.caption(f"Calls saved by reusing cache: **{st.session_state.api_calls_saved}**")

        st.divider()
        with st.expander("💡 Token-Saving Tips"):
            st.markdown(
                "- Fill in **Research Content** fully before generating AI sections.\n"
                "- Click **Use Cached** instead of **Regenerate** when a result already looks good.\n"
                "- Generate sections one at a time -- you rarely need all six at once.\n"
                "- Keep notes concise; shorter input means cheaper, faster calls.\n"
                "- The app never resends your whole paper, only the section you're working on."
            )


# ============================================================================
# STEP 1 -- PAPER INFORMATION
# ============================================================================
def render_paper_info():
    st.subheader("Step 1 · Paper Information")
    pi = st.session_state.paper_info
    col1, col2 = st.columns(2)
    with col1:
        pi["title"] = st.text_input("Paper Title", value=pi.get("title", ""))
        pi["keywords"] = st.text_input("Keywords (comma-separated)", value=pi.get("keywords", ""))
        pi["target_journal"] = st.text_input("Target Journal", value=pi.get("target_journal", ""))
    with col2:
        pi["citation_style"] = st.selectbox(
            "Citation Style / Template",
            CITATION_STYLES,
            index=CITATION_STYLES.index(pi.get("citation_style", "IEEE")),
        )
        pi["page_limit"] = st.number_input("Page Limit", min_value=1, max_value=100, value=int(pi.get("page_limit", 8)))
        pi["char_limit"] = st.number_input("Character Limit (0 = none)", min_value=0, value=int(pi.get("char_limit", 0)))
    st.session_state.paper_info = pi

    if pi.get("title") and pi.get("keywords"):
        st.success("Paper info looks complete.")
    else:
        st.info("Add at least a title and keywords to mark this step complete.")


# ============================================================================
# STEP 2 -- AUTHORS
# ============================================================================
def render_authors():
    st.subheader("Step 2 · Authors")
    for i, author in enumerate(st.session_state.authors):
        with st.expander(f"Author {i + 1}: {author.get('name') or 'Unnamed'}"):
            author["name"] = st.text_input("Author Name", value=author.get("name", ""), key=f"auth_name_{i}")
            author["affiliation"] = st.text_input("Affiliation", value=author.get("affiliation", ""), key=f"auth_aff_{i}")
            author["department"] = st.text_input("Department", value=author.get("department", ""), key=f"auth_dept_{i}")
            author["email"] = st.text_input("Email", value=author.get("email", ""), key=f"auth_email_{i}")
            author["corresponding"] = st.checkbox("Corresponding Author", value=author.get("corresponding", False), key=f"auth_corr_{i}")
            if st.button("🗑️ Remove Author", key=f"auth_remove_{i}"):
                st.session_state.authors.pop(i)
                st.rerun()

    if st.button("➕ Add Author"):
        st.session_state.authors.append({
            "name": "", "affiliation": "", "department": "", "email": "", "corresponding": False,
        })
        st.rerun()


# ============================================================================
# STEP 3 -- RESEARCH CONTENT
# ============================================================================
def render_research_content():
    st.subheader("Step 3 · Research Content")
    c = st.session_state.content
    fields = [
        ("problem_statement", "Problem Statement"),
        ("objectives", "Objectives"),
        ("methodology", "Methodology"),
        ("results", "Results"),
        ("discussion_notes", "Discussion Notes"),
        ("conclusion_notes", "Conclusion Notes"),
        ("future_work", "Future Work"),
        ("limitations", "Limitations"),
    ]
    for key, label in fields:
        c[key] = st.text_area(label, value=c.get(key, ""), height=120, key=f"content_{key}")
        st.caption(f"Word count: {word_count(c[key])}")
    st.session_state.content = c


# ============================================================================
# STEP 4 -- REFERENCES
# ============================================================================
def render_references():
    st.subheader("Step 4 · References")
    method = st.radio("Add reference via", ["PDF Upload", "DOI", "URL", "BibTeX Upload"], horizontal=True)

    if method == "PDF Upload":
        pdf_file = st.file_uploader("Upload reference PDF", type=["pdf"], key="ref_pdf")
        if pdf_file is not None:
            extracted, err = cached_pdf_extract(pdf_file.getvalue())
            if err:
                st.error(err)
            else:
                st.markdown("**Extracted metadata -- verify before adding, nothing is auto-trusted:**")
                title = st.text_input("Title", value=extracted.get("title", ""), key="pdf_ref_title")
                authors = st.text_input("Authors", value=extracted.get("author", ""), key="pdf_ref_authors")
                year = st.text_input("Year", value="", key="pdf_ref_year")
                if st.button("Add Reference from PDF"):
                    st.session_state.references.append({
                        "key": f"ref{len(st.session_state.references) + 1}",
                        "entry_type": "article",
                        "title": title, "authors": authors, "year": year,
                        "venue": "", "doi": "", "url": "", "source_type": "pdf",
                    })
                    st.success("Reference added.")
                    st.rerun()

    elif method == "DOI":
        st.caption("Live DOI lookup is disabled to avoid fabricated metadata -- please confirm details manually.")
        doi = st.text_input("DOI", key="doi_input")
        title = st.text_input("Title", key="doi_title")
        authors = st.text_input("Authors", key="doi_authors")
        year = st.text_input("Year", key="doi_year")
        venue = st.text_input("Journal / Venue", key="doi_venue")
        if st.button("Add DOI Reference"):
            if doi and title:
                st.session_state.references.append({
                    "key": f"ref{len(st.session_state.references) + 1}",
                    "entry_type": "article",
                    "title": title, "authors": authors, "year": year,
                    "venue": venue, "doi": doi, "url": "", "source_type": "doi",
                })
                st.success("Reference added.")
                st.rerun()
            else:
                st.warning("DOI and Title are required.")

    elif method == "URL":
        url = st.text_input("URL", key="url_input")
        title = st.text_input("Title", key="url_title")
        authors = st.text_input("Authors", key="url_authors")
        year = st.text_input("Year", key="url_year")
        if st.button("Add URL Reference"):
            if url and title:
                st.session_state.references.append({
                    "key": f"ref{len(st.session_state.references) + 1}",
                    "entry_type": "misc",
                    "title": title, "authors": authors, "year": year,
                    "venue": "", "doi": "", "url": url, "source_type": "url",
                })
                st.success("Reference added.")
                st.rerun()
            else:
                st.warning("URL and Title are required.")

    else:  # BibTeX Upload
        bib_file = st.file_uploader("Upload .bib file", type=["bib"], key="bib_upload")
        if bib_file is not None:
            entries, err = parse_bibtex_file(bib_file.getvalue())
            if err:
                st.error(err)
            else:
                st.success(f"Found {len(entries)} entries.")
                if st.button("Add All BibTeX Entries"):
                    st.session_state.references.extend(entries)
                    st.success("Entries added.")
                    st.rerun()

    st.divider()
    st.markdown(f"**Current References ({len(st.session_state.references)})**")
    for i, ref in enumerate(st.session_state.references):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"`{ref['key']}` — {ref.get('title', '(untitled)')} ({ref.get('year', 'n.d.')}) · _{ref.get('source_type')}_")
        with col2:
            if st.button("🗑️", key=f"ref_del_{i}"):
                st.session_state.references.pop(i)
                st.rerun()


# ============================================================================
# STEP 5 -- FIGURES AND CHARTS
# ============================================================================
def render_figures():
    st.subheader("Step 5 · Figures and Charts")
    uploaded = st.file_uploader("Upload figure", type=["png", "jpg", "jpeg", "svg"], key="fig_upload")
    if uploaded is not None:
        render_image_preview(uploaded.getvalue(), uploaded.name)
        title = st.text_input("Figure Title", key="fig_title_input")
        caption = st.text_area("Figure Caption", key="fig_caption_input", height=80)
        number = st.text_input("Figure Number", value=str(len(st.session_state.figures) + 1), key="fig_number_input")
        if st.button("Add Figure"):
            st.session_state.figures.append({
                "title": title, "caption": caption, "number": number,
                "filename": uploaded.name, "bytes": uploaded.getvalue(),
            })
            st.success("Figure added.")
            st.rerun()

    st.divider()
    st.markdown(f"**Current Figures ({len(st.session_state.figures)})**")
    for i, fig in enumerate(st.session_state.figures):
        with st.expander(f"Figure {fig.get('number')}: {fig.get('title') or fig['filename']}"):
            render_image_preview(fig["bytes"], fig["filename"])
            st.caption(fig.get("caption") or "_No caption_")
            if st.button("Remove", key=f"fig_del_{i}"):
                st.session_state.figures.pop(i)
                st.rerun()


# ============================================================================
# STEP 6 -- TABLES
# ============================================================================
def render_tables():
    st.subheader("Step 6 · Tables")
    st.caption("Create a table manually. It will be exported as a LaTeX table.")
    title = st.text_input("Table Title", key="table_title_input")
    default_df = pd.DataFrame({"Column 1": ["", ""], "Column 2": ["", ""]})
    edited_df = st.data_editor(default_df, num_rows="dynamic", key="table_editor")
    if st.button("Add Table"):
        if title:
            st.session_state.tables.append({"title": title, "df": edited_df.copy(), "label": title})
            st.success("Table added.")
            st.rerun()
        else:
            st.warning("Give the table a title first.")

    st.divider()
    st.markdown(f"**Current Tables ({len(st.session_state.tables)})**")
    for i, t in enumerate(st.session_state.tables):
        with st.expander(t["title"]):
            st.dataframe(t["df"], use_container_width=True)
            st.code(df_to_latex_table(t["df"], t["title"], t["label"]), language="latex")
            if st.button("Remove Table", key=f"table_del_{i}"):
                st.session_state.tables.pop(i)
                st.rerun()


# ============================================================================
# STEP 7 -- CASE STUDIES
# ============================================================================
def render_case_studies():
    st.subheader("Step 7 · Case Studies")
    method = st.radio("Add case study via", ["Upload PDF", "Enter Text"], horizontal=True, key="case_method")

    if method == "Upload PDF":
        pdf_file = st.file_uploader("Upload case study PDF", type=["pdf"], key="case_pdf")
        if pdf_file is not None:
            extracted, err = cached_pdf_extract(pdf_file.getvalue())
            if err:
                st.error(err)
            else:
                title = st.text_input("Case Study Title", key="case_pdf_title")
                if st.button("Add Case Study (PDF)"):
                    st.session_state.case_studies.append({
                        "title": title or pdf_file.name, "content": extracted["text"],
                        "source": "pdf", "filename": pdf_file.name,
                    })
                    st.success("Case study added.")
                    st.rerun()
    else:
        title = st.text_input("Case Study Title", key="case_text_title")
        content = st.text_area("Case Study Content", height=180, key="case_text_content")
        if st.button("Add Case Study (Text)"):
            if title and content:
                st.session_state.case_studies.append({
                    "title": title, "content": content, "source": "text", "filename": None,
                })
                st.success("Case study added.")
                st.rerun()
            else:
                st.warning("Title and content are required.")

    st.divider()
    for i, cs in enumerate(st.session_state.case_studies):
        with st.expander(cs["title"]):
            preview = cs["content"][:2000] + ("..." if len(cs["content"]) > 2000 else "")
            st.write(preview)
            if st.button("Remove", key=f"case_del_{i}"):
                st.session_state.case_studies.pop(i)
                st.rerun()


# ============================================================================
# STEP 8 -- EXPERIMENT DETAILS
# ============================================================================
def render_experiment():
    st.subheader("Step 8 · Experiment Details")
    e = st.session_state.experiment
    e["dataset"] = st.text_input("Dataset", value=e.get("dataset", ""))
    e["tools"] = st.text_input("Tools", value=e.get("tools", ""))
    e["environment"] = st.text_input("Environment", value=e.get("environment", ""))
    e["hardware"] = st.text_input("Hardware", value=e.get("hardware", ""))
    e["metrics"] = st.text_input("Metrics", value=e.get("metrics", ""))
    e["procedure"] = st.text_area("Procedure", value=e.get("procedure", ""), height=150)
    st.session_state.experiment = e


# ============================================================================
# AI GENERATION TAB -- one button = one Gemini call, with cache reuse
# ============================================================================
def _module_buttons(section_key):
    """Render Generate/Regenerate + Use-Cached buttons for one AI module."""
    has_cache = bool(st.session_state.generated.get(section_key))
    col1, col2 = st.columns(2)
    with col1:
        label = "🔄 Regenerate" if has_cache else "▶️ Generate"
        if st.button(label, key=f"gen_btn_{section_key}"):
            st.session_state[f"_trigger_{section_key}"] = True
    with col2:
        if has_cache:
            if st.button("✅ Use Cached (saves 1 call)", key=f"cache_btn_{section_key}"):
                st.session_state.api_calls_saved += 1
                st.toast("Using cached result -- no API call made.")


def render_ai_generation():
    st.subheader("AI Generation")
    st.caption("Each module makes exactly one Gemini call per click, sending only the section of your content it needs.")

    content = st.session_state.content
    pi = st.session_state.paper_info
    gen = st.session_state.generated

    with st.expander("🔎 Research Analyzer", expanded=True):
        st.caption("Checks your raw notes for completeness before you generate anything else.")
        _module_buttons("research_analysis")
        if st.session_state.get("_trigger_research_analysis"):
            with st.spinner("Analyzing research content..."):
                result, err = ai_research_analyzer(content)
            if err:
                st.error(err)
            else:
                gen["research_analysis"] = result
            st.session_state["_trigger_research_analysis"] = False
        if gen.get("research_analysis"):
            st.json(gen["research_analysis"])

    with st.expander("📝 Abstract Generator"):
        _module_buttons("abstract")
        if st.session_state.get("_trigger_abstract"):
            with st.spinner("Writing abstract..."):
                result, err = ai_generate_abstract(content, pi)
            if err:
                st.error(err)
            else:
                gen["abstract"] = result
            st.session_state["_trigger_abstract"] = False
        if gen.get("abstract"):
            gen["abstract"] = st.text_area("Abstract (editable)", value=gen["abstract"], height=150, key="abstract_edit")
            st.caption(f"Word count: {word_count(gen['abstract'])} / 250")

    with st.expander("📖 Introduction Generator"):
        _module_buttons("introduction")
        if st.session_state.get("_trigger_introduction"):
            with st.spinner("Writing introduction..."):
                result, err = ai_generate_introduction(content, pi, gen.get("research_analysis"))
            if err:
                st.error(err)
            else:
                gen["introduction"] = result
            st.session_state["_trigger_introduction"] = False
        if gen.get("introduction"):
            gen["introduction"] = st.text_area("Introduction (editable)", value=gen["introduction"], height=200, key="intro_edit")
            st.caption(f"Word count: {word_count(gen['introduction'])}")

    with st.expander("🧪 Methodology Refiner"):
        _module_buttons("methodology_refined")
        if st.session_state.get("_trigger_methodology_refined"):
            with st.spinner("Refining methodology..."):
                result, err = ai_refine_methodology(content.get("methodology", ""))
            if err:
                st.error(err)
            else:
                gen["methodology_refined"] = result
            st.session_state["_trigger_methodology_refined"] = False
        if gen.get("methodology_refined"):
            gen["methodology_refined"] = st.text_area("Methodology (editable)", value=gen["methodology_refined"], height=200, key="method_edit")

    with st.expander("💬 Discussion Generator"):
        _module_buttons("discussion")
        if st.session_state.get("_trigger_discussion"):
            with st.spinner("Writing discussion..."):
                result, err = ai_generate_discussion(content.get("results", ""), content.get("discussion_notes", ""))
            if err:
                st.error(err)
            else:
                gen["discussion"] = result
            st.session_state["_trigger_discussion"] = False
        if gen.get("discussion"):
            gen["discussion"] = st.text_area("Discussion (editable)", value=gen["discussion"], height=200, key="disc_edit")

    with st.expander("🏁 Conclusion Generator"):
        _module_buttons("conclusion")
        if st.session_state.get("_trigger_conclusion"):
            with st.spinner("Writing conclusion..."):
                result, err = ai_generate_conclusion(content)
            if err:
                st.error(err)
            else:
                gen["conclusion"] = result
            st.session_state["_trigger_conclusion"] = False
        if gen.get("conclusion"):
            gen["conclusion"] = st.text_area("Conclusion (editable)", value=gen["conclusion"], height=200, key="concl_edit")

    st.session_state.generated = gen


# ============================================================================
# AI REVIEW PANEL -- rule-based, zero API calls
# ============================================================================
def render_ai_review():
    st.subheader("AI Review Panel")
    st.caption("Rule-based checks -- no API call needed.")
    st.session_state["review_viewed"] = True

    content = st.session_state.content
    pi = st.session_state.paper_info
    gen = st.session_state.generated
    warnings, successes = [], []

    if not pi.get("title"):
        warnings.append("Missing paper title.")
    if not pi.get("keywords"):
        warnings.append("Missing keywords.")
    else:
        successes.append("Keywords present.")

    if word_count(content.get("methodology", "")) < 50:
        warnings.append("Methodology looks thin (under 50 words) -- consider expanding before refining with AI.")
    else:
        successes.append("Methodology has reasonable length.")

    if not st.session_state.references:
        warnings.append("No references added yet.")
    else:
        successes.append(f"{len(st.session_state.references)} reference(s) added.")

    missing_captions = [f for f in st.session_state.figures if not f.get("caption")]
    if missing_captions:
        warnings.append(f"{len(missing_captions)} figure(s) missing captions.")

    if not gen.get("conclusion") and not content.get("conclusion_notes"):
        warnings.append("No conclusion content or conclusion notes found.")
    else:
        successes.append("Conclusion content available.")

    if not content.get("problem_statement"):
        warnings.append("Problem statement is empty.")
    if not content.get("objectives"):
        warnings.append("Objectives are empty.")

    if not warnings:
        st.success("No issues found -- your paper looks complete!")
    for w in warnings:
        st.warning(w)
    for s in successes:
        st.success(s)


# ============================================================================
# GENERATE & EXPORT TAB
# ============================================================================
def render_generate_export():
    st.subheader("Generate Paper & Export")
    pi = st.session_state.paper_info
    st.markdown(f"**Template:** {pi.get('citation_style', 'Generic')}  ·  **Title:** {pi.get('title') or '_untitled_'}")

    with st.expander("Preview main.tex"):
        try:
            st.code(generate_main_tex(), language="latex")
        except Exception as e:
            st.error(f"Could not render LaTeX preview: {e}")

    with st.expander("Preview references.bib"):
        st.code(generate_references_bib(), language="latex")

    if st.button("📦 Build ZIP Package", type="primary"):
        try:
            st.session_state["_zip_bytes"] = build_zip_package()
            st.session_state["zip_generated"] = True
            st.success("Package built successfully!")
        except Exception as e:
            st.error(f"Failed to build package: {e}")

    if st.session_state.get("_zip_bytes"):
        st.download_button(
            "⬇️ Download Overleaf-Ready ZIP",
            data=st.session_state["_zip_bytes"],
            file_name="research_paper_package.zip",
            mime="application/zip",
        )


# ============================================================================
# LIGHT, THEME-NEUTRAL CSS -- works in both light and dark mode
# ============================================================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 14px; }
        div[data-testid="stExpander"] {
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.25);
        }
        .stButton button { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    init_session_state()
    inject_custom_css()
    render_sidebar()

    st.title("📄 ResearchPaper Builder")
    st.caption("Convert research materials into a journal-formatted LaTeX paper and Overleaf-ready ZIP package.")

    tabs = st.tabs([
        "1️⃣ Paper Info", "2️⃣ Authors", "3️⃣ Research Content", "4️⃣ References",
        "5️⃣ Figures", "6️⃣ Tables", "7️⃣ Case Studies", "8️⃣ Experiment",
        "🤖 AI Generation", "🔍 AI Review", "📦 Generate & Export",
    ])

    with tabs[0]:
        render_paper_info()
    with tabs[1]:
        render_authors()
    with tabs[2]:
        render_research_content()
    with tabs[3]:
        render_references()
    with tabs[4]:
        render_figures()
    with tabs[5]:
        render_tables()
    with tabs[6]:
        render_case_studies()
    with tabs[7]:
        render_experiment()
    with tabs[8]:
        render_ai_generation()
    with tabs[9]:
        render_ai_review()
    with tabs[10]:
        render_generate_export()


if __name__ == "__main__":
    main()