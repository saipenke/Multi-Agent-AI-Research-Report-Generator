"""Streamlit dashboard for the multi-agent PDF research generator (Gemini main)."""
import os
import re
import traceback
from datetime import datetime

import streamlit as st

from config import (
    RATE_HOOK, REPORT_PROFILES, TOKEN_STATS, TokenBudgetExceeded,
    assign_models, get_models, get_llm, has_key,
    reset_token_stats, set_token_budget, set_token_stats,
)
from agents import build_crew, clean_topic, finalize_report, run_presearch
from pdf_generator import generate_pdf
st.set_page_config(page_title="AI Research Report Generator", page_icon="📄", layout="wide")
import hmac

APP_PASSWORD = os.getenv("Kiran@123", "")
if APP_PASSWORD and not st.session_state.get("authed"):
    st.title("🔒 Login")
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        if hmac.compare_digest(pw, APP_PASSWORD):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()
    
IS_CLOUD = (
    os.getenv("APP_ENV", "").lower() == "cloud"
    or bool(os.getenv("RENDER"))
    or bool(os.getenv("RAILWAY_ENVIRONMENT"))
)

st.set_page_config(page_title="AI Research Report Generator", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .agent-card {border:1px solid #334155; border-radius:10px; padding:12px 14px;
                 background: rgba(148,163,184,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Multi-Agent PDF Research Generator")
st.caption("Researcher → Writer → Editor → PDF  |  Free stack (CrewAI + Gemini/Groq/Ollama + DuckDuckGo)")

PROVIDER_LABELS = {
    "gemini": "Gemini (free cloud, main)",
    "groq": "Groq (free cloud, backup)",
    "ollama": "Ollama (local)",
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    providers = ["gemini", "groq"] + ([] if IS_CLOUD else ["ollama"])
    provider = st.radio("LLM Provider", providers, format_func=lambda x: PROVIDER_LABELS[x])
    if provider in ("gemini", "groq") and not has_key(provider):
        st.warning(f"No {provider.upper()}_API_KEY found in your .env file.")

    model = st.selectbox("Model", get_models(provider))

    use_strong = False
    if provider == "gemini":
        use_strong = st.checkbox(
            "Use the stronger Flash model for the Writer only",
            value=True,
            help="Researcher and Editor use the model above (Flash-Lite). Only the Writer uses full "
                 "Flash, so you spend about 1 request per report on it.",
        )

    auto_fallback = st.checkbox("If it fails, switch to a backup provider automatically", value=True)
    temperature = st.slider(
        "Creativity (Groq and Ollama only)", 0.0, 1.0, 0.3, 0.1,
        help="Gemini uses its own default setting.",
    )

    st.divider()
    st.subheader("🎛️ Report size")
    profile_name = st.radio("Size", list(REPORT_PROFILES), index=1)
    profile = REPORT_PROFILES[profile_name]
    st.caption(profile["note"])
    use_editor = st.checkbox("Include Editor stage", value=True)
    budget = st.number_input(
        "Hard token limit (stops the run)", min_value=5000, max_value=300000,
        value=profile["budget"], step=5000,
    )

    st.divider()
    st.caption(
        "Report sections: 1 Introduction, 2 Background, 3 Key Concepts, "
        "4 Important Insights, 5 Applications, 6 Conclusion, 7 References"
    )

# ---------------- Input ----------------
topic = st.text_input("Research topic", placeholder="e.g. Agentic AI: Applications, Trends and Challenges")
run = st.button("🚀 Generate Report", type="primary", disabled=not topic.strip())

AGENT_LABELS = {
    "researcher": "🔎 Lead Research Specialist",
    "writer": "✍️ Senior Technical Writer",
    "editor": "🧐 Executive Chief Editor",
}


def describe_step(step) -> str:
    tool = getattr(step, "tool", None)
    if tool:
        tool_input = str(getattr(step, "tool_input", ""))[:110]
        return f"Using **{tool}** → `{tool_input}`"
    if hasattr(step, "output"):
        return "Finishing this stage"
    return "Thinking..."


if run:
    st.session_state.pop("report_md", None)
    st.session_state.pop("pdf_bytes", None)
    reset_token_stats()
    set_token_budget(budget)

    clean = clean_topic(topic)

    cols = st.columns(3)
    status_boxes = {}
    for col, key in zip(cols, AGENT_LABELS):
        with col:
            status_boxes[key] = st.empty()

    def set_status(key: str, state: str):
        icon = {
            "waiting": "⏳ Waiting",
            "working": "🔄 Working...",
            "done": "✅ Done",
            "skipped": "⏭️ Skipped",
        }[state]
        status_boxes[key].markdown(
            f'<div class="agent-card"><b>{AGENT_LABELS[key]}</b><br>{icon}</div>',
            unsafe_allow_html=True,
        )

    def reset_cards():
        for k in AGENT_LABELS:
            set_status(k, "waiting")
        if not use_editor:
            set_status("editor", "skipped")

    reset_cards()

    st.subheader("📡 Live Activity Log")
    token_box = st.empty()
    log_box = st.empty()
    log_lines = []

    def log(msg: str):
        log_lines.append(f"`{datetime.now():%H:%M:%S}`  {msg}")
        log_box.markdown("\n\n".join(log_lines[-30:]))

    active = {"provider": provider}
    ctx = {"results": "", "sources": [], "draft": ""}

    def show_tokens():
        total = TOKEN_STATS["total"]
        text = (
            f"📊 **Tokens used this report:** {total:,} "
            f"(prompt {TOKEN_STATS['prompt']:,} + output {TOKEN_STATS['completion']:,}, "
            f"{TOKEN_STATS['calls']} LLM requests)  \n"
            f"🛑 Your limit: **{int(budget):,}** ({total / budget * 100:.0f}% used)"
        )
        if active["provider"] == "groq":
            text += f"  \n≈ {total / 200_000 * 100:.1f}% of one Groq model's free daily tokens (200,000)"
        elif active["provider"] == "gemini":
            text += "  \nℹ️ Gemini's free tier limits requests per day. See aistudio.google.com/rate-limit"
        token_box.markdown(text)

    RATE_HOOK["fn"] = lambda wait, n: log(
        f"⏳ Rate limit or busy server. Waiting {wait:.0f}s, then retrying (try {n})..."
    )

    def on_step(role_key: str, step):
        set_status(role_key, "working")
        log(f"**{AGENT_LABELS[role_key]}**: {describe_step(step)}")
        show_tokens()

    next_agent = {"research": "writer"}
    if use_editor:
        next_agent["writer"] = "editor"
    done_agent = {"research": "researcher", "writer": "writer", "editor": "editor"}

    def on_task_done(name: str, text: str = ""):
        if name == "writer":
            ctx["draft"] = text
        set_status(done_agent[name], "done")
        log(f"✅ **{AGENT_LABELS[done_agent[name]]}** finished.")
        if name in next_agent:
            set_status(next_agent[name], "working")
        show_tokens()

    def plan_for(p: str, m: str, strong: bool = False) -> dict:
        return assign_models(p, m, get_models(p), strong)

    def make_crew(p: str, plan: dict):
        cache, llms = {}, {}
        for role, mm in plan.items():
            if mm not in cache:
                cache[mm] = get_llm(p, mm, temperature, profile["max_tokens"])
            llms[role] = cache[mm]
        log(
            f"🧠 {p} → researcher: `{plan['researcher']}`, "
            f"writer: `{plan['writer']}`, editor: `{plan['editor']}`"
        )
        # Groq and Ollama have small limits or small memory, so give them shorter notes
        results = ctx["results"] if p == "gemini" else ctx["results"][:9000]
        rpm = {"gemini": 4, "groq": 6}.get(p)
        return build_crew(
            clean, llms, profile, use_editor=use_editor, max_rpm=rpm,
            search_results=results, on_step=on_step, on_task_done=on_task_done,
        )

    def kickoff_safe(crew):
        """Run the crew. If a later stage fails or returns junk, keep the Writer's draft."""
        try:
            res = crew.kickoff()
            usage = getattr(res, "token_usage", None)
            if usage is not None and TOKEN_STATS["total"] == 0:
                set_token_stats(
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                    getattr(usage, "total_tokens", 0) or 0,
                    getattr(usage, "successful_requests", 0) or 0,
                )
            text = getattr(res, "raw", None) or str(res)
            draft = ctx["draft"]
            if draft and (not text.strip() or len(text) < 0.5 * len(draft)):
                log("⚠️ The Editor's version looks cut off. Using the Writer's draft instead.")
                return draft
            return text
        except Exception as err:  # noqa: BLE001
            if ctx["draft"]:
                log(f"⚠️ A later stage failed ({str(err)[:120]}). Using the Writer's draft instead.")
                set_status("editor", "skipped")
                return ctx["draft"]
            raise

    # Order of attempts: chosen provider first, then backups
    attempts = []
    if provider == "gemini":
        first = plan_for("gemini", model, use_strong)
        attempts.append(("gemini", first))
        if first["writer"] != model:
            attempts.append(("gemini", plan_for("gemini", model, False)))
        backups = ["groq"]
    elif provider == "groq":
        attempts.append(("groq", plan_for("groq", model)))
        backups = ["gemini"]
    else:
        attempts.append(("ollama", plan_for("ollama", model)))
        backups = ["gemini", "groq"]
    if auto_fallback:
        for b in backups:
            if has_key(b) and get_models(b):
                attempts.append((b, plan_for(b, get_models(b)[0])))

    try:
        set_status("researcher", "working")
        log(f"🧹 Topic used for research: **{clean}**")
        log("🌐 Searching the web with DuckDuckGo...")
        ctx["results"], ctx["sources"] = run_presearch(
            clean, profile, on_query=lambda q: log(f"🔍 Search: `{q}`")
        )
        if not ctx["sources"]:
            log("⚠️ No search results (DuckDuckGo may be rate limiting). Agents will try to search themselves.")
            ctx["results"] = ""
        else:
            log(f"✅ Collected {len(ctx['sources'])} sources ({len(ctx['results']):,} characters)")

        report_md, last_err = None, None
        for i, (p, plan) in enumerate(attempts):
            try:
                active["provider"] = p
                if i > 0:
                    log(f"🔁 Trying backup: **{p}**")
                    reset_cards()
                    reset_token_stats()
                    ctx["draft"] = ""
                    set_status("researcher", "working")
                crew = make_crew(p, plan)
                log(f"🚀 Starting agents ({PROVIDER_LABELS[p]}, size: {profile_name}, "
                    f"editor: {'on' if use_editor else 'off'})")
                report_md = kickoff_safe(crew)
                break
            except TokenBudgetExceeded:
                raise
            except Exception as err:  # noqa: BLE001
                last_err = err
                log(f"⚠️ {p} failed: {str(err)[:200]}")
        if report_md is None:
            raise last_err

        show_tokens()
        log(f"🧮 Final count for this report: **{TOKEN_STATS['total']:,}** tokens, "
            f"**{TOKEN_STATS['calls']}** LLM requests")

        report_md = re.sub(r"^```(?:markdown|md)?\s*\n", "", report_md.strip())
        report_md = re.sub(r"\n```\s*$", "", report_md)
        report_md = finalize_report(report_md, ctx["sources"])

        log("📄 Rendering PDF...")
        pdf_bytes, engine = generate_pdf(report_md, clean)
        log(f"🎉 PDF ready (engine: {engine})")

        st.session_state["report_md"] = report_md
        st.session_state["pdf_bytes"] = pdf_bytes
        st.session_state["engine"] = engine
        st.session_state["topic"] = clean
        st.session_state["tokens"] = TOKEN_STATS["total"]

    except TokenBudgetExceeded as e:
        st.error(f"🛑 {e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong: {e}")
        st.code(traceback.format_exc())
        st.markdown(
            "**Common fixes:**\n"
            "- `GEMINI_API_KEY` missing or wrong → check your `.env`\n"
            "- Daily limit reached → pick another model, switch provider, or wait for the reset\n"
            "- Check your real limits at https://aistudio.google.com/rate-limit"
        )

# ---------------- Results ----------------
if "pdf_bytes" in st.session_state:
    st.divider()
    st.success(
        f"Report ready! (PDF engine: {st.session_state.get('engine')}, "
        f"tokens used: {st.session_state.get('tokens', 0):,})"
    )
    st.caption(
        "The References list comes straight from the web search results. "
        "AI writing can still contain mistakes, so check key statistics against the sources before sharing."
    )

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", st.session_state["topic"]).strip("_")[:50] or "report"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"{safe_name}.pdf",
            mime="application/pdf",
            type="primary",
        )
    with c2:
        st.download_button(
            "⬇️ Download Markdown",
            data=st.session_state["report_md"],
            file_name=f"{safe_name}.md",
            mime="text/markdown",
        )

    st.subheader("👀 Report Preview")
    st.markdown(st.session_state["report_md"])
