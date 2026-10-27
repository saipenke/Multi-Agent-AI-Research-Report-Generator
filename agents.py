"""Agents, tasks, topic cleaner, web pre-search and report finishing."""
import re
import time
from typing import Callable, Optional

from crewai import Agent, Crew, Process, Task

from tools import DuckDuckGoSearchTool

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS

SECTIONS = [
    "Introduction",
    "Background",
    "Key Concepts",
    "Important Insights",
    "Applications",
    "Conclusion",
]


def clean_topic(raw: str) -> str:
    """Turn 'generate a pdf on the topic of the agentic ai' into 'agentic ai'."""
    t = raw.strip()
    t = re.sub(
        r"^(please\s+)?(generate|create|make|write|give me|get me)\s+(an?\s+)?"
        r"(pdf|report|research report|research paper)\s*"
        r"(on|about|for|regarding)?\s*(the\s+)?(topic\s+)?(of\s+)?(the\s+)?",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    return t or raw.strip()


def _ddg(query: str, n: int) -> list:
    for attempt in range(3):
        try:
            with DDGS() as d:
                return list(d.text(query, max_results=n))
        except Exception:  # rate limits etc.
            time.sleep(2 * (attempt + 1))
    return []


def run_presearch(topic: str, profile: dict, on_query: Optional[Callable[[str], None]] = None):
    """Run the web searches in Python. Returns (text_for_the_agents, sources).

    Sources get tags like [S1], [S2]. URLs stay out of the text the model sees,
    so it cannot invent them. The real URLs are added back at the end.
    """
    templates = [
        "{t} overview definition",
        "{t} history background origins",
        "{t} key concepts how it works",
        "{t} statistics market data",
        "{t} applications real-world use cases",
        "{t} challenges risks future outlook",
    ]
    sources, seen, blocks = [], {}, []
    for tpl in templates[: profile["searches"]]:
        query = tpl.format(t=topic)
        if on_query:
            on_query(query)
        lines = []
        for r in _ddg(query, profile["results"]):
            url = (r.get("href") or r.get("url") or "").strip()
            if not url:
                continue
            key = url.rstrip("/")
            if key in seen:
                continue
            sid = len(sources) + 1
            seen[key] = sid
            title = (r.get("title") or "Untitled").strip()
            sources.append({"id": sid, "title": title, "url": url})
            body = (r.get("body") or "")[: profile["snippet"]].strip()
            lines.append(f"[S{sid}] {title}\nSummary: {body}")
        blocks.append(f"### Search: {query}\n" + ("\n\n".join(lines) or "No new results."))
        time.sleep(1)
    return "\n\n".join(blocks), sources


def normalize_headings(md: str) -> str:
    """Make sure the six main sections are numbered exactly 1-6."""
    for n, name in enumerate(SECTIONS, 1):
        pattern = r"(?im)^##\s*(?:\d+[.)]?\s*)?" + re.escape(name) + r"\s*$"
        md = re.sub(pattern, f"## {n}. {name}", md)
    return md


def finalize_report(md: str, sources: list) -> str:
    """Number the sections, replace [S#] tags with [1], [2]... and add '7. References'."""
    md = re.sub(r"【[^】]*】", "", md)  # odd citation markers some models produce
    md = normalize_headings(md)

    m = re.search(r"\n#{1,3}\s*(?:\d+[.)]?\s*)?References\b.*$", md, flags=re.S | re.I)
    body = md[: m.start()] if m else md
    heading = f"## {len(SECTIONS) + 1}. References"

    if not sources:
        return body.rstrip() + f"\n\n{heading}\nNo web sources were available for this report.\n"

    # [S1, S2] -> [S1][S2]
    body = re.sub(
        r"\[(S\d+(?:\s*,\s*S?\d+)+)\]",
        lambda mm: "".join(f"[S{x}]" for x in re.findall(r"\d+", mm.group(1))),
        body,
    )

    by_id = {s["id"]: s for s in sources}
    order = []
    for x in re.findall(r"\[S(\d+)\]", body):
        i = int(x)
        if i in by_id and i not in order:
            order.append(i)
    newnum = {sid: n for n, sid in enumerate(order, 1)}

    def repl(match):
        i = int(match.group(1))
        return f"[{newnum[i]}]" if i in newnum else ""

    body = re.sub(r"\[S(\d+)\]", repl, body)

    if order:
        refs = "\n".join(f"{newnum[s]}. {by_id[s]['title']}: {by_id[s]['url']}" for s in order)
    else:
        refs = "\n".join(
            f"{k}. {s['title']}: {s['url']}" for k, s in enumerate(sources[:15], 1)
        )
    return body.rstrip() + f"\n\n{heading}\n{refs}\n"


def build_crew(
    topic: str,
    llms: dict,
    profile: dict,
    use_editor: bool = True,
    max_rpm: Optional[int] = 4,
    search_results: str = "",
    on_step: Optional[Callable[[str, object], None]] = None,
    on_task_done: Optional[Callable[[str, str], None]] = None,
) -> Crew:
    """llms = {"researcher": LLM, "writer": LLM, "editor": LLM}"""
    search_tool = DuckDuckGoSearchTool(
        max_results=profile["results"], snippet_chars=profile["snippet"]
    )

    def step_cb(role_key: str):
        def _cb(step):
            if on_step:
                on_step(role_key, step)
        return _cb

    def task_cb(name: str):
        def _cb(output):
            if on_task_done:
                on_task_done(name, getattr(output, "raw", None) or str(output))
        return _cb

    # ---------------- AGENTS ----------------
    researcher = Agent(
        role="Lead Research Specialist",
        goal=f"Turn web search results about '{topic}' into verified research notes.",
        backstory=(
            "You are a meticulous research analyst. You never invent facts. "
            "You keep the [S#] source tag next to every fact. You keep notes short."
        ),
        tools=[] if search_results else [search_tool],
        llm=llms["researcher"],
        allow_delegation=False,
        verbose=True,
        max_iter=profile["researcher_iter"],
        step_callback=step_cb("researcher"),
    )

    writer = Agent(
        role="Senior Technical Writer",
        goal="Turn research notes into a polished, well-structured Markdown report.",
        backstory=(
            "You are a senior writer at a top consulting firm. You use only facts that "
            "appear in the research notes and cite them with their [S#] tags."
        ),
        llm=llms["writer"],
        allow_delegation=False,
        verbose=True,
        max_iter=3,
        step_callback=step_cb("writer"),
    )

    editor = Agent(
        role="Executive Chief Editor",
        goal="Audit and perfect the report for accuracy, flow, tone and Markdown formatting.",
        backstory=(
            "You are a chief editor. You remove unsupported claims, fix structure and "
            "tighten language. You never add new facts."
        ),
        llm=llms["editor"],
        allow_delegation=False,
        verbose=True,
        max_iter=3,
        step_callback=step_cb("editor"),
    )

    # ---------------- TASKS ----------------
    if search_results:
        research_description = (
            f"Research the topic: '{topic}'.\n"
            "The web searches have ALREADY been run. Their results are below, each with a "
            "source tag like [S1]. Use ONLY these results.\n"
            "Write structured notes in these groups: Introduction facts, Background and history, "
            "Key concepts and terms, Statistics and insights, Applications and examples, "
            "Challenges and outlook. Use short bullet points, one fact per line, and keep the "
            "[S#] tag next to every fact, number and example. Do NOT add anything that is not in "
            "the results. If something is missing, write 'not found in sources'.\n\n"
            "=== SEARCH RESULTS ===\n" + search_results
        )
    else:
        research_description = (
            f"Research the topic: '{topic}'.\n"
            f"Run exactly {profile['searches']} different web searches covering: overview, "
            "history and background, key concepts, statistics, applications, challenges and outlook.\n"
            "Write short structured notes (bullet points, one line per fact) with the source URL "
            "next to every fact. Do NOT make up numbers."
        )

    research_task = Task(
        description=research_description,
        expected_output=(
            "Structured research notes grouped by: Introduction, Background, Key Concepts, "
            "Insights and Statistics, Applications, Challenges and Outlook. "
            "Each fact keeps its [S#] tag."
        ),
        agent=researcher,
        callback=task_cb("research"),
    )

    write_task = Task(
        description=(
            f"Using ONLY the research notes, write a professional research report on '{topic}' in Markdown.\n"
            "Use exactly this structure and these headings:\n"
            "# <Report Title>\n"
            "## 1. Introduction   (what the topic is, why it matters, what the report covers)\n"
            "## 2. Background   (origins, history, how it developed, where it stands today)\n"
            "## 3. Key Concepts   (explain the core ideas and terms, use ### subsections)\n"
            "## 4. Important Insights   (bullet list of the main findings, include at least one "
            "Markdown table of facts or statistics from the notes, and ONE callout line starting "
            "with '> **Key Takeaway:**')\n"
            "## 5. Applications   (real-world uses and examples grouped by area, use ### subsections)\n"
            "## 6. Conclusion   (summary, main challenges, future outlook)\n"
            "STRICT RULES:\n"
            "- Every statistic, number, company, product or date must come from the notes.\n"
            "- Cite sources only as the [S#] tags from the notes, for example [S3]. No other citation format.\n"
            "- If the notes do not contain a fact, do not state it.\n"
            "- Do NOT write a References section. It is added automatically as section 7.\n"
            f"- Professional tone, no filler, {profile['words']} words.\n"
            "Return ONLY the Markdown report. No commentary and no code fences around it."
        ),
        expected_output="A complete Markdown report with sections 1 to 6 exactly as specified.",
        agent=writer,
        context=[research_task],
        callback=task_cb("writer"),
    )

    agents = [researcher, writer]
    tasks = [research_task, write_task]

    if use_editor:
        edit_task = Task(
            description=(
                "Audit the Markdown report. Improve flow and tone, fix formatting, and make sure "
                "headings, tables, bullet lists and the '> **Key Takeaway:**' callout are valid Markdown.\n"
                "RULES: keep the exact headings '## 1. Introduction' to '## 6. Conclusion'. Keep every "
                "[S#] tag exactly as written. Do not add new facts, numbers, names or sources. "
                "Remove or soften any claim that no [S#] tag supports. Do not add a References section.\n"
                "Return ONLY the full final Markdown report. No commentary and no code fences."
            ),
            expected_output="The full, publication-ready Markdown report only.",
            agent=editor,
            context=[write_task],
            callback=task_cb("editor"),
        )
        agents.append(editor)
        tasks.append(edit_task)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        max_rpm=max_rpm,
    )