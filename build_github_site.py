#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
LOGO_SOURCE = ROOT / "10_Connection" / "01_Images" / "88_KG-Logo-1.png"
LOGO_DEST = ASSETS_DIR / "katana-graph-logo.png"
YEARS = ("2016", "2017")

TOPIC_OVERRIDES = {
    "10_Connection": {
        "title": "Connection and Graph Lifecycle",
        "customer": "I am getting started with KatanaGraph and need the basic operational flow. How do I connect to the service, validate pod placement, and create or drop graphs cleanly from notebooks?",
        "daniel": "This topic folder walks through the connection bootstrap, graph create and drop patterns, and a lightweight environment check for pod placement so you can validate the platform before moving on to analytics work.",
    },
    "20_ImportData": {
        "title": "Importing Data into KatanaGraph",
        "customer": "I have files and data frames ready to go, but the import path is not obvious. What is the right way to load bucket data, run mini-batch ingestion, and manage multiple node labels in KatanaGraph?",
        "daniel": "These notebooks focus on ingestion mechanics, including bucket-based loading, incremental and mini-batch patterns, and variations in node-label modeling so you can move raw source data into a working graph quickly.",
    },
    "22_FirstStepsCypher": {
        "title": "First Steps with Cypher",
        "customer": "My team knows SQL better than graph query languages. What does the KatanaGraph Cypher workflow look like for first traversals, dictionary and array handling, cleanup operations, and directional edge patterns?",
        "daniel": "This folder introduces the everyday Cypher basics: setup, traversals, collection handling, graph cleanup, and practical update patterns that help a new user become productive with the query language.",
    },
    "23_Rest(V0)": {
        "title": "REST API Basics",
        "customer": "We may need to drive KatanaGraph from external services instead of only through notebooks. What does the early REST interface look like, and what is the minimum setup for exercising it?",
        "daniel": "The notebooks here show a first-pass REST workflow, pairing environment setup with direct service calls so you can test remote control patterns before investing in a larger application wrapper.",
    },
    "A1_Path": {
        "title": "Path Analytics",
        "customer": "I need route and traversal analytics over connected data. How do I set up sample graphs and run shortest path, breadth-first search, KSSSP, and random-walk style analysis in KatanaGraph?",
        "daniel": "This topic collects the path-oriented analytics material, using airport and related sample graphs to explore traversal setup, SSSP, BFS, KSSSP, and other path-centric algorithms.",
    },
    "A2_Community": {
        "title": "Community Detection",
        "customer": "My graph has clusters that I need to expose for analysis and segmentation. How do I prepare representative data sets and run Louvain-style community detection in KatanaGraph?",
        "daniel": "The notebooks here move from setup into community analytics, including multiple graph sizes and environments, so you can see how KatanaGraph surfaces clusters and group structure in connected data.",
    },
    "A3_Centrality": {
        "title": "Centrality Analytics",
        "customer": "I want to identify the most influential nodes in several graphs. What is the KatanaGraph workflow for setting up sample data and computing betweenness and PageRank style centrality measures?",
        "daniel": "This folder focuses on centrality routines and shows how to prepare several data sets, then run ranking and influence-oriented analytics such as betweenness and PageRank.",
    },
    "AH_Houston": {
        "title": "Houston Use Case",
        "customer": "Can KatanaGraph support a location-specific analytical use case with realistic data rather than only toy examples? I want to see a concrete Houston-centered scenario with graph queries and results.",
        "daniel": "These notebooks build a Houston-oriented example that combines setup with applied Cypher work, giving you a more concrete end-to-end use case than the smaller tutorial graphs.",
    },
    "AI_LdbcAndRoutines_Justin": {
        "title": "LDBC and Routines",
        "customer": "We want to benchmark and experiment with richer graph structures. How do I load LDBC-style data and combine it with routines like PageRank, Louvain, and betweenness in KatanaGraph?",
        "daniel": "This topic folder covers loading the LDBC benchmark-style data set and pairing it with algorithmic routines, making it useful for larger-scale experiments and platform evaluation work.",
    },
    "C1_Compulsaries": {
        "title": "Core Notebook Compulsories",
        "customer": "Before I get fancy, I need the repeatable basics. What are the core notebook patterns for display options, dataframe reshaping, graph import, versioning, saving RDG artifacts, and fetching results?",
        "daniel": "This is the utility belt for the repository: display setup, dataframe transformation, import helpers, versioning notes, save and fetch patterns, and other foundational notebook techniques used throughout the project.",
    },
    "C2_UDF": {
        "title": "User-Defined Functions",
        "customer": "I need logic that goes beyond stock queries and built-in analytics. How do I set up and test KatanaGraph UDF workflows, from simple hello-world patterns to graph mutation and MPI-oriented experiments?",
        "daniel": "The UDF notebooks cover setup, implementation patterns, iterative experiments, and load tests so you can extend KatanaGraph behavior with custom logic rather than staying limited to built-in operations.",
    },
    "C3_RestGraphQL": {
        "title": "REST and GraphQL Services",
        "customer": "We are considering an application layer in front of KatanaGraph. What does a service-oriented pattern look like when combining setup, a simple client, a web server, and GraphQL-style schema wiring?",
        "daniel": "This folder packages KatanaGraph access into early service examples, showing how REST and GraphQL ideas fit together with lightweight Python components and notebook-driven setup.",
    },
    "J1_Jiras": {
        "title": "Jira Reproductions and Fixes",
        "customer": "Our team needs to reproduce product issues quickly and keep a record of what was tested. How can we use notebooks to isolate KatanaGraph bugs around counts, Louvain, dataframe import, Cypher behavior, and result handling?",
        "daniel": "The Jira section is a structured archive of issue reproductions, setup notebooks, and validation tests. It is especially useful when you need concrete examples for troubleshooting or regression checking.",
    },
    "P1_Prospects": {
        "title": "Prospects and Solution Experiments",
        "customer": "We need prospect-facing demos that connect KatanaGraph to real business problems such as oncology NLP pipelines, knowledge graphs, and large banking workloads. What does that exploratory material look like?",
        "daniel": "These notebooks collect prospect-oriented solution work, pairing external data pipelines and industry use cases with graph loading and analytics so the platform can be evaluated against customer scenarios.",
    },
    "S1_N_Labs": {
        "title": "N Labs and Graph Loading",
        "customer": "I want a compact lab that demonstrates how a smaller graph is assembled and loaded. Is there a minimal KatanaGraph example that focuses on source files, graph structure, and the loading sequence?",
        "daniel": "This small lab folder provides a stripped-down graph-loading example, making it useful when you want a contained setup for quick experimentation rather than a larger analytics scenario.",
    },
    "S2_Bk_PracDpLrng": {
        "title": "Practical Deep Learning Experiments",
        "customer": "Not every graph project is only about traversal. How were notebooks used here to explore model-oriented work such as classic ML, neural networks, charting, and MNIST-style experiments around the KatanaGraph effort?",
        "daniel": "This section widens the scope beyond pure graph operations and captures a practical deep-learning track with visualizations, classic models, neural-network samples, and supporting data assets.",
    },
    "Z2_Retired__": {
        "title": "Retired Experiments and Legacy Work",
        "customer": "We have a large amount of older exploratory material that still has historical value. Can we keep legacy KatanaGraph notebooks available without mixing them into the main current-path topics?",
        "daniel": "This archive folder preserves retired and superseded work, including prior demos, prototypes, and side experiments, so the historical record remains accessible without implying that the content is current or maintained.",
    },
}


@dataclass(frozen=True)
class TopicEntry:
    folder: str
    year: str
    month: int
    month_name: str
    title: str
    customer: str
    daniel: str
    notebooks: list[Path]
    image_assets: list[Path]

    @property
    def month_label(self) -> str:
        return f"{self.month_name} {self.year}"

    @property
    def article_title(self) -> str:
        return f"{self.month_name} {self.year}: {self.title}"


def slug_to_title(name: str) -> str:
    cleaned = re.sub(r"^[A-Z0-9]+_", "", name)
    cleaned = cleaned.replace("_", " ").replace("(V0)", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or name


def notebook_title(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^[0-9A-Z]{1,3}[A-Z]?_", "", stem)
    stem = stem.replace("__", " ")
    stem = stem.replace("_", " ")
    stem = stem.replace(".", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -")
    return stem or path.name


def notebook_heading(path: Path) -> str:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return ""

    generic_prefixes = (
        "setup stuff",
        "setup:",
        "setup ..",
        "<div>",
        "must run file",
    )
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        for line in text.splitlines():
            cleaned = re.sub(r"^[#>\-\s]+", "", line).strip()
            cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()
            lower = cleaned.lower()
            if not cleaned:
                continue
            if any(lower.startswith(prefix) for prefix in generic_prefixes):
                continue
            if lower in {"div", "/div"}:
                continue
            if len(cleaned) < 4:
                continue
            return cleaned
    return ""


def list_topic_folders() -> list[str]:
    return sorted(
        [
            path.name
            for path in ROOT.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and not path.name.startswith("__")
            and path.name not in {"assets", "2016", "2017"}
        ]
    )


def monthly_slots() -> list[tuple[str, int, str]]:
    slots: list[tuple[str, int, str]] = []
    current = date(2016, 2, 1)
    end = date(2017, 5, 1)
    while current <= end:
        slots.append((str(current.year), current.month, current.strftime("%B")))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return slots


def assign_topics_to_slots(folders: list[str]) -> list[TopicEntry]:
    slots = monthly_slots()
    total_topics = len(folders)
    total_slots = len(slots)
    entries: list[TopicEntry] = []

    for index, folder in enumerate(folders):
        if total_topics <= 1:
            slot_index = 0
        else:
            slot_index = round(index * (total_slots - 1) / (total_topics - 1))
        year, month, month_name = slots[slot_index]
        override = TOPIC_OVERRIDES.get(folder, {})
        topic_path = ROOT / folder
        notebooks = sorted(topic_path.rglob("*.ipynb"))
        image_assets = sorted(
            [
                image
                for image in topic_path.rglob("*")
                if image.is_file() and image.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg"}
            ]
        )
        entries.append(
            TopicEntry(
                folder=folder,
                year=year,
                month=month,
                month_name=month_name,
                title=override.get("title", slug_to_title(folder)),
                customer=override.get(
                    "customer",
                    f"I need a practical KatanaGraph example in the {slug_to_title(folder)} area. What does this topic folder cover and how should I use it?",
                ),
                daniel=override.get(
                    "daniel",
                    "This archive section groups related KatanaGraph notebooks and supporting assets into a single topic path so you can work through the material in a coherent order.",
                ),
                notebooks=notebooks,
                image_assets=image_assets,
            )
        )

    return sorted(entries, key=lambda entry: (entry.year, entry.month, entry.folder))


def logo_block(relative_logo_path: str) -> str:
    return (
        '<p align="center">\n'
        f'  <img src="{relative_logo_path}" alt="KatanaGraph logo" width="320">\n'
        "</p>\n"
    )


def markdown_path(path: str) -> str:
    return quote(path, safe="/-_.")


def tabs_block(active: str) -> str:
    root_cell = "<strong>Archive Home</strong>" if active == "home" else '<a href="../README.md"><strong>Archive Home</strong></a>'
    year_cells = []
    for year in YEARS:
        if year == active:
            year_cells.append(f"<td><strong>{year}</strong></td>")
        else:
            year_cells.append(f'<td><a href="../{year}/README.md"><strong>{year}</strong></a></td>')
    return (
        "<table>\n"
        "  <tr>\n"
        f"    <td>{root_cell}</td>\n"
        f"    {''.join(year_cells)}\n"
        "  </tr>\n"
        "</table>\n"
    )


def root_tabs_block() -> str:
    year_cells = "".join([f'<td><a href="{year}/README.md"><strong>{year}</strong></a></td>' for year in YEARS])
    return (
        "<table>\n"
        "  <tr>\n"
        "    <td><strong>Archive Home</strong></td>\n"
        f"    {year_cells}\n"
        "  </tr>\n"
        "</table>\n"
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def topic_page(entry: TopicEntry) -> str:
    notebook_lines = []
    for notebook in entry.notebooks:
        rel = notebook.relative_to(ROOT / entry.folder).as_posix()
        heading = notebook_heading(notebook)
        line = f"- [{notebook_title(notebook)}]({markdown_path(rel)})"
        if heading:
            line += f": {heading}"
        notebook_lines.append(line)
    notebook_section = "\n".join(notebook_lines) if notebook_lines else "- No notebooks found."

    asset_lines = []
    for image in entry.image_assets[:8]:
        rel = image.relative_to(ROOT / entry.folder).as_posix()
        asset_lines.append(f"- [{image.name}]({markdown_path(rel)})")
    if len(entry.image_assets) > 8:
        asset_lines.append(f"- Additional image assets in this folder: {len(entry.image_assets) - 8}")
    asset_section = "\n".join(asset_lines) if asset_lines else "- No image assets indexed for this topic."

    return f"""
{logo_block("../assets/katana-graph-logo.png")}
# {entry.article_title}

{tabs_block(entry.year)}

KatanaGraph archival topic folder for the developer notebooks.

**Customer:** {entry.customer}

**Daniel:** {entry.daniel}

## Topic Folder

- Folder: [`{entry.folder}`](./)
- Archive placement: `{entry.month_label}`
- Notebook count: `{len(entry.notebooks)}`
- Image asset count: `{len(entry.image_assets)}`

## Notebook Inventory

{notebook_section}

## Supporting Images

{asset_section}
""".strip()


def entry_line(entry: TopicEntry, prefix: str) -> str:
    return "\n".join(
        [
            f"### [{entry.article_title}]({markdown_path(prefix + entry.folder + '/README.md')})",
            "",
            f"**Customer:** {entry.customer}",
            "",
            f"**Daniel:** {entry.daniel}",
        ]
    )


def year_page(year: str, entries: list[TopicEntry]) -> str:
    body = "\n\n".join(entry_line(entry, "../") for entry in entries if entry.year == year)
    return f"""
{logo_block("../assets/katana-graph-logo.png")}
# KatanaGraph Developers Notebook Archive {year}

{tabs_block(year)}

This year view preserves topic folders from the KatanaGraph notebook archive and maps them onto a monthly article cadence, matching the archive style used for the MongoDB developers notebook site.

{body}
""".strip()


def root_page(entries: list[TopicEntry]) -> str:
    years_section = []
    for year in YEARS:
        year_entries = [entry for entry in entries if entry.year == year]
        lines = [f"## [{year}]({markdown_path(year + '/README.md')})"]
        for entry in year_entries:
            lines.append(f"- [{entry.article_title}]({markdown_path(entry.folder + '/README.md')})")
        years_section.append("\n".join(lines))
    return f"""
{logo_block("assets/katana-graph-logo.png")}
# KatanaGraph Developers Notebook Archive

{root_tabs_block()}

KatanaGraph no longer exists as an independent company, but this repository preserves the notebook work as an organized developer archive. The topic folders below have been recast into a monthly article-style sequence modeled after the MongoDB and DataStax notebook sites.

{"\n\n".join(years_section)}
""".strip()


def ensure_logo() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    shutil.copy2(LOGO_SOURCE, LOGO_DEST)


def main() -> None:
    ensure_logo()
    entries = assign_topics_to_slots(list_topic_folders())

    for year in YEARS:
        write_text(ROOT / year / "README.md", year_page(year, entries))

    for entry in entries:
        write_text(ROOT / entry.folder / "README.md", topic_page(entry))

    write_text(ROOT / "README.md", root_page(entries))


if __name__ == "__main__":
    main()
