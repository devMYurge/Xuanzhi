"""Reusable Streamlit render helpers.

Keeping the rendering vocabulary in one module means the views read like
a storyboard and the visual style stays consistent.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from xuanzhi.db import Store
from xuanzhi.schema import Paper

# Colour per edge family for the graph view.
EDGE_COLOURS = {
    "shared_concept": "#4C9AFF",
    "shared_author": "#57D9A3",
    "shared_area": "#FFAB00",
    "citation": "#FF5630",
    "similar_embedding": "#9F7AEA",
    "mixed": "#8993A4",
}

# Colour per source for graph nodes.
SOURCE_COLOURS = {
    "arxiv": "#B3261E",
    "semantic_scholar": "#1A73E8",
    "openalex": "#188038",
    "google_scholar": "#E37400",
    "pubmed": "#7B1FA2",
    "biorxiv": "#00838F",
    "manual": "#5F6368",
}


def paper_card(paper: Paper, *, expanded: bool = False) -> None:
    """Render a compact paper summary block."""
    meta_bits = []
    if paper.year:
        meta_bits.append(str(paper.year))
    meta_bits.append(paper.source.value)
    if paper.citation_count is not None:
        meta_bits.append(f"{paper.citation_count} citations")
    st.markdown(f"**{paper.title}**")
    st.caption(" · ".join(meta_bits))
    if paper.authors:
        names = ", ".join(a.name for a in paper.authors[:6])
        if len(paper.authors) > 6:
            names += f" (+{len(paper.authors) - 6} more)"
        st.write(f":grey[{names}]")
    if paper.research_areas:
        st.write(" ".join(f"`{a.name}`" for a in paper.research_areas))
    if paper.abstract:
        if expanded:
            st.write(paper.abstract)
        else:
            st.caption(
                paper.abstract[:280] + ("…" if len(paper.abstract) > 280 else "")
            )
    if paper.url:
        st.markdown(f"[Open source page]({paper.url})")


def summaries_block(store: Store, paper: Paper) -> None:
    """Show every stored summary for a paper, side by side per model."""
    summaries = store.get_summaries_for_paper(paper.id)
    if not summaries:
        st.info(
            "No summaries yet. Run `python scripts/run_summarise.py` "
            "(add `--claude` to compare against a frontier model)."
        )
        return
    cols = st.columns(len(summaries))
    for col, summary in zip(cols, summaries):
        with col:
            st.markdown(f"**{summary.model}**")
            st.write(summary.summary_text)


def figure_grid(store: Store, paper: Paper, *, columns: int = 3) -> None:
    """Render a paper's extracted figures with type + caption."""
    figures = list(store.iter_figures(paper_id=paper.id))
    if not figures:
        st.info(
            "No figures extracted yet. Run "
            "`python scripts/run_extract_figures.py`."
        )
        return
    rows = [figures[i : i + columns] for i in range(0, len(figures), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, fig in zip(cols, row):
            with col:
                img_path = Path(fig.image_path)
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption("_(image file missing)_")
                st.caption(
                    f"`{fig.figure_type.value}`"
                    + (f" · p.{fig.page_num}" if fig.page_num else "")
                )
                if fig.caption:
                    st.caption(fig.caption[:160])


def render_graph(
    graph,
    *,
    height: int = 600,
    paper_extras: dict | None = None,
) -> str | None:
    """Interactive graph: zoom, pan, drag nodes, click for a floating card.

    Clicking a node opens an in-graph info card (title, authors, abstract,
    link).  A ▶ Animate / ⏸ Freeze button lets the user toggle the physics
    simulation on demand.  Returns the paper id picked in the selectbox
    below the graph (to open the full Streamlit detail panel), or ``None``.
    """
    import json
    import networkx as nx
    from pyvis.network import Network

    if graph.number_of_nodes() == 0:
        st.info("No nodes to display.")
        return None

    extras = paper_extras or {}

    # Build JSON payload for the in-graph click card.
    node_data: dict[str, dict] = {}
    for nid, data in graph.nodes(data=True):
        title = data.get("title", nid) or nid
        ex = extras.get(nid, {})
        node_data[nid] = {
            "title":    title,
            "authors":  ex.get("authors", ""),
            "abstract": ex.get("abstract", ""),
            "year":     str(data.get("year") or ""),
            "source":   data.get("source", ""),
            "citations": data.get("citation_count") or 0,
            "areas":    data.get("areas") or [],
            "url":      data.get("url") or "",
        }

    # Pre-compute spring layout so nodes are stable from the start.
    scale = 800
    pos = nx.spring_layout(
        graph, seed=42,
        k=2.5 / max(graph.number_of_nodes() ** 0.5, 1),
    )

    net = Network(height=f"{height}px", width="100%",
                  bgcolor="#0e1117", font_color="#ffffff")
    net.toggle_physics(False)

    for nid, data in graph.nodes(data=True):
        source = data.get("source", "manual")
        degree = graph.degree(nid)
        title  = node_data[nid]["title"]
        x, y   = pos[nid]
        net.add_node(
            nid,
            label=_short(title, 28),
            title=title,                     # vis.js hover tooltip
            size=10 + min(degree * 2, 22),
            color=SOURCE_COLOURS.get(source, "#5F6368"),
            x=float(x * scale),
            y=float(y * scale),
        )

    for a, b, data in graph.edges(data=True):
        etype = data.get("edge_type", "shared_concept")
        net.add_edge(
            a, b,
            color=EDGE_COLOURS.get(etype, "#8993A4"),
            title=data.get("evidence", etype),
            width=1,
        )

    # Inject floating info card + physics toggle into the pyvis HTML.
    payload = json.dumps(node_data).replace("</script>", r"<\/script>")

    injection = f"""
<div id="_xz_panel" style="
    display:none; position:fixed; top:14px; right:14px; width:300px;
    max-height:420px; overflow-y:auto;
    background:#1e1e2e; color:#cdd6f4;
    border:1px solid #45475a; border-radius:10px;
    padding:14px 16px; font-family:system-ui,sans-serif;
    font-size:12px; line-height:1.55; z-index:9999;
    box-shadow:0 8px 32px rgba(0,0,0,.7);">
</div>

<button id="_xz_phys" onclick="_xzToggle()" style="
    position:fixed; bottom:14px; left:14px;
    background:#313244; color:#cdd6f4; border:none;
    padding:5px 12px; border-radius:6px;
    font-size:12px; cursor:pointer; z-index:9999;">
  ▶ Animate
</button>

<script>
var _xzData = {payload};
var _xzOn   = false;

function _xzToggle() {{
    _xzOn = !_xzOn;
    network.setOptions({{physics:{{enabled:_xzOn}}}});
    document.getElementById('_xz_phys').textContent = _xzOn ? '⏸ Freeze' : '▶ Animate';
}}

(function _xzWait() {{
    if (typeof network === 'undefined') {{ setTimeout(_xzWait, 60); return; }}

    network.on('click', function(p) {{
        var panel = document.getElementById('_xz_panel');
        if (!p.nodes.length) {{ panel.style.display='none'; return; }}

        var d = _xzData[p.nodes[0]];
        if (!d) return;

        var h = '<div style="display:flex;justify-content:space-between;align-items:flex-start">';
        h += '<b style="color:#cba6f7;font-size:13px;flex:1;padding-right:6px">' + d.title + '</b>';
        h += '<span onclick="document.getElementById(\'_xz_panel\').style.display=\'none\'" ';
        h += 'style="cursor:pointer;color:#6c7086;font-size:16px;flex-shrink:0">✕</span></div>';

        if (d.authors) h += '<div style="color:#a6e3a1;margin-top:6px">' + d.authors + '</div>';

        var meta = [];
        if (d.year) meta.push(d.year);
        if (d.source) meta.push(d.source);
        if (d.citations) meta.push(d.citations + ' cit.');
        if (meta.length) h += '<div style="color:#6c7086;margin-top:4px">' + meta.join(' · ') + '</div>';

        if (d.areas && d.areas.length) {{
            h += '<div style="margin-top:7px">';
            d.areas.forEach(function(a){{
                h += '<span style="background:#313244;padding:2px 7px;border-radius:4px;'
                  +  'margin-right:4px;display:inline-block;margin-bottom:3px">' + a + '</span>';
            }});
            h += '</div>';
        }}

        if (d.abstract)
            h += '<div style="border-top:1px solid #313244;margin-top:9px;padding-top:8px">'
              +  d.abstract + '…</div>';

        if (d.url)
            h += '<div style="margin-top:9px">'
              +  '<a href="' + d.url + '" target="_blank" '
              +  'style="color:#89b4fa;text-decoration:none">Open source ↗</a></div>';

        panel.innerHTML = h;
        panel.style.display = 'block';
    }});
}})();
</script>
"""

    html = net.generate_html()
    html = html.replace("</body>", injection + "\n</body>")
    st.components.v1.html(html, height=height, scrolling=False)

    # Selectbox to open full Streamlit detail panel below the graph.
    options: dict[str, str | None] = {"— or pick from list to open full detail —": None}
    for nid in sorted(node_data, key=lambda n: node_data[n]["title"].lower()):
        options[_short(node_data[nid]["title"], 72)] = nid

    choice = st.selectbox(
        "Open paper detail",
        list(options.keys()),
        key="graph_node_picker",
        label_visibility="collapsed",
    )
    return options[choice]


def edge_legend() -> None:
    """A small inline legend for the edge colours."""
    chips = " ".join(
        f"<span style='color:{c}'>&#9632;</span> {name.replace('_', ' ')}"
        for name, c in EDGE_COLOURS.items()
    )
    st.markdown(chips, unsafe_allow_html=True)


def _short(text: str, length: int = 40) -> str:
    text = text or ""
    return text if len(text) <= length else text[: length - 1] + "…"


def paper_detail_panel(store: Store, paper: Paper) -> None:
    """Full markdown-style paper detail, shown when clicking a graph node."""
    collection: list = st.session_state.setdefault("collection", [])
    in_collection = paper.id in collection

    col_title, col_btn = st.columns([7, 1])
    with col_title:
        st.markdown(f"## {paper.title}")
    with col_btn:
        label = "★ Saved" if in_collection else "☆ Save"
        btn_type = "secondary" if in_collection else "primary"
        if st.button(label, type=btn_type, key=f"save_{paper.id}"):
            if in_collection:
                collection.remove(paper.id)
            else:
                collection.append(paper.id)
            st.rerun()

    meta = []
    if paper.year:
        meta.append(f"**{paper.year}**")
    meta.append(paper.source.value)
    if paper.citation_count is not None:
        meta.append(f"{paper.citation_count} citations")
    st.caption(" · ".join(meta))

    if paper.authors:
        names = ", ".join(a.name for a in paper.authors[:8])
        if len(paper.authors) > 8:
            names += f" (+{len(paper.authors) - 8} more)"
        st.write(f"*{names}*")

    if paper.research_areas:
        st.write(" ".join(f"`{a.name}`" for a in paper.research_areas))

    if paper.url:
        st.markdown(f"[Open source page →]({paper.url})")

    st.divider()

    tab_abstract, tab_concepts, tab_sum, tab_fig = st.tabs(
        ["Abstract", "Concepts", "Summaries", "Figures"]
    )
    with tab_abstract:
        if paper.abstract:
            st.write(paper.abstract)
        else:
            st.info("No abstract available.")
    with tab_concepts:
        concepts = store.get_concepts_for_paper(paper.id)
        if concepts:
            st.write(" ".join(f"`{c.name}`" for c, _ in concepts))
        else:
            empty_state(
                "No concepts yet.",
                hint="python scripts/run_embed_papers.py --cluster kmeans",
            )
    with tab_sum:
        summaries_block(store, paper)
    with tab_fig:
        figure_grid(store, paper)


def empty_state(message: str, *, hint: str | None = None) -> None:
    """Consistent 'no data yet' panel."""
    st.info(message)
    if hint:
        st.code(hint, language="bash")
