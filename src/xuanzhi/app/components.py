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

# Colour per source for graph nodes (vivid so they read on dark background).
SOURCE_COLOURS = {
    "arxiv":            "#F87171",  # red
    "semantic_scholar": "#60A5FA",  # blue
    "openalex":         "#4ADE80",  # green
    "google_scholar":   "#FBBF24",  # amber
    "pubmed":           "#C084FC",  # purple
    "biorxiv":          "#22D3EE",  # cyan
    "manual":           "#94A3B8",  # slate
}

# vis.js shape per source so nodes are visually distinct beyond colour.
SOURCE_SHAPES = {
    "arxiv":            "dot",
    "semantic_scholar": "diamond",
    "openalex":         "square",
    "google_scholar":   "triangleDown",
    "pubmed":           "star",
    "biorxiv":          "triangle",
    "manual":           "dot",
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

    # Pre-compute a stable layout so nodes don't move on load.
    n = graph.number_of_nodes()
    pos = nx.spring_layout(graph, seed=42, k=2.5 / max(n ** 0.5, 1))
    scale = 800

    net = Network(height=f"{height}px", width="100%",
                  bgcolor="#0e1117", font_color="#ffffff")
    net.toggle_physics(False)

    for nid, data in graph.nodes(data=True):
        source    = data.get("source", "manual")
        degree    = graph.degree(nid)
        citations = data.get("citation_count") or 0
        title     = node_data[nid]["title"]
        x, y      = pos[nid]
        size = 9 + min(degree * 2.5, 20) + min(citations / 40, 12)
        net.add_node(
            nid,
            label=_short(title, 26),
            title=title,
            size=float(size),
            shape=SOURCE_SHAPES.get(source, "dot"),
            color=SOURCE_COLOURS.get(source, "#94A3B8"),
            x=float(x * scale),
            y=float(y * scale),
        )

    # Build adjacency map for the "Connected because" panel before adding edges.
    edge_map: dict[str, list] = {nid: [] for nid in node_data}
    for a, b, data in graph.edges(data=True):
        etype = data.get("edge_type", "shared_concept")
        evidence = data.get("evidence") or ""
        color = EDGE_COLOURS.get(etype, "#8993A4")
        reason = evidence if (evidence and evidence != etype) else etype.replace("_", " ")
        edge_map[a].append({"n": b, "reason": reason, "color": color, "type": etype})
        edge_map[b].append({"n": a, "reason": reason, "color": color, "type": etype})
        net.add_edge(
            a, b,
            color=color,
            title=reason,
            width=1,
        )

    # Inject search bar, info card, physics toggle, and vis.js polish.
    payload = json.dumps(node_data).replace("</script>", r"<\/script>")
    edge_payload = json.dumps(edge_map).replace("</script>", r"<\/script>")

    injection = f"""
<style>
  #_xz_bar   {{ box-sizing:border-box; }}
  #_xz_search::placeholder {{ color:#6c7086; }}
  #_xz_search:focus {{ border-color:#cba6f7 !important; outline:none; }}
  #_xz_panel::-webkit-scrollbar {{ width:4px; }}
  #_xz_panel::-webkit-scrollbar-thumb {{ background:#45475a; border-radius:2px; }}
  ._xz_tag   {{ background:#313244; padding:2px 8px; border-radius:4px;
                margin:2px 3px 2px 0; display:inline-block; font-size:10px;
                color:#a6e3a1; }}
  ._xz_btn   {{ background:#313244; color:#cdd6f4; border:1px solid #45475a;
                padding:5px 11px; border-radius:6px; font-size:12px;
                cursor:pointer; transition:background .15s; }}
  ._xz_btn:hover {{ background:#45475a; }}
</style>

<!-- top toolbar: search + animate toggle -->
<div id="_xz_bar" style="
    position:fixed; top:10px; left:10px; right:10px;
    display:flex; gap:8px; align-items:center; z-index:9998;">
  <input id="_xz_search" type="text" placeholder="🔍  search papers…"
    oninput="_xzSearch(this.value)"
    style="flex:1; background:#1e1e2e; color:#cdd6f4;
           border:1px solid #45475a; border-radius:7px;
           padding:6px 12px; font-size:12px; font-family:system-ui,sans-serif;" />
  <button class="_xz_btn" id="_xz_phys" onclick="_xzToggle()">▶ Animate</button>
  <button class="_xz_btn" onclick="_xzFit()" title="Fit graph">⊞ Fit</button>
</div>

<!-- floating info card -->
<div id="_xz_panel" style="
    display:none; position:fixed; top:52px; right:12px; width:300px;
    max-height:calc(100vh - 70px); overflow-y:auto;
    background:#1e1e2e; color:#cdd6f4;
    border:1px solid #45475a; border-radius:12px;
    font-family:system-ui,sans-serif; font-size:12px;
    line-height:1.6; z-index:9999;
    box-shadow:0 12px 40px rgba(0,0,0,.8);">
  <div id="_xz_card_header" style="
      padding:12px 14px 10px; border-radius:12px 12px 0 0;
      background:linear-gradient(135deg,#313244,#1e1e2e);">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
      <span id="_xz_card_title"
            style="color:#cba6f7;font-size:13px;font-weight:600;flex:1;line-height:1.4"></span>
      <span onclick="document.getElementById('_xz_panel').style.display='none'"
            style="cursor:pointer;color:#6c7086;font-size:18px;flex-shrink:0;
                   line-height:1;margin-top:-1px">✕</span>
    </div>
    <div id="_xz_card_authors" style="color:#a6e3a1;font-size:11px;margin-top:5px"></div>
    <div id="_xz_card_meta"    style="color:#6c7086;font-size:11px;margin-top:3px"></div>
  </div>
  <div id="_xz_card_body" style="padding:10px 14px 14px"></div>
</div>

<script>
var _xzData  = {payload};
var _xzEdges = {edge_payload};
var _xzOn    = false;

var _PHYS_OPTS = {{
    physics: {{
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {{
            gravitationalConstant: -26,
            centralGravity: 0.005,
            springLength: 230,
            springConstant: 0.18,
            damping: 0.85,
            avoidOverlap: 0.1
        }},
        stabilization: {{ enabled: false }}
    }}
}};

function _xzToggle() {{
    var btn = document.getElementById('_xz_phys');
    if (_xzOn) {{
        _xzOn = false;
        network.stopSimulation();
        network.setOptions({{physics:{{enabled:false}}}});
        btn.textContent = '▶ Animate';
        return;
    }}
    _xzOn = true;
    btn.textContent = '⏸ Freeze';
    network.setOptions(_PHYS_OPTS);
}}

function _xzFit() {{ network.fit({{animation:{{duration:400,easingFunction:'easeInOutQuad'}}}}); }}

function _xzSearch(q) {{
    q = q.trim().toLowerCase();
    var updates = [];
    Object.keys(_xzData).forEach(function(id) {{
        var match = !q || (_xzData[id].title || '').toLowerCase().indexOf(q) >= 0;
        updates.push({{id:id, opacity: match ? 1.0 : 0.12,
                       font:{{color: match ? '#cdd6f4' : 'transparent'}}}});
    }});
    network.body.data.nodes.update(updates);
}}

(function _xzWait() {{
    if (typeof network === 'undefined') {{ setTimeout(_xzWait, 60); return; }}

    /* ── visual polish ─────────────────────────────────────────── */
    network.setOptions({{
        nodes: {{
            shadow: {{enabled:true, color:'rgba(0,0,0,0.55)', size:9, x:0, y:4}},
            font: {{
                size: 11, face:'system-ui,-apple-system,sans-serif',
                color:'#e2e8f0', strokeWidth:4, strokeColor:'#080b10'
            }},
            borderWidth: 1.5, borderWidthSelected: 3.5,
            scaling: {{
                min:8, max:42,
                label: {{
                    enabled:true, min:9, max:14,
                    drawThreshold:8, maxVisible:18
                }}
            }},
            chosen: {{
                node: function(values, id, selected, hovering) {{
                    var n  = network.body.data.nodes.get(id);
                    var c  = n && n.color;
                    var bg = c ? (typeof c === 'string' ? c : (c.background || '#888')) : '#888';
                    if (hovering) {{
                        values.shadowColor = bg;
                        values.shadowSize  = 22;
                        values.shadowX = 0; values.shadowY = 0;
                        values.size = values.size * 1.12;
                    }}
                    if (selected) {{
                        values.borderWidth = 4;
                        values.shadowColor = bg;
                        values.shadowSize  = 32;
                        values.shadowX = 0; values.shadowY = 0;
                    }}
                }}
            }}
        }},
        edges: {{
            smooth: {{type:'dynamic', roundness:0.2}},
            width: 0.75, hoverWidth: 2, selectionWidth: 3,
            shadow: {{enabled:false}}
        }},
        interaction: {{
            hover:true, tooltipDelay:100,
            hideEdgesOnDrag:false,
            keyboard: {{enabled:true, bindToWindow:false}}
        }}
    }});

    /* semi-transparent white border + vivid highlight per node */
    var _borderUpd = [];
    network.body.data.nodes.forEach(function(n) {{
        var c  = n.color;
        var bg = c ? (typeof c === 'string' ? c : (c.background || '#94A3B8')) : '#94A3B8';
        _borderUpd.push({{
            id: n.id,
            color: {{
                background: bg,
                border: 'rgba(255,255,255,0.22)',
                highlight: {{background: bg, border: '#ffffff'}},
                hover:     {{background: bg, border: 'rgba(255,255,255,0.75)'}}
            }}
        }});
    }});
    network.body.data.nodes.update(_borderUpd);

    /* ── node click → info card ────────────────────────────────── */
    network.on('click', function(p) {{
        var panel = document.getElementById('_xz_panel');
        if (!p.nodes.length) {{ panel.style.display='none'; return; }}
        var id = p.nodes[0];
        var d  = _xzData[id];
        if (!d) return;

        document.getElementById('_xz_card_title').textContent   = d.title;
        document.getElementById('_xz_card_authors').textContent = d.authors || '';

        var meta = [];
        if (d.year)      meta.push(d.year);
        if (d.source)    meta.push(d.source);
        if (d.citations) meta.push(d.citations + ' citations');
        document.getElementById('_xz_card_meta').textContent = meta.join(' · ');

        var body = '';
        if (d.areas && d.areas.length)
            body += '<div style="margin-bottom:9px">'
                  + d.areas.map(function(a){{return '<span class="_xz_tag">'+a+'</span>';}}).join('')
                  + '</div>';
        if (d.abstract)
            body += '<div style="color:#bac2de;border-top:1px solid #313244;'
                  + 'padding-top:9px;font-size:11px;line-height:1.6">'
                  + d.abstract + '…</div>';

        /* ── "Connected because" section ── */
        var conns = (_xzEdges[id] || []).slice();
        if (conns.length) {{
            conns.sort(function(a,b){{ return a.type.localeCompare(b.type); }});
            var shown = conns.slice(0, 8);
            body += '<div style="margin-top:12px;border-top:1px solid #313244;padding-top:9px">'
                  + '<div style="font-size:10px;color:#6c7086;text-transform:uppercase;'
                  + 'letter-spacing:0.8px;margin-bottom:7px">Connected because</div>';
            shown.forEach(function(c) {{
                var nb = _xzData[c.n];
                var nbTitle = nb ? nb.title : c.n;
                nbTitle = nbTitle.length > 44 ? nbTitle.slice(0,43)+'…' : nbTitle;
                body += '<div style="display:flex;gap:7px;align-items:flex-start;margin-bottom:6px">'
                      + '<span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;'
                      + 'background:'+c.color+';margin-top:3px"></span>'
                      + '<div><div style="color:#a6adc8;font-size:11px;line-height:1.4">'+nbTitle+'</div>'
                      + '<div style="color:#6c7086;font-size:10px;margin-top:1px">'+c.reason+'</div>'
                      + '</div></div>';
            }});
            if (conns.length > 8)
                body += '<div style="color:#6c7086;font-size:10px;margin-top:2px">+'+(conns.length-8)+' more connections</div>';
            body += '</div>';
        }}

        if (d.url)
            body += '<div style="margin-top:10px">'
                  + '<a href="'+d.url+'" target="_blank" '
                  + 'style="color:#89b4fa;font-size:11px;text-decoration:none;'
                  + 'border:1px solid #89b4fa;padding:3px 10px;border-radius:5px">'
                  + 'Open source ↗</a></div>';

        document.getElementById('_xz_card_body').innerHTML = body;
        panel.style.display = 'block';
    }});

    /* auto-fit so all nodes are visible on first load */
    network.fit({{animation:false}});

    /* ── hover: dim non-neighbours ─────────────────────────────── */
    var _xzHovered = null;
    network.on('hoverNode', function(p) {{
        if (_xzHovered === p.node) return;
        _xzHovered = p.node;
        var neighbours = new Set(network.getConnectedNodes(p.node));
        neighbours.add(p.node);
        var upd = [];
        Object.keys(_xzData).forEach(function(id) {{
            upd.push({{id:id, opacity: neighbours.has(id) ? 1.0 : 0.18}});
        }});
        network.body.data.nodes.update(upd);
    }});
    network.on('blurNode', function() {{
        _xzHovered = null;
        var upd = Object.keys(_xzData).map(function(id){{return {{id:id,opacity:1.0}};}});
        network.body.data.nodes.update(upd);
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
