# app.py
# SENTRY — RoRS Research Security Operations Dashboard (FULL SCRIPT)
#
# This version matches your *new* direction:
# ✅ "Attention needed" is NOT venue-based anymore.
# ✅ Attention rules are:
#    - F1: US federal funding (e.g., NSF/DOE/DARPA/NIH/ONR/AFRL/ARPA-E) + NON-US author country
#    - I1: Name-collision / identity ambiguity (same name appears across many countries/affiliations/email domains)
#    - M1: Metadata mismatch placeholder (auto-activates only if your data later includes header/footer fields)
# ✅ Map is rectangular, edge-to-edge using pydeck (no oval projection).
# ✅ Map uses 👤 glyph INSIDE the map (TextLayer) instead of circles.
#    (True silhouette icons in Plotly without Mapbox token are not reliable; pydeck TextLayer is stable.)
# ✅ KPI sparklines are inside the KPI cards.
# ✅ Case Detail uses label “Evidence-grounded review report”.
#
# Run:
#   streamlit run app.py
#
# Data files expected:
#   data/authors.json
#   data/documents.json
#   data/occurrences.json
#   data/evidence.json (optional)

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import pydeck as pdk
except Exception:
    pdk = None

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="SENTRY: RoRS Research Security Dashboard", layout="wide")

DATA_DIR = Path("data")
AUTHORS_PATH = DATA_DIR / "authors.json"
DOCS_PATH = DATA_DIR / "documents.json"
OCC_PATH = DATA_DIR / "occurrences.json"
EVIDENCE_PATH = DATA_DIR / "evidence.json"  # optional

# No-API geocoding: country centroids (expand as needed)
COUNTRY_CENTROIDS = {
    "USA": (39.8283, -98.5795),
    "US": (39.8283, -98.5795),
    "UNITED STATES": (39.8283, -98.5795),
    "UNITED STATES OF AMERICA": (39.8283, -98.5795),
    "UK": (55.3781, -3.4360),
    "UNITED KINGDOM": (55.3781, -3.4360),
    "JAPAN": (36.2048, 138.2529),
    "GERMANY": (51.1657, 10.4515),
    "CHINA": (35.8617, 104.1954),
    "CANADA": (56.1304, -106.3468),
    "SINGAPORE": (1.3521, 103.8198),
}

US_COUNTRY_ALIASES = {"USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"}

# ---------------------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
body, .stApp {background-color: #0e1117; color: #e0e0e0;}
h1,h2,h3,h4 {color:#90caf9;}
.small {font-size: 0.9rem; opacity: 0.9;}

.badge {display:inline-block; padding:0.15rem 0.6rem; border-radius:999px; background:#22293a; border:1px solid #2d3752; font-size:0.8rem;}
.badge-flag {background:#3a1f1f; border-color:#5c2d2d;}
.badge-ok {background:#1f2f23; border-color:#2d4a35;}

hr {border-color:#232838;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

authors = load_json(AUTHORS_PATH) or []
documents = load_json(DOCS_PATH) or []
occurrences = load_json(OCC_PATH) or []
evidence = load_json(EVIDENCE_PATH) or None  # optional

authors_df = pd.DataFrame(authors)
docs_df = pd.DataFrame(documents)
occ_df = pd.DataFrame(occurrences)

# Numeric coercion for scoring columns
for df in (authors_df, docs_df):
    for col in ["A1", "A2", "A_final"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize occurrences venue field name if needed
if not occ_df.empty:
    if "conf/journal" not in occ_df.columns and "conf_journal" in occ_df.columns:
        occ_df["conf/journal"] = occ_df["conf_journal"]

# Fast lookup for documents
doc_by_id = {d.get("doc_id"): d for d in documents if isinstance(d, dict) and d.get("doc_id")}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_list(x):
    return x if isinstance(x, list) else []

def badge_status(s: str) -> str:
    s = (s or "").upper()
    if "FLAG" in s:
        return "<span class='badge badge-flag'>FLAGGED</span>"
    return "<span class='badge badge-ok'>NORMAL</span>"

def is_us_country(country: str) -> bool:
    c = (country or "").strip().upper()
    return c in US_COUNTRY_ALIASES

def distribution_spark(values, bins=14):
    """Sparkline based on distribution shape (no timestamps in your dataset)."""
    s = pd.Series(values).dropna()
    if s.empty:
        return []
    b = pd.cut(s, bins=bins, labels=False, duplicates="drop")
    bc = pd.Series(b).value_counts().sort_index()
    full = [int(bc.get(i, 0)) for i in range(int(bc.index.max()) + 1)]
    return full if len(full) >= 2 else []

def kpi_card(title: str, value: str, spark_y=None, delta_text: str = "", accent="#8b93ff"):
    """KPI card with sparkline INSIDE the same card box."""
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
              <div>
                <div style="opacity:.85; font-size:.95rem;">{title}</div>
                <div style="font-size:2.2rem; font-weight:800; margin-top:6px;">{value}</div>
                <div style="opacity:.7; font-size:.85rem; margin-top:2px;">{delta_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if spark_y and len(spark_y) >= 2:
            fig = go.Figure(
                go.Scatter(
                    x=list(range(len(spark_y))),
                    y=spark_y,
                    mode="lines",
                    line=dict(width=3, color=accent),
                    hoverinfo="skip",
                )
            )
            fig.update_layout(
                height=90,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def score_line(row: dict, title="Anomaly Score Progression"):
    vals = {"A1": row.get("A1"), "A2": row.get("A2"), "Final": row.get("A_final")}
    fig = go.Figure(
        go.Scatter(
            x=list(vals.keys()),
            y=list(vals.values()),
            mode="lines+markers",
            line=dict(width=3),
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=35, b=20),
        title=title,
        template="plotly_white",
    )
    fig.update_yaxes(range=[0, 1])
    return fig

def metrics_bar(metrics: dict, title="Metric Breakdown"):
    metrics = metrics or {}
    num = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not num:
        return None
    df = pd.DataFrame({"Metric": list(num.keys()), "Value": list(num.values())})
    return px.bar(df, x="Metric", y="Value", title=title, height=260, template="plotly_white")

def compute_dashboard_metrics(a_df: pd.DataFrame, d_df: pd.DataFrame):
    open_statuses = {"FLAGGED", "REVIEW", "OPEN", "IN REVIEW"}
    a_status = a_df.get("status", pd.Series([], dtype=str)).fillna("").astype(str).str.upper()
    d_status = d_df.get("status", pd.Series([], dtype=str)).fillna("").astype(str).str.upper()

    open_alerts = int(
        (a_status.isin(open_statuses) | a_status.str.contains("FLAG")).sum()
        + (d_status.isin(open_statuses) | d_status.str.contains("FLAG")).sum()
    )

    critical_high = int(
        (a_df.get("A_final", pd.Series([], dtype=float)).fillna(0) >= 0.8).sum()
        + (d_df.get("A_final", pd.Series([], dtype=float)).fillna(0) >= 0.8).sum()
    )

    total = max(len(a_df) + len(d_df), 1)
    flagged = int(a_status.str.contains("FLAG").sum() + d_status.str.contains("FLAG").sum())
    pct_attention = round(100 * flagged / total, 1)

    return open_alerts, critical_high, pct_attention

def build_triage_queue(a_df: pd.DataFrame, d_df: pd.DataFrame):
    a = a_df.copy()
    d = d_df.copy()

    a["case_kind"] = "AUTHOR"
    a["case_key"] = a.get("case_id")
    a["title_or_name"] = a.get("name_string")

    d["case_kind"] = "DOCUMENT"
    d["case_key"] = d.get("doc_id")
    d["title_or_name"] = d.get("title")

    keep = ["case_kind", "case_key", "title_or_name", "status", "A1", "A2", "A_final", "top_reason"]
    a = a[[c for c in keep if c in a.columns]]
    d = d[[c for c in keep if c in d.columns]]

    q = pd.concat([a, d], ignore_index=True)
    if "A_final" in q.columns:
        q = q.sort_values("A_final", ascending=False, na_position="last")
    return q

# ---------------------------------------------------------------------
# ATTENTION NEEDED — NEW RULE ENGINE (funding + identity ambiguity + metadata)
# ---------------------------------------------------------------------
def _doc_us_funding_flags(doc_obj: dict, us_agencies_set: set[str]) -> tuple[bool, list[str]]:
    """
    Returns: (has_us_federal_funding, agencies_list)
    Supports:
      - doc["funding"]["has_us_federal_funding"] + doc["funding"]["agencies"]
      - OR doc["funding_agencies"] flat list
      - OR doc["funding_text"] (string) naive scan
    """
    if not isinstance(doc_obj, dict):
        return (False, [])

    # Preferred nested format:
    funding = doc_obj.get("funding")
    if isinstance(funding, dict):
        agencies = funding.get("agencies", [])
        agencies = agencies if isinstance(agencies, list) else []
        agencies_norm = [str(a).strip() for a in agencies if str(a).strip()]
        has_flag = bool(funding.get("has_us_federal_funding", False))
        # If flag missing, infer from agencies list
        if not has_flag and agencies_norm:
            has_flag = any(a.strip().upper() in us_agencies_set for a in agencies_norm)
        return (has_flag, agencies_norm)

    # Flat list format:
    agencies = doc_obj.get("funding_agencies")
    if isinstance(agencies, list):
        agencies_norm = [str(a).strip() for a in agencies if str(a).strip()]
        has_flag = any(a.strip().upper() in us_agencies_set for a in agencies_norm)
        return (has_flag, agencies_norm)

    # Text format:
    txt = doc_obj.get("funding_text")
    if isinstance(txt, str) and txt.strip():
        txt_up = txt.upper()
        hit = [a for a in us_agencies_set if a in txt_up]
        return (len(hit) > 0, sorted(hit))

    return (False, [])

def compute_attention_flags(
    occ_df: pd.DataFrame,
    doc_by_id: dict,
    us_agencies_set: set[str],
    min_countries_for_collision: int = 3,
    min_affils_for_collision: int = 3,
    min_email_domains_for_collision: int = 3,
) -> pd.DataFrame:
    """
    Returns a row-level table of "attention needed" flags.

    Rules:
      F1: US federal funding detected for doc_id AND author country is non-US
      I1: name_string appears across many countries / affiliations / email domains (identity ambiguity)
      M1: metadata mismatch placeholder (activates only if columns exist, e.g., email_header vs email_footer)
    """
    if occ_df.empty:
        return pd.DataFrame()

    df = occ_df.copy()

    # Normalize fields
    df["country_norm"] = df.get("country", "").fillna("").astype(str).str.strip()
    df["name_norm"] = df.get("name_string", "").fillna("").astype(str).str.strip()
    df["affil_norm"] = df.get("affiliation", "").fillna("").astype(str).str.strip()
    df["email_norm"] = df.get("email_domain", "").fillna("").astype(str).str.strip()
    df["venue_norm"] = df.get("conf/journal", "").fillna("").astype(str).str.strip()

    # Add doc funding lookup to each row
    has_us_list = []
    agencies_list = []
    for doc_id in df.get("doc_id", pd.Series([""] * len(df))).tolist():
        doc = doc_by_id.get(doc_id, {})
        has_us, agencies = _doc_us_funding_flags(doc, us_agencies_set)
        has_us_list.append(bool(has_us))
        agencies_list.append(agencies)
    df["doc_has_us_funding"] = has_us_list
    df["doc_funding_agencies"] = agencies_list

    rows = []

    # ----------------
    # F1: US funding + non-US author country
    # ----------------
    f1 = df[(df["doc_has_us_funding"] == True) & (~df["country_norm"].apply(is_us_country))].copy()
    if not f1.empty:
        f1["rule_id"] = "F1"
        f1["rule_label"] = "US federal funding (agency) → Non-US author country (review for rare funding / eligibility)"
        rows.append(f1)

    # ----------------
    # I1: Identity ambiguity / name collision / multi-cluster behavior
    # ----------------
    g = df.groupby("name_norm").agg(
        distinct_countries=("country_norm", lambda s: s[s != ""].nunique()),
        distinct_affils=("affil_norm", lambda s: s[s != ""].nunique()),
        distinct_emails=("email_norm", lambda s: s[s != ""].nunique()),
        total_rows=("name_norm", "count"),
    )
    amb_names = g[
        (g["distinct_countries"] >= min_countries_for_collision)
        | (g["distinct_affils"] >= min_affils_for_collision)
        | (g["distinct_emails"] >= min_email_domains_for_collision)
    ].index.tolist()

    i1 = df[df["name_norm"].isin(amb_names)].copy()
    if not i1.empty:
        i1["rule_id"] = "I1"
        i1["rule_label"] = "Identity ambiguity: same author name spans many countries/affiliations/email domains"
        rows.append(i1)

    # ----------------
    # M1: Metadata mismatch placeholder (auto-activates only if your data later includes these)
    # Example future columns:
    #   email_domain_header, email_domain_footer
    # ----------------
    if ("email_domain_header" in df.columns) and ("email_domain_footer" in df.columns):
        m1 = df[df["email_domain_header"].fillna("").astype(str).str.strip()
                != df["email_domain_footer"].fillna("").astype(str).str.strip()].copy()
        if not m1.empty:
            m1["rule_id"] = "M1"
            m1["rule_label"] = "Metadata mismatch: email differs between header and footer (verify extraction / identity)"
            rows.append(m1)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # Deduplicate identical row+rule combos
    dedup_cols = []
    for c in ["rule_id", "case_id", "doc_id", "evidence_id", "page", "name_norm", "country_norm", "email_norm", "affil_norm"]:
        if c in out.columns:
            dedup_cols.append(c)
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols)

    # Select columns for display
    keep = [
        "rule_id", "rule_label",
        "case_id", "name_string",
        "doc_id",
        "country", "conf/journal",
        "affiliation", "email_domain",
        "doc_funding_agencies",
        "evidence_id", "page",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()

def compute_attention_needed_cases(flags_df: pd.DataFrame) -> dict:
    if flags_df.empty:
        return {"author_ids": set(), "doc_ids": set(), "count_unique_cases": 0}
    author_ids = set(flags_df["case_id"].dropna().unique()) if "case_id" in flags_df.columns else set()
    doc_ids = set(flags_df["doc_id"].dropna().unique()) if "doc_id" in flags_df.columns else set()
    return {"author_ids": author_ids, "doc_ids": doc_ids, "count_unique_cases": len(author_ids) + len(doc_ids)}

# ---------------------------------------------------------------------
# REPORT / EVIDENCE PANELS
# ---------------------------------------------------------------------
def show_case_report(report: dict):
    report = report or {}
    st.markdown("#### Evidence-grounded review report")
    st.write(f"**{report.get('report_title','Report')}**")
    st.write(report.get("summary", ""))

    kp = safe_list(report.get("key_points", []))
    if kp:
        st.markdown("**Key points (with evidence IDs):**")
        for x in kp:
            st.write(x)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Uncertainty:** {report.get('uncertainty','')}")
    c2.markdown(f"**Evidence IDs:** {', '.join(safe_list(report.get('evidence_ids', [])))}")
    c3.markdown(f"<span class='small'>{report.get('no_claims_notice','')}</span>", unsafe_allow_html=True)

def show_evidence_panel_for_author(case_id: str, author_obj: dict):
    st.markdown("#### Evidence panel (grounding)")
    report = (author_obj or {}).get("report", {}) or {}
    ev_ids = safe_list(report.get("evidence_ids", []))

    if occ_df.empty:
        st.info("No occurrences.json loaded.")
        return

    sub = occ_df[occ_df.get("case_id") == case_id].copy()
    if ev_ids and "evidence_id" in sub.columns:
        sub = sub[sub["evidence_id"].isin(ev_ids)].copy()

    if sub.empty:
        st.info("No matching occurrences for this author + evidence IDs.")
        return

    cols = ["evidence_id", "doc_id", "page", "field_type", "affiliation", "country", "email_domain", "conf/journal"]
    cols = [c for c in cols if c in sub.columns]
    sub = sub[cols].rename(columns={
        "evidence_id": "Evidence ID",
        "doc_id": "Doc",
        "page": "Page",
        "field_type": "Field",
        "conf/journal": "Venue",
        "email_domain": "Email domain",
    })
    st.dataframe(sub, use_container_width=True, hide_index=True)
    st.caption("To fully match RoRS workflow, store verbatim snippet text per Evidence ID in evidence.json.")

def show_author_detail(case_id: str):
    author = next((a for a in authors if a.get("case_id") == case_id), None)
    if not author:
        st.warning("Author not found.")
        return

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.markdown(f"### {author.get('name_string','(unknown)')}")
        st.markdown(f"Status: {badge_status(author.get('status'))}", unsafe_allow_html=True)
        st.write(f"**Top reason:** {author.get('top_reason','')}")
        sigs = safe_list(author.get("signals", []))
        if sigs:
            st.markdown("**Triggered signals:**")
            st.write(" • " + "\n • ".join(sigs))
        st.plotly_chart(score_line(author, title="A1 → A2 → Afinal"), use_container_width=True)
        mb = metrics_bar(author.get("metrics", {}), title="Author metrics")
        if mb:
            st.plotly_chart(mb, use_container_width=True)

    with c2:
        show_case_report(author.get("report", {}))
        show_evidence_panel_for_author(case_id, author)

        docs = safe_list(author.get("docs", []))
        if docs and not docs_df.empty and "doc_id" in docs_df.columns:
            st.markdown("#### Linked documents")
            cols = [c for c in ["doc_id", "title", "year", "A_final", "status"] if c in docs_df.columns]
            doc_rows = docs_df[docs_df["doc_id"].isin(docs)][cols].copy()
            st.dataframe(doc_rows, use_container_width=True, hide_index=True)

def show_doc_detail(doc_id: str):
    doc = next((d for d in documents if d.get("doc_id") == doc_id), None)
    if not doc:
        st.warning("Document not found.")
        return

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.markdown(f"### {doc.get('title','(unknown)')}")
        st.markdown(
            f"<span class='badge'>{doc_id}</span> &nbsp; Status: {badge_status(doc.get('status'))}",
            unsafe_allow_html=True,
        )
        st.write(f"**Year:** {doc.get('year','')}")
        st.write(f"**Top reason:** {doc.get('top_reason','')}")
        sigs = safe_list(doc.get("signals", []))
        if sigs:
            st.markdown("**Triggered signals:**")
            st.write(" • " + "\n • ".join(sigs))

        # Funding summary (new)
        has_us, agencies = _doc_us_funding_flags(doc, us_agencies_set=set())
        funding = doc.get("funding", {})
        if isinstance(funding, dict):
            has_us = bool(funding.get("has_us_federal_funding", False)) or has_us
            agencies = funding.get("agencies", agencies) or agencies
        agencies = agencies if isinstance(agencies, list) else []
        if agencies:
            st.write(f"**Funding agencies:** {', '.join([str(a) for a in agencies])}")
        st.write(f"**US federal funding:** {'Yes' if has_us else 'No'}")

        st.plotly_chart(score_line(doc, title="Document A1 → A2 → Afinal"), use_container_width=True)
        mb = metrics_bar(doc.get("metrics", {}), title="Document metrics")
        if mb:
            st.plotly_chart(mb, use_container_width=True)

    with c2:
        show_case_report(doc.get("report", {}))
        if not occ_df.empty and "doc_id" in occ_df.columns:
            sub = occ_df[occ_df["doc_id"] == doc_id].copy()
            if not sub.empty:
                st.markdown("#### Authors on this document (from occurrences)")
                cols = ["case_id", "name_string", "affiliation", "country", "conf/journal", "evidence_id", "page"]
                cols = [c for c in cols if c in sub.columns]
                st.dataframe(sub[cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# GLOBAL MAP — pydeck (RECTANGULAR, edge-to-edge, 👤)
# ---------------------------------------------------------------------
def build_global_map_pydeck(occ_df: pd.DataFrame):
    if pdk is None:
        return None, ["pydeck not available in this environment (install pydeck)."]

    if occ_df.empty:
        return None, []

    tmp = occ_df.copy()
    tmp["country_norm"] = tmp.get("country", "").fillna("").astype(str).str.strip()
    tmp = tmp[tmp["country_norm"] != ""].copy()
    if tmp.empty:
        return None, []

    tmp["country_key"] = tmp["country_norm"].str.upper()
    tmp["lat"] = tmp["country_key"].map(lambda c: COUNTRY_CENTROIDS.get(c, (None, None))[0])
    tmp["lon"] = tmp["country_key"].map(lambda c: COUNTRY_CENTROIDS.get(c, (None, None))[1])

    missing = sorted(tmp.loc[tmp["lat"].isna(), "country_norm"].unique().tolist())
    tmp = tmp.dropna(subset=["lat", "lon"]).copy()
    if tmp.empty:
        return None, missing

    # Aggregate per country (for clean pins)
    agg = (
        tmp.groupby(["country_norm", "lat", "lon"], as_index=False)
        .agg(author_count=("case_id", "nunique"), rows=("evidence_id", "count"))
    )

    # Names list for hover
    names_per = (
        tmp.groupby("country_norm")["name_string"]
        .apply(lambda s: sorted(set(s.dropna().astype(str)))[:15])
        .to_dict()
    )

    agg["hover"] = agg.apply(
        lambda r: f"{r['country_norm']} • Authors: {int(r['author_count'])} • Rows: {int(r['rows'])}\n"
                  + "\n".join([f"👤 {n}" for n in names_per.get(r["country_norm"], [])]),
        axis=1
    )

    # Size scaling
    agg["size"] = (agg["author_count"] ** 0.5 * 18).clip(18, 46)

    # Base map style (no Mapbox token needed)
    MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    # Layer 1: small invisible scatter point for good hover anchor
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=agg,
        get_position="[lon, lat]",
        get_radius="size * 9000",
        pickable=True,
        opacity=0.01,
    )

    # Layer 2: 👤 glyph
    text = pdk.Layer(
        "TextLayer",
        data=agg,
        get_position="[lon, lat]",
        get_text="'👤'",
        get_size="size",
        get_color=[255, 82, 82],
        get_angle=0,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=0.8)

    deck = pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=view_state,
        layers=[scatter, text],
        tooltip={"text": "{hover}"},
    )
    return deck, missing

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Global filters")
    kind_filter = st.multiselect("Case types", ["AUTHOR", "DOCUMENT"], default=["AUTHOR", "DOCUMENT"])
    status_filter = st.multiselect("Status", ["FLAGGED", "NORMAL"], default=["FLAGGED", "NORMAL"])
    min_score = st.slider("Min Afinal", 0.0, 1.0, 0.0, 0.01)

    st.markdown("---")
    st.markdown("### Attention rules")
    us_agencies_text = st.text_area(
        "US federal agencies (comma-separated)",
        value="NSF, DOE, DARPA, NIH, ONR, AFRL, ARPA-E",
        height=80,
        help="Used to infer US federal funding if documents.json only lists agencies without has_us_federal_funding.",
    )
    us_agencies_set = {x.strip().upper() for x in us_agencies_text.split(",") if x.strip()}

    st.markdown("#### Identity ambiguity thresholds (I1)")
    min_countries = st.slider("Min distinct countries (same name)", 2, 6, 3, 1)
    min_affils = st.slider("Min distinct affiliations (same name)", 2, 10, 3, 1)
    min_emails = st.slider("Min distinct email domains (same name)", 2, 10, 3, 1)

# Compute attention flags (new)
flags_df = compute_attention_flags(
    occ_df=occ_df,
    doc_by_id=doc_by_id,
    us_agencies_set=us_agencies_set,
    min_countries_for_collision=min_countries,
    min_affils_for_collision=min_affils,
    min_email_domains_for_collision=min_emails,
)
attention = compute_attention_needed_cases(flags_df)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.header("SENTRY — RoRS Research Security Operations Dashboard")
st.caption("Evidence-grounded triage aligned to: triage → inspect → explain → feedback.")

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tabs = st.tabs(
    [
        "Triage Queue",
        "Case Detail",
        "Global Overview",
        "Rule Inspector",
        "Collaborations",
        "Pipeline Health",
    ]
)

# ---------------------------------------------------------------------
# TAB 0 — TRIAGE QUEUE
# ---------------------------------------------------------------------
with tabs[0]:
    open_alerts, critical_high, pct_attention = compute_dashboard_metrics(authors_df, docs_df)
    attention_needed = attention["count_unique_cases"]

    # Sparklines (distribution-based)
    spark_all = distribution_spark(
        pd.concat(
            [
                authors_df.get("A_final", pd.Series(dtype=float)),
                docs_df.get("A_final", pd.Series(dtype=float)),
            ],
            ignore_index=True,
        )
    )
    spark_auth = distribution_spark(authors_df.get("A_final", pd.Series(dtype=float)))
    spark_docs = distribution_spark(docs_df.get("A_final", pd.Series(dtype=float)))
    spark_mix = distribution_spark([open_alerts, critical_high, attention_needed, pct_attention])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Open alerts", str(open_alerts), spark_y=spark_all, delta_text="(distribution view)")
    with c2:
        kpi_card("Critical/High (Afinal ≥ 0.80)", str(critical_high), spark_y=spark_auth, delta_text="(author scores)")
    with c3:
        kpi_card("Attention needed (rules fired)", str(attention_needed), spark_y=spark_docs, delta_text="(document scores)")
    with c4:
        kpi_card("Assets requiring attention", f"{pct_attention}%", spark_y=spark_mix, delta_text="(mix of metrics)")

    st.markdown("## Unified triage queue")

    queue = build_triage_queue(authors_df, docs_df)

    # Filters
    if kind_filter:
        queue = queue[queue["case_kind"].isin(kind_filter)]

    if status_filter:
        qst = queue["status"].fillna("").astype(str).str.upper()
        is_flagged = qst.str.contains("FLAG")
        masks = []
        if "FLAGGED" in status_filter:
            masks.append(is_flagged)
        if "NORMAL" in status_filter:
            masks.append(~is_flagged)
        if masks:
            keep_mask = masks[0]
            for mm in masks[1:]:
                keep_mask = keep_mask | mm
            queue = queue[keep_mask]

    queue = queue[queue["A_final"].fillna(0) >= min_score]

    q = st.text_input("Search (name/title contains)")
    if q:
        queue = queue[queue["title_or_name"].fillna("").str.contains(q, case=False, na=False)]

    show_cols = ["case_kind", "case_key", "title_or_name", "status", "A1", "A2", "A_final", "top_reason"]
    show_cols = [c for c in show_cols if c in queue.columns]
    st.dataframe(queue[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## Attention needed — flagged by rules (F1 / I1 / M1)")
    if flags_df.empty:
        st.success("No attention-needed cases detected under current rule settings.")
    else:
        show = flags_df.rename(
            columns={
                "rule_id": "Rule",
                "rule_label": "Rule description",
                "case_id": "Author ID",
                "name_string": "Author",
                "doc_id": "Doc",
                "conf/journal": "Venue",
                "email_domain": "Email domain",
                "evidence_id": "Evidence ID",
                "page": "Page",
                "doc_funding_agencies": "Funding agencies",
            }
        ).rename(columns={"country": "Country", "affiliation": "Affiliation"})
        cols = [
            "Rule",
            "Rule description",
            "Author",
            "Author ID",
            "Doc",
            "Country",
            "Funding agencies",
            "Venue",
            "Affiliation",
            "Email domain",
            "Evidence ID",
            "Page",
        ]
        cols = [c for c in cols if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 1 — CASE DETAIL
# ---------------------------------------------------------------------
with tabs[1]:
    left, right = st.columns([0.28, 0.72])

    with left:
        st.markdown("### Select case")
        case_mode = st.radio("Case type", ["AUTHOR", "DOCUMENT"], horizontal=True)

        if case_mode == "AUTHOR":
            if authors_df.empty or "case_id" not in authors_df.columns:
                st.info("No authors loaded.")
                sel_id = None
            else:
                options = authors_df.sort_values("A_final", ascending=False)[["case_id", "name_string", "A_final"]].copy()
                options["label"] = options.apply(
                    lambda r: f"{r['name_string']} • {r['case_id']} • Afinal {r['A_final']:.2f}", axis=1
                )

                only_attention = st.checkbox("Only Attention needed", value=False)
                if only_attention:
                    options = options[options["case_id"].isin(sorted(list(attention["author_ids"])))]

                sel_id = None
                if options.empty:
                    st.info("No authors match the filter.")
                else:
                    sel = st.selectbox("Author", options["label"].tolist())
                    sel_id = options.loc[options["label"] == sel, "case_id"].iloc[0]
        else:
            if docs_df.empty or "doc_id" not in docs_df.columns:
                st.info("No documents loaded.")
                sel_id = None
            else:
                options = docs_df.sort_values("A_final", ascending=False)[["doc_id", "title", "A_final"]].copy()
                options["label"] = options.apply(
                    lambda r: f"{r['title']} • {r['doc_id']} • Afinal {r['A_final']:.2f}", axis=1
                )

                only_attention = st.checkbox("Only Attention needed", value=False)
                if only_attention:
                    options = options[options["doc_id"].isin(sorted(list(attention["doc_ids"])))]

                sel_id = None
                if options.empty:
                    st.info("No documents match the filter.")
                else:
                    sel = st.selectbox("Document", options["label"].tolist())
                    sel_id = options.loc[options["label"] == sel, "doc_id"].iloc[0]

    with right:
        if sel_id:
            if case_mode == "AUTHOR":
                show_author_detail(sel_id)
            else:
                show_doc_detail(sel_id)
        else:
            st.info("Select a case on the left.")

# ---------------------------------------------------------------------
# TAB 2 — GLOBAL OVERVIEW (RECTANGULAR MAP + 👤)
# ---------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Global overview")
    st.write("Map is rectangular (no oval projection). Markers are **👤** icons per country (aggregated).")

    deck, missing = build_global_map_pydeck(occ_df)
    if deck is None:
        if pdk is None:
            st.info("pydeck is not available here. Install it: `pip install pydeck`")
        else:
            st.info("No map data available.")
    else:
        st.pydeck_chart(deck, use_container_width=True)

    if missing:
        st.warning("Add these countries to COUNTRY_CENTROIDS: " + ", ".join(missing))

# ---------------------------------------------------------------------
# TAB 3 — RULE INSPECTOR
# ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Rule Inspector (why cases land in Attention needed)")
    st.caption("These are *triage cues* for RoRS staff — not accusations.")

    if occ_df.empty:
        st.info("No occurrences loaded.")
    else:
        if flags_df.empty:
            st.success("No attention-needed rules fired under current settings.")
        else:
            summary = (
                flags_df.groupby(["rule_id", "rule_label"], as_index=False)
                .agg(
                    flagged_rows=("evidence_id", "count") if "evidence_id" in flags_df.columns else ("doc_id", "count"),
                    unique_authors=("case_id", "nunique") if "case_id" in flags_df.columns else ("doc_id", "nunique"),
                    unique_docs=("doc_id", "nunique") if "doc_id" in flags_df.columns else ("doc_id", "nunique"),
                )
                .sort_values("flagged_rows", ascending=False)
            )

            st.markdown("#### Rule summary")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown("#### Rule drilldown")
            rule_choices = summary.apply(lambda r: f"{r['rule_id']} — {r['rule_label']}", axis=1).tolist()
            choice = st.selectbox("Select a rule", rule_choices)
            chosen_id = choice.split("—")[0].strip()

            ex = flags_df[flags_df["rule_id"] == chosen_id].copy()
            ex = ex.rename(
                columns={
                    "case_id": "Author ID",
                    "name_string": "Author",
                    "doc_id": "Doc",
                    "conf/journal": "Venue",
                    "email_domain": "Email domain",
                    "evidence_id": "Evidence ID",
                    "page": "Page",
                    "doc_funding_agencies": "Funding agencies",
                }
            ).rename(columns={"country": "Country", "affiliation": "Affiliation"})

            cols = [
                "Author",
                "Author ID",
                "Doc",
                "Country",
                "Funding agencies",
                "Venue",
                "Affiliation",
                "Email domain",
                "Evidence ID",
                "Page",
            ]
            cols = [c for c in cols if c in ex.columns]
            st.dataframe(ex[cols].head(100), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 4 — COLLABORATIONS
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("### International collaboration patterns")
    st.caption("Counts authors who appear across multiple countries in occurrences.json (triage cue).")

    if occ_df.empty or "case_id" not in occ_df.columns or "country" not in occ_df.columns:
        st.info("No occurrences data loaded.")
    else:
        collabs = {}
        for cid in occ_df["case_id"].dropna().unique():
            cts = set(occ_df.loc[occ_df["case_id"] == cid, "country"].dropna().astype(str).str.strip())
            if len(cts) > 1:
                key = " ↔ ".join(sorted(cts))
                collabs[key] = collabs.get(key, 0) + 1

        if not collabs:
            st.info("No multi-country patterns found.")
        else:
            df = pd.DataFrame(collabs.items(), columns=["Country link", "Authors"])
            fig = px.bar(
                df.sort_values("Authors", ascending=False),
                x="Country link",
                y="Authors",
                title="Multi-country links (authors with >1 country)",
                template="plotly_white",
            )
            fig.update_layout(height=420, xaxis_tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 5 — PIPELINE HEALTH
# ---------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Pipeline / data health checks")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### Authors")
        st.write(f"Rows: **{len(authors_df)}**")

    with c2:
        st.markdown("#### Documents")
        st.write(f"Rows: **{len(docs_df)}**")
        # Funding coverage check
        if not docs_df.empty:
            has_funding_field = docs_df.apply(
                lambda r: isinstance(r.get("funding"), dict) or isinstance(r.get("funding_agencies"), list) or isinstance(r.get("funding_text"), str),
                axis=1
            )
            st.write(f"Docs with funding info: **{int(has_funding_field.sum())}/{len(docs_df)}**")

    with c3:
        st.markdown("#### Occurrences")
        st.write(f"Rows: **{len(occ_df)}**")
        if not occ_df.empty and "evidence_id" in occ_df.columns:
            st.write(f"Unique evidence IDs: **{occ_df['evidence_id'].nunique()}**")

    with c4:
        st.markdown("#### Attention rules")
        st.write(f"Flag rows: **{len(flags_df)}**")
        st.write(f"Unique cases: **{attention['count_unique_cases']}**")

    st.markdown("---")
    st.markdown("#### What to add next (to match your proposal better)")
    st.write(
        "- **Funding extraction**: store `funding.agencies` and `funding.has_us_federal_funding` per document.\n"
        "- **Metadata mismatch**: add `email_domain_header` and `email_domain_footer` in occurrences.json (or evidence.json) to enable M1.\n"
        "- **Footnote/header context**: add evidence items with `field_type` like `email_header`, `email_footer`, `funding_ack` to keep reports grounded."
    )

st.markdown("---")
st.caption("Run: `streamlit run app.py` • Data expected in ./data/*.json")
