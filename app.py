# app.py
# SENTRY — RoRS Research Security Operations Dashboard (Toolkit)
#
# Tabs:
#   - Author Profiles (author-only: identity + activity + publications)
#   - Anomalies (queue/KPIs)
#   - Reports (report-only: narratives + evidence grounding)
#   - Global Overview (country distribution across dataset)
#   - Pipeline Health (link checks)
#
# Status model:
#   - NORMAL
#   - ATTENTION
#
# Triage cues (rule-based review triggers):
#   - F1: US federal funding present AND doc has NO US authors (author_countries excludes US)
#   - M2: Metadata mismatch — same author + same affiliation shows multiple email domains across docs
#
# Data files expected:
#   data/authors.json
#   data/documents.json
#   data/occurrences.json
#   data/evidence.json (optional)

import json
from pathlib import Path
from typing import Any

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
st.set_page_config(page_title="SENTRY: Research Security Toolkit", layout="wide")

DATA_DIR = Path("data")
AUTHORS_PATH = DATA_DIR / "authors.json"
DOCS_PATH = DATA_DIR / "documents.json"
OCC_PATH = DATA_DIR / "occurrences.json"
EVIDENCE_PATH = DATA_DIR / "evidence.json"  # optional

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
    "IRELAND": (53.1424, -7.6921),
    "INDIA": (20.5937, 78.9629),
    "AUSTRALIA": (-25.2744, 133.7751),
    "NEW ZEALAND": (-40.9006, 174.8860),
    "ANTARCTICA": (-82.8628, 135.0000),
}

US_COUNTRY_ALIASES = {"USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"}

CONTINENT_COLORS = {
    "Asia": [255, 60, 60],
    "Europe": [40, 90, 200],
    "North America": [80, 200, 140],
    "South America": [255, 170, 0],
    "Africa": [170, 90, 255],
    "Oceania": [120, 120, 120],
    "Antarctica": [200, 200, 200],
    "Unknown": [160, 160, 160],
}

COUNTRY_TO_CONTINENT = {
    "USA": "North America",
    "US": "North America",
    "UNITED STATES": "North America",
    "UNITED STATES OF AMERICA": "North America",
    "CANADA": "North America",
    "MEXICO": "North America",
    "UK": "Europe",
    "UNITED KINGDOM": "Europe",
    "GERMANY": "Europe",
    "FRANCE": "Europe",
    "ITALY": "Europe",
    "SPAIN": "Europe",
    "IRELAND": "Europe",
    "SINGAPORE": "Asia",
    "JAPAN": "Asia",
    "CHINA": "Asia",
    "INDIA": "Asia",
    "NIGERIA": "Africa",
    "SOUTH AFRICA": "Africa",
    "AUSTRALIA": "Oceania",
    "NEW ZEALAND": "Oceania",
    "ANTARCTICA": "Antarctica",
}


# ---------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
body, .stApp {background-color: #0e1117; color: #e0e0e0;}
h1,h2,h3,h4 {color:#E7EAF0;}
.small {font-size: 0.9rem; opacity: 0.9;}

.badge {display:inline-block; padding:0.15rem 0.6rem; border-radius:999px; background:#22293a; border:1px solid #2d3752; font-size:0.8rem;}
.badge-attn {background:#2b2a14; border-color:#5f5b1d; color:#ffe082;}
.badge-ok {background:#1f2f23; border-color:#2d4a35; color:#b9f6ca;}
.badge-muted {background:#1b1f2a; border-color:#2d3752; color:#cfd8dc;}

hr {border-color:#232838;}

.dir-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.4rem;}
.dir-sub {opacity: .75; font-size: .95rem; margin-bottom: .6rem;}
.alpha-h {font-size: 0.9rem; letter-spacing: .12em; opacity: .7; margin-top: .9rem; margin-bottom:.4rem;}
.smallbtn div.stButton > button {font-size: 0.88rem; padding: 0.55rem 0.8rem; width:100%;}
.kv {display:flex; gap:10px; flex-wrap: wrap;}
.kv > div {background:#141926; border:1px solid #232838; padding:10px 12px; border-radius:12px; min-width:180px;}
.kv .k {opacity:.7; font-size:.85rem;}
.kv .v {font-size:1.25rem; font-weight:800; margin-top:2px;}
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
evidence = load_json(EVIDENCE_PATH) or []

authors_df = pd.DataFrame(authors)
docs_df = pd.DataFrame(documents)
occ_df = pd.DataFrame(occurrences)
evidence_df = pd.DataFrame(evidence) if isinstance(evidence, list) else pd.DataFrame()

for df in (authors_df, docs_df):
    for col in ["A1", "A2", "A_final"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

if not occ_df.empty:
    if "conf/journal" not in occ_df.columns and "conf_journal" in occ_df.columns:
        occ_df["conf/journal"] = occ_df["conf_journal"]

doc_by_id = {d.get("doc_id"): d for d in documents if isinstance(d, dict) and d.get("doc_id")}
author_by_id = {a.get("case_id"): a for a in authors if isinstance(a, dict) and a.get("case_id")}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_list(x):
    return x if isinstance(x, list) else []

def is_us_country(country: str) -> bool:
    c = (country or "").strip().upper()
    return c in US_COUNTRY_ALIASES

def badge_status(s: str) -> str:
    s = (s or "").upper().strip()
    if s == "FLAGGED":
        return "<span class='badge badge-attn'>FLAGGED</span>"
    if s == "ATTENTION":
        return "<span class='badge badge-attn'>ATTENTION</span>"
    if s == "NORMAL":
        return "<span class='badge badge-ok'>NORMAL</span>"
    return f"<span class='badge badge-muted'>{s or 'UNKNOWN'}</span>"

def status_from_score(a_final: float, attn_thr: float = 0.60, flag_thr: float = 0.80) -> str:
    v = float(a_final or 0.0)
    if v >= flag_thr:
        return "FLAGGED"
    if v >= attn_thr:
        return "ATTENTION"
    return "NORMAL"


def distribution_spark(values, bins=14):
    s = pd.Series(values).dropna()
    if s.empty:
        return []
    b = pd.cut(s, bins=bins, labels=False, duplicates="drop")
    bc = pd.Series(b).value_counts().sort_index()
    full = [int(bc.get(i, 0)) for i in range(int(bc.index.max()) + 1)]
    return full if len(full) >= 2 else []

def kpi_card(title: str, value: str, spark_y=None, delta_text: str = "", accent="#8b93ff"):
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
            fig = go.Figure(go.Scatter(
                x=list(range(len(spark_y))),
                y=spark_y,
                mode="lines",
                line=dict(width=3, color=accent),
                hoverinfo="skip",
            ))
            fig.update_layout(
                height=90,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def score_single(row: dict):
    v = row.get("A_final")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(v or 0),
                                 gauge={"axis": {"range": [0, 1]}}))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10), title="Anomaly Score", template="plotly_white")
    return fig

def metrics_bar(metrics: dict, title="Metric Breakdown"):
    metrics = metrics or {}
    num = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not num:
        return None
    df = pd.DataFrame({"Metric": list(num.keys()), "Value": list(num.values())})
    fig = px.bar(df, x="Metric", y="Value", title=title, height=260, template="plotly_white")
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    return fig

def compute_author_metrics_from_occ(case_id: str) -> dict[str, Any]:
    if occ_df.empty or "case_id" not in occ_df.columns:
        return {}
    sub = occ_df[occ_df["case_id"] == case_id].copy()
    if sub.empty:
        return {}

    def nunique_nonempty(series: pd.Series) -> int:
        s = series.fillna("").astype(str).str.strip()
        s = s[s != ""]
        return int(s.nunique())

    num_docs = int(sub["doc_id"].nunique()) if "doc_id" in sub.columns else 0
    num_distinct_affiliations = nunique_nonempty(sub["affiliation"]) if "affiliation" in sub.columns else 0
    num_distinct_countries = nunique_nonempty(sub["country"]) if "country" in sub.columns else 0
    num_email_domains = nunique_nonempty(sub["email_domain"]) if "email_domain" in sub.columns else 0

    return {
        "num_docs": num_docs,
        "num_distinct_affiliations": num_distinct_affiliations,
        "num_distinct_countries": num_distinct_countries,
        "num_email_domains": num_email_domains,
    }

def list_profile_facets(case_id: str):
    """
    Author-only facets (no report): countries, affiliations, email domains, venues.
    """
    if occ_df.empty or "case_id" not in occ_df.columns:
        return {}, {}, {}, {}
    sub = occ_df[occ_df["case_id"] == case_id].copy()
    if sub.empty:
        return {}, {}, {}, {}

    def top_counts(col: str, k=10):
        if col not in sub.columns:
            return {}
        s = sub[col].fillna("").astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            return {}
        return s.value_counts().head(k).to_dict()

    return (
        top_counts("country", 10),
        top_counts("affiliation", 12),
        top_counts("email_domain", 10),
        top_counts("conf/journal", 10),
    )

def _doc_us_funding_flags(doc_obj: dict, us_agencies_set: set[str]) -> tuple[bool, list[str]]:
    if not isinstance(doc_obj, dict):
        return (False, [])
    funding = doc_obj.get("funding")
    agencies_norm = []
    has_flag = False

    if isinstance(funding, dict):
        agencies = funding.get("agencies", [])
        agencies = agencies if isinstance(agencies, list) else []
        agencies_norm = [str(a).strip() for a in agencies if str(a).strip()]
        has_flag = bool(funding.get("has_us_federal_funding", False))
        if not has_flag and agencies_norm:
            has_flag = any(a.strip().upper() in us_agencies_set for a in agencies_norm)
    else:
        agencies = doc_obj.get("funding_agencies")
        if isinstance(agencies, list):
            agencies_norm = [str(a).strip() for a in agencies if str(a).strip()]
            has_flag = any(a.strip().upper() in us_agencies_set for a in agencies_norm)

    if not has_flag:
        txt = doc_obj.get("funding_text")
        if isinstance(txt, str) and txt.strip():
            txt_up = txt.upper()
            hit = [a for a in us_agencies_set if a in txt_up]
            if hit:
                has_flag = True
                agencies_norm = agencies_norm or sorted(hit)

    return (has_flag, agencies_norm)

def _doc_author_countries(doc_obj: dict) -> list[str]:
    if not isinstance(doc_obj, dict):
        return []
    ac = doc_obj.get("author_countries", [])
    if isinstance(ac, list):
        return [str(x).strip() for x in ac if str(x).strip()]
    return []

def compute_triage_rules(
    occ_df: pd.DataFrame,
    docs: list[dict],
    us_agencies_set: set[str],
    require_same_affil_for_email_mismatch: bool = True,
) -> pd.DataFrame:
    rows = []

    # F1 (document-level)
    for d in docs:
        doc_id = d.get("doc_id")
        has_us, agencies = _doc_us_funding_flags(d, us_agencies_set)
        if not has_us:
            continue
        countries = [c.strip() for c in _doc_author_countries(d)]
        has_us_author = any(is_us_country(c) for c in countries)
        if countries and (not has_us_author):
            rows.append({
                "rule_id": "F1",
                "rule_label": "US federal funding present, but extracted author countries include no US (review cue)",
                "case_kind": "DOCUMENT",
                "doc_id": doc_id,
                "doc_title": d.get("title", ""),
                "doc_author_countries": countries,
                "doc_funding_agencies": agencies,
                "case_id": None,
                "name_string": None,
                "affiliation": None,
                "country": None,
                "email_domain": None,
                "conf/journal": None,
                "evidence_id": None,
                "page": None,
            })

    # M2 (author-level)
    if not occ_df.empty and "case_id" in occ_df.columns:
        tmp = occ_df.copy()
        tmp["case_id"] = tmp["case_id"].fillna("").astype(str)
        tmp["name_string"] = tmp.get("name_string", "").fillna("").astype(str)
        tmp["email_domain"] = tmp.get("email_domain", "").fillna("").astype(str).str.strip()
        tmp["affiliation"] = tmp.get("affiliation", "").fillna("").astype(str).str.strip()
        tmp["doc_id"] = tmp.get("doc_id", "").fillna("").astype(str)
        tmp["country"] = tmp.get("country", "").fillna("").astype(str).str.strip()
        if "conf/journal" not in tmp.columns:
            tmp["conf/journal"] = ""
        tmp["conf/journal"] = tmp["conf/journal"].fillna("").astype(str)
        tmp["evidence_id"] = tmp.get("evidence_id", None)
        tmp["page"] = tmp.get("page", None)

        tmp = tmp[tmp["case_id"].str.len() > 0].copy()

        if require_same_affil_for_email_mismatch:
            grp = tmp.groupby(["case_id", "affiliation"], dropna=False).agg(
                n_docs=("doc_id", "nunique"),
                n_email_domains=("email_domain", lambda s: len({x for x in s if x})),
            ).reset_index()
            hit = grp[(grp["n_docs"] >= 2) & (grp["n_email_domains"] >= 2)].copy()
            for _, r in hit.iterrows():
                cid = r["case_id"]
                aff = r["affiliation"]
                sub = tmp[(tmp["case_id"] == cid) & (tmp["affiliation"] == aff)].copy()
                for item in sub.head(4).to_dict("records"):
                    rows.append({
                        "rule_id": "M2",
                        "rule_label": "Metadata mismatch: same author + same affiliation shows multiple email domains across documents",
                        "case_kind": "AUTHOR",
                        "case_id": cid,
                        "name_string": item.get("name_string"),
                        "doc_id": item.get("doc_id"),
                        "country": item.get("country"),
                        "affiliation": item.get("affiliation"),
                        "email_domain": item.get("email_domain"),
                        "conf/journal": item.get("conf/journal"),
                        "doc_funding_agencies": None,
                        "doc_author_countries": None,
                        "doc_title": None,
                        "evidence_id": item.get("evidence_id"),
                        "page": item.get("page"),
                    })
        else:
            grp = tmp.groupby("case_id").agg(
                n_docs=("doc_id", "nunique"),
                n_email_domains=("email_domain", lambda s: len({x for x in s if x})),
            ).reset_index()
            hit = grp[(grp["n_docs"] >= 2) & (grp["n_email_domains"] >= 2)].copy()
            for _, r in hit.iterrows():
                cid = r["case_id"]
                sub = tmp[tmp["case_id"] == cid].copy()
                for item in sub.head(4).to_dict("records"):
                    rows.append({
                        "rule_id": "M2",
                        "rule_label": "Metadata mismatch: same author cluster shows multiple email domains across documents",
                        "case_kind": "AUTHOR",
                        "case_id": cid,
                        "name_string": item.get("name_string"),
                        "doc_id": item.get("doc_id"),
                        "country": item.get("country"),
                        "affiliation": item.get("affiliation"),
                        "email_domain": item.get("email_domain"),
                        "conf/journal": item.get("conf/journal"),
                        "doc_funding_agencies": None,
                        "doc_author_countries": None,
                        "doc_title": None,
                        "evidence_id": item.get("evidence_id"),
                        "page": item.get("page"),
                    })

    return pd.DataFrame(rows)

def compute_attention_sets(flags_df: pd.DataFrame) -> dict:
    if flags_df.empty:
        return {"author_ids": set(), "doc_ids": set(), "count_unique_cases": 0}
    author_ids = set(flags_df.loc[flags_df["case_kind"] == "AUTHOR", "case_id"].dropna().unique())
    doc_ids = set(flags_df.loc[flags_df["case_kind"] == "DOCUMENT", "doc_id"].dropna().unique())
    return {"author_ids": author_ids, "doc_ids": doc_ids, "count_unique_cases": len(author_ids) + len(doc_ids)}

def apply_attention_status(authors_df: pd.DataFrame, docs_df: pd.DataFrame, attention_sets: dict):
    a = authors_df.copy()
    d = docs_df.copy()

    if not a.empty:
        a["status"] = a.get("status", "NORMAL").fillna("NORMAL").astype(str).str.upper()
        a.loc[a["case_id"].isin(attention_sets["author_ids"]), "status"] = "ATTENTION"
        a.loc[~a["status"].isin(["NORMAL", "ATTENTION"]), "status"] = "NORMAL"

    if not d.empty:
        d["status"] = d.get("status", "NORMAL").fillna("NORMAL").astype(str).str.upper()
        d.loc[d["doc_id"].isin(attention_sets["doc_ids"]), "status"] = "ATTENTION"
        d.loc[~d["status"].isin(["NORMAL", "ATTENTION"]), "status"] = "NORMAL"

    return a, d

def build_review_queue(a_df: pd.DataFrame, d_df: pd.DataFrame):
    a = a_df.copy()
    d = d_df.copy()

    if not a.empty:
        a["case_kind"] = "AUTHOR"
        a["case_key"] = a.get("case_id")
        a["title_or_name"] = a.get("name_string")

    if not d.empty:
        d["case_kind"] = "DOCUMENT"
        d["case_key"] = d.get("doc_id")
        d["title_or_name"] = d.get("title")

    keep = ["case_kind", "case_key", "title_or_name", "status", "A_final", "top_reason"]
    a = a[[c for c in keep if c in a.columns]] if not a.empty else pd.DataFrame(columns=keep)
    d = d[[c for c in keep if c in d.columns]] if not d.empty else pd.DataFrame(columns=keep)

    q = pd.concat([a, d], ignore_index=True)
    order = {"FLAGGED": 0, "ATTENTION": 1, "NORMAL": 2}
    q["status_sort"] = q["status"].fillna("NORMAL").astype(str).str.upper().map(lambda s: order.get(s, 3))
    q["score_sort"] = pd.to_numeric(q.get("A_final", 0), errors="coerce").fillna(-1)

    q = q.sort_values(["status_sort", "score_sort"], ascending=[True, False], na_position="last")
    q = q.drop(columns=["status_sort", "score_sort"], errors="ignore")
    return q

def compute_dashboard_metrics(a_df: pd.DataFrame, d_df: pd.DataFrame, flags_df: pd.DataFrame):
    total_assets = int(len(a_df) + len(d_df))
    attn_assets = int((a_df.get("status", "").astype(str).str.upper() == "ATTENTION").sum()
                      + (d_df.get("status", "").astype(str).str.upper() == "ATTENTION").sum())
    high_score = int((pd.to_numeric(a_df.get("A_final", 0), errors="coerce").fillna(0) >= 0.80).sum()
                     + (pd.to_numeric(d_df.get("A_final", 0), errors="coerce").fillna(0) >= 0.80).sum())
    rules_fired_rows = int(len(flags_df)) if flags_df is not None and not flags_df.empty else 0
    pct_attn = round((100 * attn_assets / max(total_assets, 1)), 1)
    return total_assets, attn_assets, high_score, rules_fired_rows, pct_attn


# ---------------------------------------------------------------------
# REPORT / EVIDENCE (REPORTS TAB ONLY)
# ---------------------------------------------------------------------
def show_case_report(report: dict):
    report = report or {}
    st.markdown("### Evidence-grounded review report")
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

def show_evidence_table(evidence_ids: list[str]):
    if evidence_df.empty:
        st.info("No evidence.json loaded (optional).")
        return
    ev_ids = [x for x in (evidence_ids or []) if isinstance(x, str) and x.strip()]
    if not ev_ids:
        st.info("No evidence IDs listed for this case.")
        return
    sub = evidence_df[evidence_df.get("evidence_id").isin(ev_ids)].copy()
    if sub.empty:
        st.info("Evidence IDs not found in evidence.json.")
        return
    cols = ["evidence_id", "doc_id", "page", "field_type", "snippet"]
    cols = [c for c in cols if c in sub.columns]
    st.dataframe(sub[cols], use_container_width=True, hide_index=True)

def show_occurrences_for_author(case_id: str, evidence_ids: list[str] | None = None):
    st.markdown("### Evidence panel (occurrences)")
    if occ_df.empty:
        st.info("No occurrences.json loaded.")
        return

    sub = occ_df[occ_df.get("case_id") == case_id].copy()
    if evidence_ids and "evidence_id" in sub.columns:
        sub = sub[sub["evidence_id"].isin(evidence_ids)].copy()

    if sub.empty:
        st.info("No matching occurrences for this author.")
        return

    cols = ["evidence_id", "doc_id", "page", "field_type", "affiliation", "country", "email_domain", "conf/journal"]
    cols = [c for c in cols if c in sub.columns]
    show = sub[cols].rename(columns={
        "evidence_id": "Evidence ID",
        "doc_id": "Doc",
        "page": "Page",
        "field_type": "Field",
        "conf/journal": "Venue",
        "email_domain": "Email domain",
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

def show_occurrences_for_doc(doc_id: str, evidence_ids: list[str] | None = None):
    st.markdown("### Evidence panel (occurrences)")
    if occ_df.empty:
        st.info("No occurrences.json loaded.")
        return
    sub = occ_df[occ_df.get("doc_id") == doc_id].copy()
    if evidence_ids and "evidence_id" in sub.columns:
        sub = sub[sub["evidence_id"].isin(evidence_ids)].copy()
    if sub.empty:
        st.info("No matching occurrences for this document.")
        return
    cols = ["case_id", "name_string", "affiliation", "country", "email_domain", "conf/journal", "evidence_id", "page"]
    cols = [c for c in cols if c in sub.columns]
    st.dataframe(sub[cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# GLOBAL MAP (country distribution across dataset)
# ---------------------------------------------------------------------
def build_country_distribution_agg(occ_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregates the entire dataset by country:
      - rows = number of occurrences (author-in-doc instances)
      - author_count = unique authors in that country (by case_id)
    """
    if occ_df.empty:
        return pd.DataFrame(), []

    tmp = occ_df.copy()
    tmp["country_norm"] = tmp.get("country", "").fillna("").astype(str).str.strip()
    tmp = tmp[tmp["country_norm"] != ""].copy()
    if tmp.empty:
        return pd.DataFrame(), []

    tmp["country_key"] = tmp["country_norm"].astype(str).str.upper().str.strip()
    tmp["lat"] = tmp["country_key"].map(lambda c: COUNTRY_CENTROIDS.get(c, (None, None))[0])
    tmp["lon"] = tmp["country_key"].map(lambda c: COUNTRY_CENTROIDS.get(c, (None, None))[1])

    missing = sorted(tmp.loc[tmp["lat"].isna(), "country_norm"].unique().tolist())
    tmp = tmp.dropna(subset=["lat", "lon"]).copy()
    if tmp.empty:
        return pd.DataFrame(), missing

    agg = (
        tmp.groupby(["country_norm", "lat", "lon"], as_index=False)
        .agg(
            rows=("country_norm", "count"),
            author_count=("case_id", "nunique") if "case_id" in tmp.columns else ("country_norm", "count"),
        )
    )

    agg["country_key"] = agg["country_norm"].astype(str).str.upper().str.strip()
    agg["continent"] = agg["country_key"].map(lambda c: COUNTRY_TO_CONTINENT.get(c, "Unknown"))
    agg["color"] = agg["continent"].map(lambda k: CONTINENT_COLORS.get(k, CONTINENT_COLORS["Unknown"]))

    # top names for hover
    if "name_string" in tmp.columns:
        names_per = (
            tmp.groupby("country_norm")["name_string"]
            .apply(lambda s: sorted(set(s.dropna().astype(str)))[:12])
            .to_dict()
        )
    else:
        names_per = {}

    agg["hover"] = agg.apply(
        lambda r: f"{r['country_norm']} • {r['continent']}\n"
                  f"Occurrences: {int(r['rows'])} • Unique authors: {int(r['author_count'])}\n"
                  + "\n".join([f"👤 {n}" for n in names_per.get(r["country_norm"], [])]),
        axis=1
    )

    # size uses occurrence rows (dataset-wide)
    agg["size"] = (agg["rows"] ** 0.5 * 14).clip(16, 52)
    return agg, missing


def build_global_map_pydeck_country_distribution(occ_df: pd.DataFrame):
    if pdk is None:
        return None, ["pydeck not available (install pydeck)."], pd.DataFrame()

    agg, missing = build_country_distribution_agg(occ_df)
    if agg.empty:
        return None, missing, agg

    MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    # ✅ IMPORTANT: use your continent color column in the scatter layer
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=agg,
        get_position="[lon, lat]",
        get_radius="size * 9000",
        get_fill_color="color",
        get_line_color="color",
        pickable=True,
        opacity=0.65,
        stroked=True,
        filled=True,
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=0.8)

    deck = pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=view_state,
        layers=[scatter],
        tooltip={"text": "{hover}"},
    )

    return deck, missing, agg


def render_legend(agg: pd.DataFrame | None = None):
    st.markdown("### Continents")

    base_items = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania", "Antarctica"]

    show_unknown = False
    if agg is not None and not agg.empty and "continent" in agg.columns:
        show_unknown = (agg["continent"] == "Unknown").any()

    items = base_items + (["Unknown"] if show_unknown else [])

    def chip(label: str) -> str:
        rgb = CONTINENT_COLORS.get(label, [180, 180, 180])
        return (
            "<div style='display:flex;align-items:center;gap:10px;"
            "padding:6px 10px;border-radius:10px;border:1px solid #232838;"
            "background:#141926;'>"
            f"<div style='width:16px;height:16px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"
            "border-radius:4px;'></div>"
            f"<div style='font-size:16px;'>{label}</div>"
            "</div>"
        )

    html = (
        "<div style='display:flex;flex-wrap:wrap;gap:18px;align-items:center;'>"
        + "".join([chip(x) for x in items])
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    kind_filter = st.multiselect("Case types", ["AUTHOR", "DOCUMENT"], default=["AUTHOR", "DOCUMENT"])
    status_filter = st.multiselect("Status", ["FLAGGED", "ATTENTION", "NORMAL"], default=["FLAGGED", "ATTENTION", "NORMAL"])
    min_score = st.slider("Min anomaly score (A_final)", 0.0, 1.0, 0.0, 0.01)
    st.markdown("---")
    st.markdown("### Score thresholds")
    attn_thr = st.slider("ATTENTION threshold", 0.0, 1.0, 0.60, 0.01)
    flag_thr = st.slider("FLAGGED threshold", 0.0, 1.0, 0.80, 0.01)


    st.markdown("---")
    st.markdown("### Attention rules (review cues)")
    st.caption("Rule cues support triage and do not assert wrongdoing.")

    us_agencies_text = st.text_area(
        "US federal agencies (comma-separated)",
        value="NSF, DOE, DARPA, NIH, ONR, AFRL, ARPA-E",
        height=80,
    )
    us_agencies_set = {x.strip().upper() for x in us_agencies_text.split(",") if x.strip()}

    require_same_affil_for_email = st.checkbox(
        "M2: require same affiliation for email mismatch",
        value=True
    )


# ---------------------------------------------------------------------
# COMPUTE TRIAGE + APPLY ATTENTION
# ---------------------------------------------------------------------
flags_df = compute_triage_rules(
    occ_df=occ_df,
    docs=documents,
    us_agencies_set=us_agencies_set,
    require_same_affil_for_email_mismatch=require_same_affil_for_email,
)
attention_sets = compute_attention_sets(flags_df)

# --- status from score (FLAGGED/ATTENTION/NORMAL) ---
authors_df2 = authors_df.copy()
docs_df2 = docs_df.copy()

if not authors_df2.empty:
    authors_df2["A_final"] = pd.to_numeric(authors_df2.get("A_final", 0), errors="coerce").fillna(0)
    authors_df2["status"] = authors_df2["A_final"].apply(lambda v: status_from_score(v, attn_thr, flag_thr))

if not docs_df2.empty:
    docs_df2["A_final"] = pd.to_numeric(docs_df2.get("A_final", 0), errors="coerce").fillna(0)
    docs_df2["status"] = docs_df2["A_final"].apply(lambda v: status_from_score(v, attn_thr, flag_thr))

# --- rule-based attention can only "raise" to ATTENTION (never downgrade FLAGGED) ---
if not authors_df2.empty:
    mask = authors_df2["case_id"].isin(attention_sets["author_ids"]) & (authors_df2["status"] != "FLAGGED")
    authors_df2.loc[mask, "status"] = "ATTENTION"

if not docs_df2.empty:
    mask = docs_df2["doc_id"].isin(attention_sets["doc_ids"]) & (docs_df2["status"] != "FLAGGED")
    docs_df2.loc[mask, "status"] = "ATTENTION"

authors = authors_df2.to_dict("records")
documents = docs_df2.to_dict("records")
doc_by_id = {d.get("doc_id"): d for d in documents if isinstance(d, dict) and d.get("doc_id")}
author_by_id = {a.get("case_id"): a for a in authors if isinstance(a, dict) and a.get("case_id")}


# ---------------------------------------------------------------------
# HEADER + TABS
# ---------------------------------------------------------------------
st.header("SENTRY: Research Security Anomaly Detection Toolkit")
st.caption("Triage states (NORMAL / ATTENTION / FLAGGED) are driven by anomaly scores and rule-based signals; reports provide evidence-grounded summaries for human review.")

tabs = st.tabs([
    "Author Profiles",
    "Anomalies",
    "Reports",
    "Global Overview",
    "Pipeline Health",
])


# ---------------------------------------------------------------------
# TAB 0 — AUTHOR PROFILES (AUTHOR-ONLY, NO REPORT TEXT)
# ---------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([0.32, 0.68], gap="large")

    with left:
        st.markdown("<div class='dir-title'>Author Directory</div>", unsafe_allow_html=True)
        st.markdown("<div class='dir-sub'>Search and select an author to review.</div>", unsafe_allow_html=True)

        query = st.text_input("Search author name", value="", placeholder="Type a name…")

        if authors_df2.empty or "case_id" not in authors_df2.columns:
            st.info("No authors loaded.")
            selected_author_id = None
        else:
            opts = authors_df2.copy()
            opts["name_string"] = opts.get("name_string", "").fillna("").astype(str)
            opts["status"] = opts.get("status", "NORMAL").fillna("NORMAL").astype(str).str.upper()
            opts["A_final"] = pd.to_numeric(opts.get("A_final", 0), errors="coerce").fillna(0)

            if query.strip():
                opts = opts[opts["name_string"].str.contains(query.strip(), case=False, na=False)].copy()

            # ATTENTION first, then score
            opts["status_sort"] = opts["status"].map(lambda s: 0 if s == "ATTENTION" else 1)
            opts = opts.sort_values(["status_sort", "A_final", "name_string"], ascending=[True, False, True], na_position="last")

            if "selected_author_id" not in st.session_state:
                st.session_state["selected_author_id"] = None

            opts["alpha"] = opts["name_string"].str[:1].str.upper()
            opts["alpha"] = opts["alpha"].where(opts["alpha"].str.match(r"[A-Z]"), "#")

            st.markdown("<div class='smallbtn'>", unsafe_allow_html=True)
            selected_author_id = st.session_state["selected_author_id"]

            for alpha in sorted(opts["alpha"].unique()):
                sub = opts[opts["alpha"] == alpha]
                st.markdown(f"<div class='alpha-h'>{alpha}</div>", unsafe_allow_html=True)
                for _, r in sub.iterrows():
                    label = r["name_string"] if r["name_string"] else r.get("case_id", "(unknown)")
                    badge = "🚩" if r["status"] == "FLAGGED" else ("⚠️" if r["status"] == "ATTENTION" else " ")
                    if st.button(f"{badge} {label}", key=f"author_btn_{r['case_id']}"):
                        st.session_state["selected_author_id"] = r["case_id"]
                        selected_author_id = r["case_id"]
            st.markdown("</div>", unsafe_allow_html=True)

            if selected_author_id is None and not opts.empty:
                st.session_state["selected_author_id"] = opts.iloc[0]["case_id"]
                selected_author_id = st.session_state["selected_author_id"]

    with right:
        if not selected_author_id:
            st.info("Select an author from the directory.")
        else:
            author = author_by_id.get(selected_author_id, {})
            st.markdown("## Author Detail Panel")
            st.markdown(f"### {author.get('name_string','(unknown)')}")

            st.markdown(f"Status: {badge_status(author.get('status'))}", unsafe_allow_html=True)

            if author.get("top_reason"):
                st.write(f"**Top reason:** {author.get('top_reason')}")

            cp = author.get("collaboration_profile")
            if cp:
                st.write(f"**Collaboration profile:** {cp}")

            # Triage cues (short list, author-only)
            cues = pd.DataFrame()
            if flags_df is not None and not flags_df.empty:
                cues = flags_df[(flags_df["case_kind"] == "AUTHOR") & (flags_df["case_id"] == selected_author_id)].copy()
            if not cues.empty:
                st.markdown("#### Triage cues fired")
                for rid, lbl in cues[["rule_id", "rule_label"]].drop_duplicates().itertuples(index=False):
                    st.write(f"• **{rid}** — {lbl}")

            sigs = safe_list(author.get("signals", []))
            if sigs:
                st.markdown("#### Signals")
                st.write(" • " + "\n • ".join(sigs))

            st.markdown("---")

            # Metrics (ensure present)
            metrics = author.get("metrics") if isinstance(author.get("metrics"), dict) else {}
            occ_metrics = compute_author_metrics_from_occ(selected_author_id)
            merged_metrics = {**occ_metrics, **metrics}

            # quick metric cards
            st.markdown("#### Metrics")
            st.markdown(
                f"""
                <div class="kv">
                  <div><div class="k">Documents</div><div class="v">{merged_metrics.get('num_docs',0)}</div></div>
                  <div><div class="k">Distinct affiliations</div><div class="v">{merged_metrics.get('num_distinct_affiliations',0)}</div></div>
                  <div><div class="k">Distinct countries</div><div class="v">{merged_metrics.get('num_distinct_countries',0)}</div></div>
                  <div><div class="k">Email domains</div><div class="v">{merged_metrics.get('num_email_domains',0)}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([0.55, 0.45], gap="large")
            with c1:
                if author.get("A_final") is not None:
                    st.plotly_chart(score_single(author), use_container_width=True)
                mb = metrics_bar(merged_metrics, title="Metric Breakdown")
                if mb:
                    st.plotly_chart(mb, use_container_width=True)

            with c2:
                countries, affils, domains, venues = list_profile_facets(selected_author_id)

                st.markdown("#### Countries (occurrences)")
                if countries:
                    st.dataframe(pd.DataFrame({"Country": list(countries.keys()), "Count": list(countries.values())}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No country occurrences for this author.")

                st.markdown("#### Affiliations")
                if affils:
                    st.dataframe(pd.DataFrame({"Affiliation": list(affils.keys()), "Count": list(affils.values())}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No affiliations for this author.")

                st.markdown("#### Email domains")
                if domains:
                    st.dataframe(pd.DataFrame({"Domain": list(domains.keys()), "Count": list(domains.values())}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No email domains for this author.")

            # Publications / research output (linked docs)
            st.markdown("---")
            st.markdown("#### Publications")
            doc_ids = safe_list(author.get("docs", []))
            if doc_ids and not docs_df2.empty and "doc_id" in docs_df2.columns:
                cols = [c for c in ["doc_id", "title", "year", "collaboration_type", "conf/journal", "A_final", "status"] if c in docs_df2.columns]
                pub = docs_df2[docs_df2["doc_id"].isin(doc_ids)].copy()
                # if conf/journal is not in documents, try to infer from occurrences
                if "conf/journal" not in pub.columns and (not occ_df.empty) and "conf/journal" in occ_df.columns:
                    ven = occ_df[occ_df["doc_id"].isin(doc_ids)].groupby("doc_id")["conf/journal"].agg(lambda s: ", ".join(sorted(set([x for x in s.dropna().astype(str) if x.strip()]))[:2]))
                    pub = pub.merge(ven.rename("conf/journal"), on="doc_id", how="left")
                    cols = [c for c in cols if c != "conf/journal"] + (["conf/journal"] if "conf/journal" in pub.columns else [])

                cols = [c for c in cols if c in pub.columns]
                pub = pub.sort_values(["year", "title"], ascending=[False, True], na_position="last")
                st.dataframe(pub[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No linked documents listed for this author.")


# ---------------------------------------------------------------------
# TAB 1 — ANOMALIES
# ---------------------------------------------------------------------
with tabs[1]:
    st.markdown("## Anomalies")
    st.caption("Queue prioritizes Flagged first, then anomaly score (A_final).")

    total_assets, attn_assets, high_score, rules_fired_rows, pct_attn = compute_dashboard_metrics(authors_df2, docs_df2, flags_df)

    spark_all = distribution_spark(pd.concat([
        authors_df2.get("A_final", pd.Series(dtype=float)),
        docs_df2.get("A_final", pd.Series(dtype=float))
    ], ignore_index=True))
    spark_auth = distribution_spark(authors_df2.get("A_final", pd.Series(dtype=float)))
    spark_docs = distribution_spark(docs_df2.get("A_final", pd.Series(dtype=float)))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total cases", str(total_assets), spark_y=spark_all, delta_text="(authors + documents)")
    with c2:
        kpi_card("High/Critical Alerts (≥ 0.80)", str(high_score), spark_y=spark_auth, delta_text="(score-based)")
    with c3:
        kpi_card("Entities needing attention", str(attn_assets), spark_y=spark_docs, delta_text="(rule-based)")
    with c4:
        kpi_card("ATTENTION rate", f"{pct_attn}%", spark_y=[total_assets, attn_assets, high_score, rules_fired_rows], delta_text="(overview)")

    st.markdown("---")
    queue = build_review_queue(authors_df2, docs_df2)

    if kind_filter:
        queue = queue[queue["case_kind"].isin(kind_filter)]

    if status_filter:
        qst = queue["status"].fillna("").astype(str).str.upper()
        masks = []
        if "FLAGGED" in status_filter:
            masks.append(qst == "FLAGGED")
        if "ATTENTION" in status_filter:
            masks.append(qst == "ATTENTION")
        if "NORMAL" in status_filter:
            masks.append(qst == "NORMAL")
        if masks:
            keep_mask = masks[0]
            for mm in masks[1:]:
                keep_mask = keep_mask | mm
            queue = queue[keep_mask]

    queue = queue[pd.to_numeric(queue.get("A_final", 0), errors="coerce").fillna(0) >= min_score]

    q = st.text_input("Search (name/title contains)", key="anomaly_search")
    if q:
        queue = queue[queue["title_or_name"].fillna("").str.contains(q, case=False, na=False)]

    show_cols = ["case_kind", "case_key", "title_or_name", "status", "A_final", "top_reason"]
    show = queue[show_cols].copy() if not queue.empty else pd.DataFrame(columns=show_cols)
    show = show.rename(columns={"A_final": "Anomaly Score"})
    st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# TAB 2 — REPORTS (REPORT-ONLY)
# ---------------------------------------------------------------------
with tabs[2]:
    st.markdown("## Reports")
    st.caption("LLM extracted Evidence Summary Reports")

    mode = st.radio("Report type", ["AUTHOR", "DOCUMENT"], horizontal=True)

    if mode == "AUTHOR":
        if authors_df2.empty or "case_id" not in authors_df2.columns:
            st.info("No authors loaded.")
        else:
            tmp = authors_df2.copy()
            tmp["status"] = tmp.get("status", "NORMAL").fillna("NORMAL").astype(str).str.upper()
            tmp["A_final"] = pd.to_numeric(tmp.get("A_final", 0), errors="coerce").fillna(0)
            tmp["has_report"] = tmp["report"].apply(lambda r: isinstance(r, dict) and bool(r.get("summary"))) if "report" in tmp.columns else False
            tmp = tmp[tmp["has_report"]].copy()

            if tmp.empty:
                st.info("No author reports found.")
            else:
                tmp = tmp.sort_values(["status", "A_final"], ascending=[True, False], na_position="last")
                labels = tmp.apply(
                    lambda r: f"{r.get('name_string','')} • {r.get('case_id','')} • {r.get('status','')} • Score {float(r.get('A_final',0)):.2f}",
                    axis=1
                ).tolist()

                sel = st.selectbox("Select author report", labels)
                cid = tmp.loc[tmp.apply(
                    lambda r: f"{r.get('name_string','')} • {r.get('case_id','')} • {r.get('status','')} • Score {float(r.get('A_final',0)):.2f}",
                    axis=1
                ) == sel, "case_id"].iloc[0]

                author = author_by_id.get(cid, {})
                st.markdown(f"### {author.get('name_string','(unknown)')}")
                st.markdown(f"Status: {badge_status(author.get('status'))}", unsafe_allow_html=True)

                report = author.get("report", {}) if isinstance(author.get("report"), dict) else {}
                show_case_report(report)

                ev_ids = safe_list(report.get("evidence_ids", []))
                show_occurrences_for_author(cid, evidence_ids=ev_ids if ev_ids else None)

                if not evidence_df.empty and ev_ids:
                    st.markdown("### Evidence snippets")
                    show_evidence_table(ev_ids)

    else:  # DOCUMENT
        if docs_df2.empty or "doc_id" not in docs_df2.columns:
            st.info("No documents loaded.")
        else:
            tmp = docs_df2.copy()
            tmp["status"] = tmp.get("status", "NORMAL").fillna("NORMAL").astype(str).str.upper()
            tmp["A_final"] = pd.to_numeric(tmp.get("A_final", 0), errors="coerce").fillna(0)
            tmp["has_report"] = tmp["report"].apply(lambda r: isinstance(r, dict) and bool(r.get("summary"))) if "report" in tmp.columns else False
            tmp = tmp[tmp["has_report"]].copy()

            if tmp.empty:
                st.info("No document reports found.")
            else:
                tmp = tmp.sort_values(["status", "A_final"], ascending=[True, False], na_position="last")
                labels = tmp.apply(
                    lambda r: f"{r.get('title','')} • {r.get('doc_id','')} • {r.get('status','')} • Score {float(r.get('A_final',0)):.2f}",
                    axis=1
                ).tolist()

                sel = st.selectbox("Select document report", labels)
                did = tmp.loc[tmp.apply(
                    lambda r: f"{r.get('title','')} • {r.get('doc_id','')} • {r.get('status','')} • Score {float(r.get('A_final',0)):.2f}",
                    axis=1
                ) == sel, "doc_id"].iloc[0]

                doc = doc_by_id.get(did, {})
                st.markdown(f"### {doc.get('title','(unknown)')}")
                st.markdown(f"<span class='badge'>{did}</span> &nbsp; Status: {badge_status(doc.get('status'))}", unsafe_allow_html=True)

                report = doc.get("report", {}) if isinstance(doc.get("report"), dict) else {}
                show_case_report(report)

                ev_ids = safe_list(report.get("evidence_ids", []))
                show_occurrences_for_doc(did, evidence_ids=ev_ids if ev_ids else None)

                if not evidence_df.empty and ev_ids:
                    st.markdown("### Evidence snippets")
                    show_evidence_table(ev_ids)


# ---------------------------------------------------------------------
# TAB 3 — GLOBAL OVERVIEW (country distribution across dataset)
# ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("## Global overview")
    st.caption("Map shows where authors are coming from (country distribution across occurrences in the dataset).")

    deck, missing, agg = build_global_map_pydeck_country_distribution(occ_df)
    if deck is None:
        if pdk is None:
            st.info("pydeck is not available. Add `pydeck` to requirements.txt.")
        else:
            st.info("No map data available.")
    else:
        st.pydeck_chart(deck, use_container_width=True)

    render_legend()

    if missing:
        st.warning("Add these countries to COUNTRY_CENTROIDS: " + ", ".join(missing))

    st.markdown("---")
    st.markdown("### Country distribution")
    if agg is None or agg.empty:
        st.info("No country data available.")
    else:
        # table + bar chart
        t = agg.sort_values("rows", ascending=False).copy()
        st.dataframe(t[["country_norm", "continent", "rows", "author_count"]].rename(columns={
            "country_norm": "Country",
            "rows": "Occurrences",
            "author_count": "Unique authors",
            "continent": "Continent",
        }), use_container_width=True, hide_index=True)

        fig = px.bar(
            t.head(25),
            x="country_norm",
            y="rows",
            title="Top countries by occurrences",
            template="plotly_white",
            height=360
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        fig.update_xaxes(title="Country")
        fig.update_yaxes(title="Occurrences")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 4 — PIPELINE HEALTH
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("## Pipeline health")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### Authors")
        st.write(f"Rows: **{len(authors_df2)}**")
        if not authors_df2.empty and "status" in authors_df2.columns:
            st.write("ATTENTION: **{}**".format(int((authors_df2["status"].astype(str).str.upper() == "ATTENTION").sum())))

    with c2:
        st.markdown("#### Documents")
        st.write(f"Rows: **{len(docs_df2)}**")
        if not docs_df2.empty and "status" in docs_df2.columns:
            st.write("ATTENTION: **{}**".format(int((docs_df2["status"].astype(str).str.upper() == "ATTENTION").sum())))

    with c3:
        st.markdown("#### Occurrences")
        st.write(f"Rows: **{len(occ_df)}**")
        if not occ_df.empty and "evidence_id" in occ_df.columns:
            st.write(f"Unique evidence IDs: **{occ_df['evidence_id'].nunique()}**")

    with c4:
        st.markdown("#### Rule outputs")
        st.write(f"Rows (rule matches): **{len(flags_df)}**")
        st.write(f"Unique cases (authors+docs): **{attention_sets['count_unique_cases']}**")

    st.markdown("---")
    st.markdown("### Link integrity checks")
    issues = []

    # authors docs list -> doc ids exist
    if not authors_df2.empty and "docs" in authors_df2.columns and "doc_id" in docs_df2.columns:
        doc_ids = set(docs_df2["doc_id"].dropna().astype(str))
        for _, r in authors_df2.iterrows():
            dids = safe_list(r.get("docs", []))
            for d in dids:
                if str(d) not in doc_ids:
                    issues.append(f"Author {r.get('case_id')} references missing doc_id: {d}")

    # occurrences doc_id exist
    if not occ_df.empty and "doc_id" in occ_df.columns:
        doc_ids = set(docs_df2["doc_id"].dropna().astype(str)) if "doc_id" in docs_df2.columns else set()
        miss = sorted(set(occ_df["doc_id"].dropna().astype(str)) - doc_ids)
        if miss:
            issues.append("occurrences.json includes doc_id(s) not present in documents.json: " + ", ".join(miss))

    # occurrences case_id exist
    if not occ_df.empty and "case_id" in occ_df.columns:
        author_ids = set(authors_df2["case_id"].dropna().astype(str)) if "case_id" in authors_df2.columns else set()
        miss = sorted(set(occ_df["case_id"].dropna().astype(str)) - author_ids)
        if miss:
            issues.append("occurrences.json includes case_id(s) not present in authors.json: " + ", ".join(miss))

    if not issues:
        st.success("No obvious linkage issues detected.")
    else:
        for x in issues[:60]:
            st.warning(x)
        if len(issues) > 60:
            st.info(f"…and {len(issues) - 60} more.")

st.markdown("---")
st.caption("Run: `streamlit run app.py` • Data expected in ./data/*.json")
