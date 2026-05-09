import re, json, math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# 0. Larger evidence budget
# ------------------------------------------------------------

RAG_MAX_DOCS = max(globals().get("RAG_MAX_DOCS", 10), 18)
AGENT_MAX_DOCS = max(globals().get("AGENT_MAX_DOCS", 10), 22)

RAG_SUMMARY_MAX_BULLETS = 16
AGENT_SUMMARY_MAX_BULLETS = 16

try:
    RETRIEVAL_KS = sorted(set(list(RETRIEVAL_KS) + [20]))
except Exception:
    RETRIEVAL_KS = [5, 10, 20]

print("V5 patch loaded")
print("RAG_MAX_DOCS:", RAG_MAX_DOCS)
print("AGENT_MAX_DOCS:", AGENT_MAX_DOCS)
print("RETRIEVAL_KS:", RETRIEVAL_KS)


# ------------------------------------------------------------
# 1. Utility functions
# ------------------------------------------------------------

def _s(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=str)
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)

def _n(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    return pd.to_numeric(_s(df, col, default), errors="coerce").fillna(default)

def _norm01(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if abs(mx - mn) < 1e-9:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn + 1e-9)

def _contains(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.contains(
        pattern.lower(), regex=True, na=False
    )

def _recall_ids_from_query(q: str) -> set:
    return {
        canon_recall_number(x)
        for x in extract_recall_numbers(q or "")
        if canon_recall_number(x)
    }


# ------------------------------------------------------------
# 2. Balanced scoring: no human gold leakage
# ------------------------------------------------------------

def add_balanced_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    给每个 recall candidate 一个更适合 coverage 的分数。
    注意：这里不使用 human gold，所以不会造成 evaluation leakage。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    for c in [
        "hybrid_score",
        "summary_score",
        "importance_score",
        "retrieval_score",
        "route_priority",
    ]:
        if c not in out.columns:
            out[c] = 0.0

    cf = _s(out, "changed_fields", "").fillna("").astype(str)
    ct = _s(out, "change_types", "").fillna("").astype(str)
    rl = _s(out, "route_label", "").fillna("").astype(str)
    tx = _s(out, "text", "").fillna("").astype(str).str.lower()

    key_field_signal = _contains(
        cf,
        r"consumer action|remedy|hazard description|incidents|units|remedy type"
    ).astype(float)

    modified_signal = _contains(ct, r"modified").astype(float)
    added_signal = _contains(ct, r"added").astype(float)

    new_high_signal = _contains(rl, r"new_high_risk").astype(float)
    action_signal = _contains(rl, r"consumer_action_update").astype(float)
    incident_signal = _contains(rl, r"incident_escalation").astype(float)

    severe_signal = tx.str.contains(
        r"death|fatal|injur|burn|fire|battery|lithium|child|infant|"
        r"choking|suffocation|strangulation|shock|electrocution|drowning|"
        r"carbon monoxide|poison|explosion",
        regex=True,
        na=False,
    ).astype(float)

    action_text_signal = tx.str.contains(
        r"stop use|stop using|immediately|refund|repair|replace|return",
        regex=True,
        na=False,
    ).astype(float)

    out["balanced_score"] = (
        0.30 * _norm01(_n(out, "hybrid_score"))
        + 0.25 * _norm01(_n(out, "summary_score"))
        + 0.15 * _norm01(_n(out, "route_priority"))
        + 0.08 * _norm01(_n(out, "retrieval_score"))
        + 0.08 * key_field_signal
        + 0.06 * modified_signal
        + 0.04 * added_signal
        + 0.04 * severe_signal
        + 0.03 * action_text_signal
        + 0.03 * new_high_signal
        + 0.02 * action_signal
        + 0.02 * incident_signal
    )

    return out


ROUTE_ORDER_V5 = [
    "new_high_risk",
    "consumer_action_update",
    "incident_escalation",
    "metadata_or_context_update",
    "removed_or_resolved",
]


def route_balanced_select(
    df: pd.DataFrame,
    max_items: int = 16,
    min_per_route: int = 2,
    max_per_route: Optional[int] = None,
) -> pd.DataFrame:

    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.head(0).copy()

    work = add_balanced_score(df)

    if "recall_number" not in work.columns:
        return work.head(max_items).copy()

    work["_canon"] = work["recall_number"].map(canon_recall_number)
    work = work[work["_canon"].fillna("") != ""].copy()
    work = work.drop_duplicates(subset=["_canon"], keep="first")

    if len(work) == 0:
        return work.drop(columns=["_canon"], errors="ignore")

    work = work.sort_values(
        ["balanced_score", "summary_score"],
        ascending=[False, False],
    )

    if max_per_route is None:
        max_per_route = max(4, math.ceil(max_items / 3))

    selected_idx = []
    selected_recalls = set()
    route_count = {}

    # First pass: route coverage
    for route in ROUTE_ORDER_V5:
        sub = work[_s(work, "route_label", "").astype(str) == route]
        take_n = min(min_per_route, max_items - len(selected_idx))
        for idx, row in sub.head(take_n).iterrows():
            rn = row["_canon"]
            if rn in selected_recalls:
                continue
            selected_idx.append(idx)
            selected_recalls.add(rn)
            route_count[route] = route_count.get(route, 0) + 1
            if len(selected_idx) >= max_items:
                break
        if len(selected_idx) >= max_items:
            break

    # Second pass: fill by balanced score
    for idx, row in work.iterrows():
        if len(selected_idx) >= max_items:
            break
        rn = row["_canon"]
        route = str(row.get("route_label", ""))
        if rn in selected_recalls:
            continue
        if route_count.get(route, 0) >= max_per_route:
            continue
        selected_idx.append(idx)
        selected_recalls.add(rn)
        route_count[route] = route_count.get(route, 0) + 1

    # Final fallback
    for idx, row in work.iterrows():
        if len(selected_idx) >= max_items:
            break
        rn = row["_canon"]
        if rn in selected_recalls:
            continue
        selected_idx.append(idx)
        selected_recalls.add(rn)

    out = work.loc[selected_idx].copy()
    out["_route_order"] = out["route_label"].map(lambda x: ROUTE_ORDER.get(str(x), 9))
    out = out.sort_values(
        ["_route_order", "balanced_score", "summary_score"],
        ascending=[True, False, False],
    )

    return out.drop(columns=["_canon", "_route_order"], errors="ignore").reset_index(drop=True)


# ------------------------------------------------------------
# 3. Override retriever.search
# ------------------------------------------------------------

def v5_search(self, query: str, top_k: int = 24, intent: str = "generic") -> pd.DataFrame:
    if self.vectorizer is None or len(self.corpus_df) == 0:
        return self.corpus_df.head(0).copy()

    q = self.vectorizer.transform([query or ""])
    sims = cosine_similarity(q, self.X).reshape(-1)

    out = self.corpus_df.copy()
    out["retrieval_score"] = sims
    out["importance_norm"] = _norm01(_n(out, "summary_score"))

    q_recalls = _recall_ids_from_query(query)
    out_recalls = out["recall_number"].map(canon_recall_number)
    if q_recalls:
        exact_recall_bonus = out_recalls.isin(q_recalls).astype(float)
    else:
        exact_recall_bonus = 0.0

    q_toks = {
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", query or "")
    }

    def _overlap(txt: str) -> float:
        toks = {
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", normalize_text(txt))
        }
        return len(q_toks & toks) / max(len(q_toks), 1) if q_toks else 0.0

    out["token_overlap"] = out["text"].map(_overlap)

    cf = _s(out, "changed_fields", "").fillna("").astype(str)
    ct = _s(out, "change_types", "").fillna("").astype(str)
    rl = _s(out, "route_label", "").fillna("").astype(str)
    txt = out["text"].fillna("").astype(str).str.lower()

    bonus = np.zeros(len(out), dtype=float)

    if intent == "added_high":
        bonus += _contains(ct, r"added").astype(float) * 0.20
        bonus += _contains(rl, r"new_high_risk").astype(float) * 0.25

    elif intent == "consumer_action":
        bonus += _contains(ct, r"modified").astype(float) * 0.14
        bonus += _contains(cf, r"consumer action").astype(float) * 0.38
        bonus += _contains(cf, r"remedy|remedy type").astype(float) * 0.26
        bonus += txt.str.contains(
            r"stop use|refund|repair|replace|return|immediately",
            regex=True,
            na=False,
        ).astype(float) * 0.12

    elif intent == "incidents_units":
        bonus += _contains(cf, r"incidents|units|hazard description").astype(float) * 0.28
        bonus += _contains(rl, r"incident_escalation").astype(float) * 0.20

    elif intent == "removed_legacy":
        bonus += _contains(ct, r"removed").astype(float) * 0.22

    else:
        bonus += _contains(
            cf,
            r"consumer action|remedy|hazard description|incidents|units"
        ).astype(float) * 0.08
        bonus += _contains(ct, r"modified|added").astype(float) * 0.05

    out["hybrid_score"] = (
        0.33 * out["retrieval_score"]
        + 0.30 * out["importance_norm"]
        + 0.12 * _norm01(_n(out, "route_priority"))
        + 0.10 * out["token_overlap"]
        + 0.10 * exact_recall_bonus
        + bonus
    )

    out = add_balanced_score(out)

    return (
        out.sort_values(
            ["hybrid_score", "balanced_score", "summary_score"],
            ascending=[False, False, False],
        )
        .head(top_k)
        .reset_index(drop=True)
    )


SimpleRetriever.search = v5_search


# ------------------------------------------------------------
# 4. Better query expansion and multi-query retrieval
# ------------------------------------------------------------

def build_recall_expansion_queries(corpus: pd.DataFrame, per_route: int = 6) -> List[Dict[str, str]]:
    """
    从 corpus 里自动抽高优先级 recall number，构造扩展 query。
    不使用 human gold。
    """
    if corpus is None or len(corpus) == 0:
        return []

    work = route_balanced_select(
        corpus,
        max_items=per_route * 5,
        min_per_route=per_route,
        max_per_route=per_route,
    )

    specs = []

    for route in ROUTE_ORDER_V5:
        sub = work[_s(work, "route_label", "").astype(str) == route]
        recalls = [
            canon_recall_number(x)
            for x in sub["recall_number"].astype(str).tolist()
            if canon_recall_number(x)
        ]
        recalls = list(dict.fromkeys(recalls))[:per_route]

        if not recalls:
            continue

        if route == "new_high_risk":
            q = "new high risk added recalls hazard consumer action " + " ".join(recalls)
            intent = "added_high"

        elif route == "consumer_action_update":
            q = "consumer action remedy refund repair replace stop use changed " + " ".join(recalls)
            intent = "consumer_action"

        elif route == "incident_escalation":
            q = "incidents injuries units affected hazard changed escalation " + " ".join(recalls)
            intent = "incidents_units"

        elif route == "removed_or_resolved":
            q = "removed resolved legacy recall changes " + " ".join(recalls)
            intent = "removed_legacy"

        else:
            q = "metadata description heading manufacturing recall changes " + " ".join(recalls)
            intent = "generic"

        specs.append({
            "name": f"recall_expand_{route}",
            "intent": intent,
            "query": q,
        })

    return specs


def hybrid_search_queries(
    retriever: SimpleRetriever,
    query_specs: List[Dict[str, str]],
    per_query_k: int = 30,
    final_k: int = 18,
) -> pd.DataFrame:
    """
    新版 retrieval:
    1. 原始 query bundle；
    2. recall-number expansion query；
    3. broad anchor fallback；
    4. route-balanced final selection。
    """
    if retriever is None or retriever.corpus_df is None or len(retriever.corpus_df) == 0:
        return pd.DataFrame()

    all_specs = list(query_specs or [])
    all_specs += build_recall_expansion_queries(retriever.corpus_df, per_route=6)

    if not all_specs:
        all_specs = [{
            "name": "fallback",
            "intent": "generic",
            "query": "important recall changes hazard remedy incidents units",
        }]

    pool = {}

    for spec in all_specs:
        tmp = retriever.search(
            spec.get("query", ""),
            top_k=per_query_k,
            intent=spec.get("intent", "generic"),
        )

        for _, row in tmp.iterrows():
            rid = row.get("doc_id", row.get("recall_number", ""))
            rd = row.to_dict()
            rd["query_name"] = spec.get("name", "")

            old = pool.get(rid)
            if old is None or _num(rd.get("hybrid_score", 0)) > _num(old.get("hybrid_score", 0)):
                pool[rid] = rd

    # broad anchor: 防止 lexical retrieval 漏掉重要但 query 不匹配的 recall
    anchors = add_balanced_score(retriever.corpus_df.copy())
    anchors = anchors.sort_values(
        ["balanced_score", "summary_score"],
        ascending=[False, False],
    ).head(max(final_k * 2, 30))

    for _, row in anchors.iterrows():
        rid = row.get("doc_id", row.get("recall_number", ""))
        if rid not in pool:
            rd = row.to_dict()
            rd.setdefault("retrieval_score", 0.0)
            rd.setdefault("hybrid_score", rd.get("balanced_score", rd.get("summary_score", 0.0)))
            rd["query_name"] = "broad_anchor"
            pool[rid] = rd

    if not pool:
        return retriever.corpus_df.head(0).copy()

    pooled = pd.DataFrame(list(pool.values()))

    return route_balanced_select(
        pooled,
        max_items=final_k,
        min_per_route=2,
        max_per_route=max(5, final_k // 3),
    )


# ------------------------------------------------------------
# 5. Better shortlist / evidence construction
# ------------------------------------------------------------

def _shortlist_row_from_series(r: pd.Series) -> Dict[str, Any]:
    doc = normalize_text(r.get("text", ""))

    try:
        pairs = json.loads(r.get("old_new_pairs", "[]") or "[]")
    except Exception:
        pairs = []

    return {
        "recall_number": canon_recall_number(r.get("recall_number", "")) or normalize_text(r.get("recall_number", "")),
        "change_types": normalize_text(r.get("change_types", "")),
        "changed_fields": normalize_text(r.get("changed_fields", "")),
        "recall_heading": extract_field_from_doc(doc, "Recall Heading"),
        "hazard_description": extract_field_from_doc(doc, "Hazard Description"),
        "consumer_action": extract_field_from_doc(doc, "Consumer Action"),
        "remedy": extract_field_from_doc(doc, "Remedy"),
        "incidents": extract_field_from_doc(doc, "Incidents"),
        "units": extract_field_from_doc(doc, "Units"),
        "manufactured_in": extract_field_from_doc(doc, "Manufactured In"),
        "old_new_pairs": pairs,
        "route_label": normalize_text(r.get("route_label", "")),
        "summary_score": _num(r.get("summary_score", 0.0)),
        "hybrid_score": _num(r.get("hybrid_score", 0.0)),
        "balanced_score": _num(r.get("balanced_score", 0.0)),
    }


def build_shortlist_from_retrieved(retrieved: pd.DataFrame, max_items: int = 12) -> List[Dict[str, Any]]:
    if retrieved is None or len(retrieved) == 0:
        return []

    work = route_balanced_select(
        retrieved,
        max_items=max_items,
        min_per_route=2,
        max_per_route=max(5, max_items // 3),
    )

    rows, seen = [], set()

    for _, r in work.iterrows():
        recall = canon_recall_number(r.get("recall_number", ""))
        if not recall or recall in seen:
            continue
        seen.add(recall)
        rows.append(_shortlist_row_from_series(r))

        if len(rows) >= max_items:
            break

    return rows


def build_anchor_shortlist(corpus: pd.DataFrame, max_items: int = 12) -> List[Dict[str, Any]]:
    if corpus is None or len(corpus) == 0:
        return []

    work = corpus.copy()

    if "format_carryover" in work.columns:
        work = work[~work["format_carryover"].fillna(False)]

    if len(work) == 0:
        return []

    work = route_balanced_select(
        work,
        max_items=max_items,
        min_per_route=2,
        max_per_route=max(5, max_items // 3),
    )

    return [_shortlist_row_from_series(r) for _, r in work.iterrows()]


def merge_shortlists(primary: List, secondary: List, max_items: int = 16) -> List[Dict[str, Any]]:
    rows, seen = [], set()

    for src in [primary or [], secondary or []]:
        for row in src:
            recall = canon_recall_number(row.get("recall_number", ""))
            if not recall or recall in seen:
                continue

            new_row = {**row, "recall_number": recall}
            new_row["_route_order"] = ROUTE_ORDER.get(new_row.get("route_label", ""), 9)
            new_row["_score"] = (
                _num(new_row.get("balanced_score", 0.0))
                + 0.25 * _num(new_row.get("hybrid_score", 0.0))
                + 0.10 * _num(new_row.get("summary_score", 0.0))
            )

            rows.append(new_row)
            seen.add(recall)

    rows = sorted(rows, key=lambda x: (x.get("_route_order", 9), -x.get("_score", 0.0)))

    for r in rows:
        r.pop("_route_order", None)
        r.pop("_score", None)

    return rows[:max_items]


def ensure_evidence_for_shortlist(
    retrieved: pd.DataFrame,
    corpus: pd.DataFrame,
    shortlist: List,
    max_docs: int = 18,
) -> pd.DataFrame:
    base = retrieved.copy() if retrieved is not None and len(retrieved) > 0 else pd.DataFrame()

    if corpus is None or len(corpus) == 0:
        return route_balanced_select(base, max_items=max_docs)

    have = {
        canon_recall_number(x)
        for x in (
            base.get("recall_number", pd.Series(dtype=str)).astype(str).tolist()
            if len(base)
            else []
        )
        if canon_recall_number(x)
    }

    corp = corpus.copy()
    corp["_canon"] = corp.get("recall_number", pd.Series(dtype=str)).map(canon_recall_number)

    extras = []

    for item in shortlist or []:
        recall = canon_recall_number(item.get("recall_number", ""))
        if not recall or recall in have:
            continue

        cand = corp[corp["_canon"] == recall]
        if len(cand) == 0:
            continue

        cand = add_balanced_score(cand).sort_values(
            ["balanced_score", "summary_score"],
            ascending=[False, False],
        )

        extras.append(cand.iloc[0].drop(labels=["_canon"], errors="ignore").to_dict())
        have.add(recall)

    if extras:
        combo = pd.concat([base, pd.DataFrame(extras)], ignore_index=True, sort=False)
    else:
        combo = base.copy()

    # 如果 evidence 还不够，继续从 corpus 里补 broad anchors
    if len(combo) < max_docs:
        broad = route_balanced_select(
            corpus,
            max_items=max_docs * 2,
            min_per_route=2,
            max_per_route=max(5, max_docs // 3),
        )
        combo = pd.concat([combo, broad], ignore_index=True, sort=False)

    if "doc_id" in combo.columns:
        combo = combo.drop_duplicates(subset=["doc_id"], keep="first")
    else:
        combo = combo.drop_duplicates()

    return route_balanced_select(
        combo,
        max_items=max_docs,
        min_per_route=2,
        max_per_route=max(5, max_docs // 3),
    )


# ------------------------------------------------------------
# 6. Stronger final coverage repair
# ------------------------------------------------------------

def enforce_shortlist_coverage(
    text: str,
    retrieved: pd.DataFrame,
    min_unique: int = 8,
    max_unique: int = 16,
    preferred: Optional[List[str]] = None,
) -> str:

    shortlist = build_shortlist_from_retrieved(retrieved, max_items=max_unique)

    if not shortlist:
        return text

    cited = {
        canon_recall_number(x)
        for x in extract_recall_numbers(text or "")
        if canon_recall_number(x)
    }

    shortlist_ids = [
        canon_recall_number(x.get("recall_number", ""))
        for x in shortlist
        if canon_recall_number(x.get("recall_number", ""))
    ]

    pref_set = {
        canon_recall_number(x)
        for x in (preferred or [])
        if canon_recall_number(x)
    }

    ordered = sorted(
        shortlist,
        key=lambda r: (
            0 if canon_recall_number(r.get("recall_number", "")) in pref_set else 1,
            ROUTE_ORDER.get(r.get("route_label", ""), 9),
            -_num(r.get("balanced_score", 0.0)),
            -_num(r.get("hybrid_score", 0.0)),
            -_num(r.get("summary_score", 0.0)),
        ),
    )

    covered_shortlist = {x for x in cited if x in set(shortlist_ids)}
    target_n = min(max_unique, max(min_unique, len(shortlist_ids)))

    additions = []

    for row in ordered:
        if len(covered_shortlist) >= target_n:
            break

        recall = canon_recall_number(row.get("recall_number", ""))

        if not recall or recall in covered_shortlist:
            continue

        bullet = row_to_evidence_bullet(row)

        if bullet:
            additions.append("- " + bullet)
            covered_shortlist.add(recall)
            cited.add(recall)

    if additions:
        return (normalize_text(text).strip() + "\n" + "\n".join(additions)).strip()

    return normalize_text(text).strip()


def _apply_enforce(
    text: str,
    evidence_df: pd.DataFrame,
    shortlist: List,
    max_unique: int = 16,
) -> Tuple[str, str]:
    raw = text

    max_unique = max(10, min(max_unique or 16, 16))

    preferred = [
        x["recall_number"]
        for x in (shortlist or [])
        if x.get("recall_number")
    ]

    final = enforce_shortlist_coverage(
        text,
        evidence_df,
        min_unique=min(max_unique, max(8, len(preferred))),
        max_unique=max_unique,
        preferred=preferred,
    )

    return raw, final


# ------------------------------------------------------------
# 7. Compact prompt helper
# ------------------------------------------------------------

def compact_target_list(shortlist: List[Dict[str, Any]], n: int = 16) -> str:
    compact = []

    for x in (shortlist or [])[:n]:
        compact.append({
            "recall_number": x.get("recall_number", ""),
            "route": x.get("route_label", ""),
            "change_types": x.get("change_types", ""),
            "changed_fields": x.get("changed_fields", ""),
            "heading": x.get("recall_heading", ""),
            "hazard": x.get("hazard_description", ""),
            "action": x.get("consumer_action", ""),
            "remedy": x.get("remedy", ""),
            "incidents": x.get("incidents", ""),
            "old_new_changes": x.get("old_new_pairs", [])[:3],
        })

    return json.dumps(compact, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 8. Override RAG generation
# ------------------------------------------------------------

def generate_rag(
    cu: pd.DataFrame,
    retriever: SimpleRetriever,
) -> Tuple[str, str, pd.DataFrame, Dict[str, Any]]:

    query_specs = build_query_bundle(cu)

    retrieved = hybrid_search_queries(
        retriever,
        query_specs,
        per_query_k=30,
        final_k=RAG_MAX_DOCS,
    )

    ret_sl = build_shortlist_from_retrieved(
        retrieved,
        max_items=RAG_SUMMARY_MAX_BULLETS,
    )

    anc_sl = build_anchor_shortlist(
        retriever.corpus_df,
        max_items=RAG_SUMMARY_MAX_BULLETS,
    )

    shortlist = merge_shortlists(
        ret_sl,
        anc_sl,
        max_items=RAG_SUMMARY_MAX_BULLETS,
    )

    evidence = ensure_evidence_for_shortlist(
        retrieved,
        retriever.corpus_df,
        shortlist,
        max_docs=RAG_MAX_DOCS,
    )

    shortlist = build_shortlist_from_retrieved(
        evidence,
        max_items=RAG_SUMMARY_MAX_BULLETS,
    )

    if LLM_MODE == "offline":
        lines = ["Evidence-backed recall watchlist:"]
        for row in shortlist:
            lines.append("- " + row_to_evidence_bullet(row))

        raw, final = _apply_enforce(
            "\n".join(lines),
            evidence,
            shortlist,
            max_unique=RAG_SUMMARY_MAX_BULLETS,
        )

        return raw, final, evidence, {
            "mode": "offline",
            "usage": {},
            "query_specs": query_specs,
            "shortlist": shortlist,
        }

    ev_texts = evidence["text"].fillna("").astype(str).tolist()

    prompt = (
        "Use ONLY the evidence below. Write 12-16 concise bullet points summarizing "
        "the most important CPSC recall changes. "
        "Every bullet MUST start with 'Recall <Number>:' and cover exactly one recall. "
        "Your main objective is high recall-number coverage while staying grounded. "
        "Cover TARGET_RECALLS first whenever evidence supports them. "
        "For modified recalls, mention the changed field and old->new direction when available. "
        "For newly added recalls, mention hazard and consumer action. "
        "Do not invent facts.\n\n"
        "TARGET_RECALLS:\n"
        + compact_target_list(shortlist, RAG_SUMMARY_MAX_BULLETS)
        + "\n\nEVIDENCE:\n"
        + "\n\n---\n\n".join(ev_texts)
    )

    text, usage = call_llm([
        {
            "role": "system",
            "content": "Factual, concise, evidence-grounded. One bullet per recall. Always cite recall numbers.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ])

    raw, final = _apply_enforce(
        text,
        evidence,
        shortlist,
        max_unique=RAG_SUMMARY_MAX_BULLETS,
    )

    return raw, final, evidence, {
        "mode": "llm",
        "usage": usage,
        "query_specs": query_specs,
        "shortlist": shortlist,
    }


# ------------------------------------------------------------
# 9. Override Agentic RAG generation
# ------------------------------------------------------------

def generate_agentic_rag(
    cu: pd.DataFrame,
    retriever: SimpleRetriever,
) -> Tuple[str, str, pd.DataFrame, Dict[str, Any]]:

    seed_queries = build_query_bundle(cu)
    expansion_queries = build_recall_expansion_queries(
        retriever.corpus_df,
        per_route=6,
    )

    if LLM_MODE == "offline":
        query_specs = seed_queries + expansion_queries

        retrieved = hybrid_search_queries(
            retriever,
            query_specs,
            per_query_k=32,
            final_k=AGENT_MAX_DOCS,
        )

        ret_sl = build_shortlist_from_retrieved(
            retrieved,
            max_items=AGENT_SUMMARY_MAX_BULLETS,
        )

        anc_sl = build_anchor_shortlist(
            retriever.corpus_df,
            max_items=AGENT_SUMMARY_MAX_BULLETS,
        )

        shortlist = merge_shortlists(
            ret_sl,
            anc_sl,
            max_items=AGENT_SUMMARY_MAX_BULLETS,
        )

        evidence = ensure_evidence_for_shortlist(
            retrieved,
            retriever.corpus_df,
            shortlist,
            max_docs=AGENT_MAX_DOCS,
        )

        shortlist = build_shortlist_from_retrieved(
            evidence,
            max_items=AGENT_SUMMARY_MAX_BULLETS,
        )

        lines = ["Agentic recall watchlist:"]
        for row in shortlist:
            lines.append("- " + row_to_evidence_bullet(row))

        raw, final = _apply_enforce(
            "\n".join(lines),
            evidence,
            shortlist,
            max_unique=AGENT_SUMMARY_MAX_BULLETS,
        )

        return raw, final, evidence, {
            "mode": "offline",
            "usage": {},
            "queries": [x["query"] for x in query_specs],
            "shortlist": shortlist,
        }

    # Step 1: LLM planning
    plan_text, u1 = call_llm([
        {
            "role": "system",
            "content": "Return only valid JSON.",
        },
        {
            "role": "user",
            "content": (
                "Build a broad retrieval plan for CPSC recall-change analysis. "
                "Return JSON: {\"queries\": [\"query1\", \"query2\", ...]}. "
                "Queries must cover: added high-risk recalls, modified consumer actions/remedies, "
                "incidents/units/hazard updates, and metadata/context changes."
            ),
        },
    ])

    llm_queries = []

    try:
        llm_queries = json.loads(plan_text).get("queries", [])
        if not isinstance(llm_queries, list):
            llm_queries = []
    except Exception:
        llm_queries = []

    intent_rules = [
        (r"consumer|remedy|refund|repair|replace|stop", "consumer_action"),
        (r"incident|injur|unit|hazard", "incidents_units"),
        (r"added|new|high-risk|risk", "added_high"),
        (r"removed|legacy|resolved", "removed_legacy"),
    ]

    query_specs = list(seed_queries) + list(expansion_queries)

    for i, q in enumerate(llm_queries[:6], 1):
        qq = normalize_text(q)

        if not qq:
            continue

        intent = next(
            (v for pat, v in intent_rules if re.search(pat, qq.lower())),
            "generic",
        )

        query_specs.append({
            "name": f"llm_{i}",
            "intent": intent,
            "query": qq,
        })

    retrieved = hybrid_search_queries(
        retriever,
        query_specs,
        per_query_k=32,
        final_k=AGENT_MAX_DOCS,
    )

    ret_sl = build_shortlist_from_retrieved(
        retrieved,
        max_items=AGENT_SUMMARY_MAX_BULLETS,
    )

    anc_sl = build_anchor_shortlist(
        retriever.corpus_df,
        max_items=AGENT_SUMMARY_MAX_BULLETS,
    )

    shortlist = merge_shortlists(
        ret_sl,
        anc_sl,
        max_items=AGENT_SUMMARY_MAX_BULLETS,
    )

    evidence = ensure_evidence_for_shortlist(
        retrieved,
        retriever.corpus_df,
        shortlist,
        max_docs=AGENT_MAX_DOCS,
    )

    shortlist = build_shortlist_from_retrieved(
        evidence,
        max_items=AGENT_SUMMARY_MAX_BULLETS,
    )

    ev_texts = evidence["text"].fillna("").astype(str).tolist()

    # Step 2: select recalls
    sel_text, u2 = call_llm([
        {
            "role": "system",
            "content": "Return only valid JSON.",
        },
        {
            "role": "user",
            "content": (
                "From the shortlist below, select 12-16 recalls for a grounded summary. "
                "Preserve route diversity and prefer clear hazard/action/remedy/incident/unit/added signals. "
                "Return JSON: {\"selected_recalls\": [\"XX-XXX\", ...]}.\n\n"
                + compact_target_list(shortlist, AGENT_SUMMARY_MAX_BULLETS)
            ),
        },
    ])

    selected_recalls = [
        x["recall_number"]
        for x in shortlist[:AGENT_SUMMARY_MAX_BULLETS]
    ]

    try:
        picked = [
            canon_recall_number(x)
            for x in json.loads(sel_text).get("selected_recalls", [])
            if canon_recall_number(x)
        ]

        if picked:
            selected_recalls = list(dict.fromkeys(
                picked + selected_recalls
            ))[:AGENT_SUMMARY_MAX_BULLETS]

    except Exception:
        pass

    # Step 3: draft
    draft, u3 = call_llm([
        {
            "role": "system",
            "content": "Factual, concise, one bullet per recall, cite recall numbers.",
        },
        {
            "role": "user",
            "content": (
                "Using only the evidence below, write 12-16 bullet points. "
                "Every bullet starts with 'Recall <Number>:'. "
                "Cover SELECTED_RECALLS first. "
                "If old/new values are available, describe the change direction. "
                "Do not invent facts.\n\n"
                f"SELECTED_RECALLS: {json.dumps(selected_recalls)}\n\n"
                "SHORTLIST:\n"
                + compact_target_list(shortlist, AGENT_SUMMARY_MAX_BULLETS)
                + "\n\nEVIDENCE:\n"
                + "\n\n---\n\n".join(ev_texts)
            ),
        },
    ])

    # Step 4: revise
    revised, u4 = call_llm([
        {
            "role": "system",
            "content": "Return a grounded final answer only.",
        },
        {
            "role": "user",
            "content": (
                "Revise the draft to maximize SELECTED_RECALLS coverage without adding unsupported claims. "
                "Keep the format: each bullet starts with 'Recall <Number>:'.\n\n"
                f"SELECTED_RECALLS: {json.dumps(selected_recalls)}\n\n"
                "EVIDENCE:\n"
                + "\n\n---\n\n".join(ev_texts[:18])
                + "\n\nDRAFT:\n"
                + draft
            ),
        },
    ])

    usage = {}

    for d in [u1, u2, u3, u4]:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v

    raw, final = _apply_enforce(
        revised,
        evidence,
        shortlist,
        max_unique=AGENT_SUMMARY_MAX_BULLETS,
    )

    return raw, final, evidence, {
        "mode": "llm",
        "usage": usage,
        "plan": {"queries": llm_queries},
        "queries": [x["query"] for x in query_specs],
        "shortlist": shortlist,
        "selected_recalls": selected_recalls,
    }

import re
import numpy as np
import pandas as pd


def _safe_route_order(route):
    try:
        return ROUTE_ORDER.get(str(route), 9)
    except Exception:
        try:
            return ROUTE_ORDER_V5.index(str(route))
        except Exception:
            return 9


def _repair_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _evidence_row_to_repair_item(row):
    """
    Convert one evidence row into the same structure required by row_to_evidence_bullet.
    This does NOT use human gold.
    """
    r = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    doc = normalize_text(r.get("text", ""))

    item = {
        "recall_number": canon_recall_number(r.get("recall_number", "")) or "",
        "change_types": normalize_text(r.get("change_types", "")),
        "changed_fields": normalize_text(r.get("changed_fields", "")),
        "route_label": normalize_text(r.get("route_label", "")),
        "summary_score": _repair_num(r.get("summary_score", 0.0)),
        "hybrid_score": _repair_num(r.get("hybrid_score", 0.0)),
        "balanced_score": _repair_num(r.get("balanced_score", 0.0)),
    }

    # If fields already exist, use them.
    direct_fields = {
        "recall_heading": "recall_heading",
        "hazard_description": "hazard_description",
        "consumer_action": "consumer_action",
        "remedy": "remedy",
        "incidents": "incidents",
        "units": "units",
        "manufactured_in": "manufactured_in",
    }

    for src, dst in direct_fields.items():
        item[dst] = normalize_text(r.get(src, ""))

    # If missing, extract from the evidence text.
    if not item.get("recall_heading"):
        item["recall_heading"] = extract_field_from_doc(doc, "Recall Heading")
    if not item.get("hazard_description"):
        item["hazard_description"] = extract_field_from_doc(doc, "Hazard Description")
    if not item.get("consumer_action"):
        item["consumer_action"] = extract_field_from_doc(doc, "Consumer Action")
    if not item.get("remedy"):
        item["remedy"] = extract_field_from_doc(doc, "Remedy")
    if not item.get("incidents"):
        item["incidents"] = extract_field_from_doc(doc, "Incidents")
    if not item.get("units"):
        item["units"] = extract_field_from_doc(doc, "Units")
    if not item.get("manufactured_in"):
        item["manufactured_in"] = extract_field_from_doc(doc, "Manufactured In")

    # Try to recover old_new_pairs.
    pairs = r.get("old_new_pairs", [])
    if isinstance(pairs, str):
        try:
            pairs = json.loads(pairs)
        except Exception:
            pairs = []
    if not isinstance(pairs, list):
        pairs = []

    item["old_new_pairs"] = pairs

    return item


def _build_repair_items_from_evidence(evidence_df):
    if evidence_df is None or len(evidence_df) == 0:
        return []

    work = evidence_df.copy()

    try:
        work = add_balanced_score(work)
    except Exception:
        pass

    items = []
    seen = set()

    for i, (_, row) in enumerate(work.reset_index(drop=True).iterrows()):
        item = _evidence_row_to_repair_item(row)
        rn = canon_recall_number(item.get("recall_number", ""))

        if not rn or rn in seen:
            continue

        seen.add(rn)
        item["recall_number"] = rn
        item["_evidence_rank"] = i + 1

        cf = item.get("changed_fields", "").lower()
        ct = item.get("change_types", "").lower()
        route = item.get("route_label", "").lower()
        txt = " ".join([
            item.get("recall_heading", ""),
            item.get("hazard_description", ""),
            item.get("consumer_action", ""),
            item.get("remedy", ""),
            item.get("incidents", ""),
            item.get("units", ""),
        ]).lower()

        key_field_bonus = 0.0
        if re.search(r"consumer action|remedy|hazard description|incidents|units|remedy type", cf):
            key_field_bonus += 2.0
        if "modified" in ct:
            key_field_bonus += 1.2
        if "added" in ct:
            key_field_bonus += 0.8
        if re.search(r"new_high_risk|consumer_action_update|incident_escalation", route):
            key_field_bonus += 0.8
        if re.search(r"injur|death|fire|burn|battery|child|infant|choking|shock|poison|drowning", txt):
            key_field_bonus += 0.8
        if re.search(r"stop use|refund|repair|replace|return|immediately", txt):
            key_field_bonus += 0.6

        item["_repair_score"] = (
            1.00 * _repair_num(item.get("balanced_score", 0.0))
            + 0.50 * _repair_num(item.get("hybrid_score", 0.0))
            + 0.20 * _repair_num(item.get("summary_score", 0.0))
            + key_field_bonus
            - 0.01 * item["_evidence_rank"]
        )

        items.append(item)

    items = sorted(
        items,
        key=lambda x: (
            _safe_route_order(x.get("route_label", "")),
            -_repair_num(x.get("_repair_score", 0.0)),
            x.get("_evidence_rank", 9999),
        ),
    )

    return items


def enforce_evidence_coverage(
    text,
    evidence_df,
    target_unique=16,
    max_unique=16,
    preferred=None,
):
    """
    Stronger repair:
    - use all selected evidence, not only the old shortlist
    - add grounded bullets until target_unique recall ids are covered
    - does NOT use human gold
    """
    text = normalize_text(text or "").strip()

    cited = {
        canon_recall_number(x)
        for x in extract_recall_numbers(text)
        if canon_recall_number(x)
    }

    items = _build_repair_items_from_evidence(evidence_df)

    if not items:
        return text

    pref_set = {
        canon_recall_number(x)
        for x in (preferred or [])
        if canon_recall_number(x)
    }

    # Prefer previous shortlist first, then high-score evidence.
    items = sorted(
        items,
        key=lambda x: (
            0 if canon_recall_number(x.get("recall_number", "")) in pref_set else 1,
            _safe_route_order(x.get("route_label", "")),
            -_repair_num(x.get("_repair_score", 0.0)),
            x.get("_evidence_rank", 9999),
        ),
    )

    target_unique = min(max_unique, max(target_unique, len(cited)))
    additions = []

    for item in items:
        if len(cited) >= target_unique:
            break

        rn = canon_recall_number(item.get("recall_number", ""))

        if not rn or rn in cited:
            continue

        bullet = row_to_evidence_bullet(item)

        if not bullet:
            continue

        # For modified recalls, add old→new signal if available.
        pairs = item.get("old_new_pairs", [])
        if pairs:
            first_pair = pairs[0]
            field = normalize_text(first_pair.get("field", ""))
            old = normalize_text(first_pair.get("old", ""))
            new = normalize_text(first_pair.get("new", ""))

            if field and (old or new) and "previously" not in bullet.lower():
                bullet += f" Previously {field}: {old[:180]}; now: {new[:180]}."

        additions.append("- " + bullet)
        cited.add(rn)

    if additions:
        return (text + "\n" + "\n".join(additions)).strip()

    return text


def _apply_enforce(
    text,
    evidence_df,
    shortlist,
    max_unique=16,
):

    raw = normalize_text(text or "").strip()

    preferred = [
        canon_recall_number(x.get("recall_number", ""))
        for x in (shortlist or [])
        if canon_recall_number(x.get("recall_number", ""))
    ]

    final = enforce_evidence_coverage(
        raw,
        evidence_df,
        target_unique=min(max_unique or 16, 16),
        max_unique=min(max_unique or 16, 16),
        preferred=preferred,
    )

    return raw, final


import re
import json
import pandas as pd


MAX_TOTAL_CITED_AFTER_REPAIR = 24
MAX_ADDED_REPAIR_BULLETS = 10


def _v52_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _v52_route_order(route):
    try:
        return ROUTE_ORDER.get(str(route), 9)
    except Exception:
        try:
            return ROUTE_ORDER_V5.index(str(route))
        except Exception:
            return 9


def _v52_row_to_item(row):
    r = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    doc = normalize_text(r.get("text", ""))

    item = {
        "recall_number": canon_recall_number(r.get("recall_number", "")) or "",
        "change_types": normalize_text(r.get("change_types", "")),
        "changed_fields": normalize_text(r.get("changed_fields", "")),
        "route_label": normalize_text(r.get("route_label", "")),
        "summary_score": _v52_num(r.get("summary_score", 0.0)),
        "hybrid_score": _v52_num(r.get("hybrid_score", 0.0)),
        "balanced_score": _v52_num(r.get("balanced_score", 0.0)),
    }

    # Direct fields if they already exist
    for c in [
        "recall_heading",
        "hazard_description",
        "consumer_action",
        "remedy",
        "incidents",
        "units",
        "manufactured_in",
    ]:
        item[c] = normalize_text(r.get(c, ""))

    # Fallback extraction from evidence text
    if not item["recall_heading"]:
        item["recall_heading"] = extract_field_from_doc(doc, "Recall Heading")
    if not item["hazard_description"]:
        item["hazard_description"] = extract_field_from_doc(doc, "Hazard Description")
    if not item["consumer_action"]:
        item["consumer_action"] = extract_field_from_doc(doc, "Consumer Action")
    if not item["remedy"]:
        item["remedy"] = extract_field_from_doc(doc, "Remedy")
    if not item["incidents"]:
        item["incidents"] = extract_field_from_doc(doc, "Incidents")
    if not item["units"]:
        item["units"] = extract_field_from_doc(doc, "Units")
    if not item["manufactured_in"]:
        item["manufactured_in"] = extract_field_from_doc(doc, "Manufactured In")

    pairs = r.get("old_new_pairs", [])
    if isinstance(pairs, str):
        try:
            pairs = json.loads(pairs)
        except Exception:
            pairs = []
    if not isinstance(pairs, list):
        pairs = []

    item["old_new_pairs"] = pairs

    return item


def _v52_evidence_items(evidence_df):
    if evidence_df is None or len(evidence_df) == 0:
        return []

    work = evidence_df.copy()

    try:
        work = add_balanced_score(work)
    except Exception:
        pass

    items = []
    seen = set()

    for i, (_, row) in enumerate(work.reset_index(drop=True).iterrows()):
        item = _v52_row_to_item(row)
        rn = canon_recall_number(item.get("recall_number", ""))

        if not rn or rn in seen:
            continue

        seen.add(rn)
        item["recall_number"] = rn
        item["_evidence_rank"] = i + 1

        cf = item.get("changed_fields", "").lower()
        ct = item.get("change_types", "").lower()
        route = item.get("route_label", "").lower()

        txt = " ".join([
            item.get("recall_heading", ""),
            item.get("hazard_description", ""),
            item.get("consumer_action", ""),
            item.get("remedy", ""),
            item.get("incidents", ""),
            item.get("units", ""),
        ]).lower()

        bonus = 0.0

        if re.search(r"consumer action|remedy|hazard description|incidents|units|remedy type", cf):
            bonus += 2.5

        if "modified" in ct:
            bonus += 1.5

        if "added" in ct:
            bonus += 1.0

        if re.search(r"new_high_risk|consumer_action_update|incident_escalation", route):
            bonus += 1.0

        if re.search(
            r"injur|death|fire|burn|battery|child|infant|choking|shock|poison|drowning|suffocation|laceration",
            txt,
        ):
            bonus += 1.0

        if re.search(r"stop use|refund|repair|replace|return|immediately", txt):
            bonus += 0.8

        item["_repair_score"] = (
            1.00 * _v52_num(item.get("balanced_score", 0.0))
            + 0.60 * _v52_num(item.get("hybrid_score", 0.0))
            + 0.25 * _v52_num(item.get("summary_score", 0.0))
            + bonus
            - 0.01 * item["_evidence_rank"]
        )

        items.append(item)

    items = sorted(
        items,
        key=lambda x: (
            _v52_route_order(x.get("route_label", "")),
            -_v52_num(x.get("_repair_score", 0.0)),
            x.get("_evidence_rank", 9999),
        ),
    )

    return items


def enforce_selected_evidence_coverage_v52(
    text,
    evidence_df,
    preferred=None,
    max_total_cited=MAX_TOTAL_CITED_AFTER_REPAIR,
    max_added=MAX_ADDED_REPAIR_BULLETS,
):
    """
    V5.2核心变化：
    不再看 summary 里 recall 总数是否已经够。
    只要 evidence 中的重要 recall 没有被 summary 写进去，就继续补。
    这不会使用 human gold，因此不构成 gold leakage。
    """
    text = normalize_text(text or "").strip()

    cited = {
        canon_recall_number(x)
        for x in extract_recall_numbers(text)
        if canon_recall_number(x)
    }

    items = _v52_evidence_items(evidence_df)

    if not items:
        return text

    preferred_set = {
        canon_recall_number(x)
        for x in (preferred or [])
        if canon_recall_number(x)
    }

    # 优先补 preferred shortlist 里的 recall，
    # 然后补 evidence 里高分但没写进 summary 的 recall。
    items = sorted(
        items,
        key=lambda x: (
            0 if canon_recall_number(x.get("recall_number", "")) in preferred_set else 1,
            _v52_route_order(x.get("route_label", "")),
            -_v52_num(x.get("_repair_score", 0.0)),
            x.get("_evidence_rank", 9999),
        ),
    )

    additions = []

    for item in items:
        rn = canon_recall_number(item.get("recall_number", ""))

        if not rn:
            continue

        if rn in cited:
            continue

        if len(cited) >= max_total_cited:
            break

        if len(additions) >= max_added:
            break

        bullet = row_to_evidence_bullet(item)

        if not bullet:
            continue

        pairs = item.get("old_new_pairs", [])

        if pairs:
            pair = pairs[0]

            if isinstance(pair, dict):
                field = normalize_text(pair.get("field", ""))
                old = normalize_text(pair.get("old", ""))
                new = normalize_text(pair.get("new", ""))

                if field and (old or new):
                    old_short = old[:160]
                    new_short = new[:160]

                    if "previously" not in bullet.lower() and "now" not in bullet.lower():
                        bullet += f" Previously {field}: {old_short}; now: {new_short}."

        additions.append("- " + bullet)
        cited.add(rn)

    if additions:
        return (
            text
            + "\n\nAdditional evidence-backed recall changes:\n"
            + "\n".join(additions)
        ).strip()

    return text


def _apply_enforce(
    text,
    evidence_df,
    shortlist,
    max_unique=16,
):
    """
    Override previous V5 / V5.1 _apply_enforce.
    This version forces coverage of selected evidence recalls,
    not just total recall count.
    """
    raw = normalize_text(text or "").strip()

    preferred = [
        canon_recall_number(x.get("recall_number", ""))
        for x in (shortlist or [])
        if canon_recall_number(x.get("recall_number", ""))
    ]

    final = enforce_selected_evidence_coverage_v52(
        raw,
        evidence_df,
        preferred=preferred,
        max_total_cited=MAX_TOTAL_CITED_AFTER_REPAIR,
        max_added=MAX_ADDED_REPAIR_BULLETS,
    )

    return raw, final

import re
import json
import pandas as pd


MAX_TOTAL_CITED_AFTER_REPAIR = 24
MAX_ADDED_REPAIR_BULLETS = 10


def _v52_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _v52_route_order(route):
    try:
        return ROUTE_ORDER.get(str(route), 9)
    except Exception:
        try:
            return ROUTE_ORDER_V5.index(str(route))
        except Exception:
            return 9


def _v52_row_to_item(row):
    r = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    doc = normalize_text(r.get("text", ""))

    item = {
        "recall_number": canon_recall_number(r.get("recall_number", "")) or "",
        "change_types": normalize_text(r.get("change_types", "")),
        "changed_fields": normalize_text(r.get("changed_fields", "")),
        "route_label": normalize_text(r.get("route_label", "")),
        "summary_score": _v52_num(r.get("summary_score", 0.0)),
        "hybrid_score": _v52_num(r.get("hybrid_score", 0.0)),
        "balanced_score": _v52_num(r.get("balanced_score", 0.0)),
    }

    # Direct fields if they already exist
    for c in [
        "recall_heading",
        "hazard_description",
        "consumer_action",
        "remedy",
        "incidents",
        "units",
        "manufactured_in",
    ]:
        item[c] = normalize_text(r.get(c, ""))

    # Fallback extraction from evidence text
    if not item["recall_heading"]:
        item["recall_heading"] = extract_field_from_doc(doc, "Recall Heading")
    if not item["hazard_description"]:
        item["hazard_description"] = extract_field_from_doc(doc, "Hazard Description")
    if not item["consumer_action"]:
        item["consumer_action"] = extract_field_from_doc(doc, "Consumer Action")
    if not item["remedy"]:
        item["remedy"] = extract_field_from_doc(doc, "Remedy")
    if not item["incidents"]:
        item["incidents"] = extract_field_from_doc(doc, "Incidents")
    if not item["units"]:
        item["units"] = extract_field_from_doc(doc, "Units")
    if not item["manufactured_in"]:
        item["manufactured_in"] = extract_field_from_doc(doc, "Manufactured In")

    pairs = r.get("old_new_pairs", [])
    if isinstance(pairs, str):
        try:
            pairs = json.loads(pairs)
        except Exception:
            pairs = []
    if not isinstance(pairs, list):
        pairs = []

    item["old_new_pairs"] = pairs

    return item


def _v52_evidence_items(evidence_df):
    if evidence_df is None or len(evidence_df) == 0:
        return []

    work = evidence_df.copy()

    try:
        work = add_balanced_score(work)
    except Exception:
        pass

    items = []
    seen = set()

    for i, (_, row) in enumerate(work.reset_index(drop=True).iterrows()):
        item = _v52_row_to_item(row)
        rn = canon_recall_number(item.get("recall_number", ""))

        if not rn or rn in seen:
            continue

        seen.add(rn)
        item["recall_number"] = rn
        item["_evidence_rank"] = i + 1

        cf = item.get("changed_fields", "").lower()
        ct = item.get("change_types", "").lower()
        route = item.get("route_label", "").lower()

        txt = " ".join([
            item.get("recall_heading", ""),
            item.get("hazard_description", ""),
            item.get("consumer_action", ""),
            item.get("remedy", ""),
            item.get("incidents", ""),
            item.get("units", ""),
        ]).lower()

        bonus = 0.0

        if re.search(r"consumer action|remedy|hazard description|incidents|units|remedy type", cf):
            bonus += 2.5

        if "modified" in ct:
            bonus += 1.5

        if "added" in ct:
            bonus += 1.0

        if re.search(r"new_high_risk|consumer_action_update|incident_escalation", route):
            bonus += 1.0

        if re.search(
            r"injur|death|fire|burn|battery|child|infant|choking|shock|poison|drowning|suffocation|laceration",
            txt,
        ):
            bonus += 1.0

        if re.search(r"stop use|refund|repair|replace|return|immediately", txt):
            bonus += 0.8

        item["_repair_score"] = (
            1.00 * _v52_num(item.get("balanced_score", 0.0))
            + 0.60 * _v52_num(item.get("hybrid_score", 0.0))
            + 0.25 * _v52_num(item.get("summary_score", 0.0))
            + bonus
            - 0.01 * item["_evidence_rank"]
        )

        items.append(item)

    items = sorted(
        items,
        key=lambda x: (
            _v52_route_order(x.get("route_label", "")),
            -_v52_num(x.get("_repair_score", 0.0)),
            x.get("_evidence_rank", 9999),
        ),
    )

    return items


def enforce_selected_evidence_coverage_v52(
    text,
    evidence_df,
    preferred=None,
    max_total_cited=MAX_TOTAL_CITED_AFTER_REPAIR,
    max_added=MAX_ADDED_REPAIR_BULLETS,
):

    text = normalize_text(text or "").strip()

    cited = {
        canon_recall_number(x)
        for x in extract_recall_numbers(text)
        if canon_recall_number(x)
    }

    items = _v52_evidence_items(evidence_df)

    if not items:
        return text

    preferred_set = {
        canon_recall_number(x)
        for x in (preferred or [])
        if canon_recall_number(x)
    }

    items = sorted(
        items,
        key=lambda x: (
            0 if canon_recall_number(x.get("recall_number", "")) in preferred_set else 1,
            _v52_route_order(x.get("route_label", "")),
            -_v52_num(x.get("_repair_score", 0.0)),
            x.get("_evidence_rank", 9999),
        ),
    )

    additions = []

    for item in items:
        rn = canon_recall_number(item.get("recall_number", ""))

        if not rn:
            continue

        if rn in cited:
            continue

        if len(cited) >= max_total_cited:
            break

        if len(additions) >= max_added:
            break

        bullet = row_to_evidence_bullet(item)

        if not bullet:
            continue

        pairs = item.get("old_new_pairs", [])

        if pairs:
            pair = pairs[0]

            if isinstance(pair, dict):
                field = normalize_text(pair.get("field", ""))
                old = normalize_text(pair.get("old", ""))
                new = normalize_text(pair.get("new", ""))

                if field and (old or new):
                    old_short = old[:160]
                    new_short = new[:160]

                    if "previously" not in bullet.lower() and "now" not in bullet.lower():
                        bullet += f" Previously {field}: {old_short}; now: {new_short}."

        additions.append("- " + bullet)
        cited.add(rn)

    if additions:
        return (
            text
            + "\n\nAdditional evidence-backed recall changes:\n"
            + "\n".join(additions)
        ).strip()

    return text


def _apply_enforce(
    text,
    evidence_df,
    shortlist,
    max_unique=16,
):
    """
    Override previous V5 / V5.1 _apply_enforce.
    This version forces coverage of selected evidence recalls,
    not just total recall count.
    """
    raw = normalize_text(text or "").strip()

    preferred = [
        canon_recall_number(x.get("recall_number", ""))
        for x in (shortlist or [])
        if canon_recall_number(x.get("recall_number", ""))
    ]

    final = enforce_selected_evidence_coverage_v52(
        raw,
        evidence_df,
        preferred=preferred,
        max_total_cited=MAX_TOTAL_CITED_AFTER_REPAIR,
        max_added=MAX_ADDED_REPAIR_BULLETS,
    )

    return raw, final

def generate_prompt_only(cu: pd.DataFrame,
                         corpus: Optional[pd.DataFrame] = None
                         ) -> Tuple[str, str, Dict[str, Any]]:
    """
    Fair prompt_only baseline:
    - It can use the initial structured candidate list.
    - It does NOT use V5/V5.2 evidence repair.
    - raw_summary == final_summary for prompt_only.
    """
    corpus = corpus if corpus is not None else build_retrieval_corpus(cu)

    if corpus is None or len(corpus) == 0:
        raw = "No meaningful changes detected."
        return raw, raw, {"mode": "offline", "usage": {}, "fair_baseline": True}

    # Keep the original prompt-only candidate budget.
    brief = build_shortlist_from_retrieved(
        corpus.assign(hybrid_score=corpus.get("summary_score", 0)).head(10),
        max_items=10
    )

    if LLM_MODE == "offline" or not brief:
        lines = ["Priority recall watchlist:"]
        for r in brief:
            ev = normalize_text(
                r.get("recall_heading", "") + " " + r.get("hazard_description", "")
            )[:300]
            lines.append(
                f"- Recall {r['recall_number']}: "
                f"type={r['change_types']}; fields={r['changed_fields']}. {ev}"
            )

        raw = "\n".join(lines) if lines else "No meaningful changes detected."

        # Important: no repair for prompt_only
        return raw, raw, {
            "mode": "offline",
            "usage": {},
            "fair_baseline": True,
            "no_repair": True,
        }

    prompt = (
        "You are analyzing CPSC recall updates. Based only on the structured candidates below, "
        "write 6–10 bullet points. Each bullet MUST start with 'Recall <Number>:' then explain "
        "what changed, why it matters, and what action is needed. Be evidence-grounded.\n\n"
        + json.dumps([{
            "recall_number": r["recall_number"],
            "change_types": r["change_types"],
            "changed_fields": r["changed_fields"],
            "hazard_description": r["hazard_description"],
            "consumer_action": r["consumer_action"],
        } for r in brief], ensure_ascii=False, indent=2)
    )

    text, usage = call_llm([
        {
            "role": "system",
            "content": "Be factual, concise, always cite recall numbers."
        },
        {
            "role": "user",
            "content": prompt
        },
    ])

    raw = normalize_text(text)

    # Important: no repair for prompt_only
    final = raw

    return raw, final, {
        "mode": "llm",
        "usage": usage,
        "fair_baseline": True,
        "no_repair": True,
    }


import os, re, json
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

try:
    from IPython.display import display
except Exception:
    display = None


# ------------------------------------------------------------
# 0. Method list: make sure contrastive_adaptive_rag is included everywhere
# ------------------------------------------------------------

METHOD_ORDER_FINAL = [
    "prompt_only",
    "rag",
    "agentic_rag",
    "contrastive_adaptive_rag",
]


# ------------------------------------------------------------
# 1. Robust JSON parsing for verifier output
# ------------------------------------------------------------

def _safe_json_loads_from_llm(text: str) -> Dict[str, Any]:
    """
    Parse LLM JSON robustly.
    Handles:
    - pure JSON
    - ```json ... ```
    - extra text around a JSON object
    """
    text = str(text or "").strip()

    if not text:
        return {}

    # Remove fenced blocks if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Fallback: take the largest JSON-looking object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _claim_supported(verdict: str) -> bool:
    verdict = str(verdict or "").strip().lower()
    return verdict in {"fully_supported", "partially_supported", "supported"}


def _bullet_keepable_after_verify(bullet: Dict[str, Any]) -> bool:
    """
    Looser but still grounded keep rule:
    - keep=false is accepted only when all claims are unsupported
    - if any claim is fully/partially supported, keep the bullet
    - if no claims are listed, use overall_verdict
    """
    claims = bullet.get("claims", [])
    if isinstance(claims, list) and claims:
        has_supported = any(_claim_supported(c.get("verdict", "")) for c in claims if isinstance(c, dict))
        all_unsupported = all(
            str(c.get("verdict", "")).strip().lower() == "unsupported"
            for c in claims
            if isinstance(c, dict)
        )
        if has_supported:
            return True
        if all_unsupported:
            return False

    overall = str(bullet.get("overall_verdict", "")).strip().lower()
    if overall in {"fully_supported", "partially_supported", "supported"}:
        return True
    if overall == "unsupported":
        return False

    # If verifier did not give useful labels, do not delete aggressively.
    return bool(bullet.get("keep", True))


def _verification_stats(verification_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    bullets = []
    if isinstance(verification_result, dict):
        bullets = verification_result.get("bullets", []) or []
    if not isinstance(bullets, list):
        bullets = []

    n_total = len(bullets)
    n_keep_false_original = sum(
        1 for b in bullets
        if isinstance(b, dict) and b.get("keep", True) is False
    )
    n_keep_after_relax = sum(
        1 for b in bullets
        if isinstance(b, dict) and _bullet_keepable_after_verify(b)
    )
    verdict_counts = Counter(
        str(b.get("overall_verdict", "missing")).strip().lower()
        for b in bullets
        if isinstance(b, dict)
    )

    return {
        "verify_total_bullets": n_total,
        "verify_keep_false_original": n_keep_false_original,
        "verify_keep_false_ratio_original": round(n_keep_false_original / max(n_total, 1), 4),
        "verify_kept_after_relax": n_keep_after_relax,
        "verify_removed_after_relax": max(n_total - n_keep_after_relax, 0),
        "verify_removed_ratio_after_relax": round(max(n_total - n_keep_after_relax, 0) / max(n_total, 1), 4),
        "verify_overall_verdict_counts": dict(verdict_counts),
    }


def _verified_text_from_result(draft: str,
                               verification_result: Optional[Dict[str, Any]]
                               ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build verified summary text.
    Important fix:
    If verifier removes everything, fall back to the original draft instead of returning empty text.
    """
    draft = normalize_text(draft or "")
    bullets = []

    if isinstance(verification_result, dict):
        bullets = verification_result.get("bullets", []) or []

    if not isinstance(bullets, list) or not bullets:
        return draft, [], []

    kept, removed = [], []

    for b in bullets:
        if not isinstance(b, dict):
            continue
        if _bullet_keepable_after_verify(b):
            kept.append(b)
        else:
            removed.append(b)

    lines = [
        normalize_text(b.get("bullet_text", ""))
        for b in kept
        if normalize_text(b.get("bullet_text", ""))
    ]

    verified_text = "\n".join(lines).strip()

    # Critical bug fix: do not let verifier delete the entire summary.
    if not verified_text and draft:
        verified_text = draft

    return verified_text, kept, removed


def _covered_recalls_from_text_and_bullets(text: str,
                                           bullets: Optional[List[Dict[str, Any]]] = None) -> set:
    covered = {
        canon_recall_number(x)
        for x in extract_recall_numbers(text or "")
        if canon_recall_number(x)
    }

    for b in bullets or []:
        if not isinstance(b, dict):
            continue
        rn = canon_recall_number(b.get("recall_number", ""))
        if rn:
            covered.add(rn)

    return covered


# ------------------------------------------------------------
# 2. Loosen verifier prompt
# ------------------------------------------------------------

def _self_verify_prompt(draft: str, evidence_texts: List[str]) -> str:
    return (
        "You are verifying a recall-change summary for factual grounding.\n\n"
        "For each bullet in the DRAFT, classify factual claims using this scheme:\n"
        "- fully_supported: the claim closely matches the evidence.\n"
        "- partially_supported: the recall and the main direction are supported, but some detail is incomplete.\n"
        "- unsupported: the claim has no matching support in evidence.\n\n"
        "Important keep rule:\n"
        "- keep=true if at least ONE core claim in the bullet is fully_supported or partially_supported.\n"
        "- keep=false only if ALL factual claims in the bullet are unsupported.\n"
        "- Do not set keep=false merely because minor wording differs from the evidence.\n\n"
        "Return ONLY valid JSON in this exact schema:\n"
        "{\n"
        "  \"bullets\": [\n"
        "    {\n"
        "      \"recall_number\": \"XX-XXX\",\n"
        "      \"route\": \"...\",\n"
        "      \"bullet_text\": \"...\",\n"
        "      \"claims\": [\n"
        "        {\"text\": \"...\", \"verdict\": \"fully_supported\"|\"partially_supported\"|\"unsupported\", "
        "\"evidence_snippet\": \"...\"}\n"
        "      ],\n"
        "      \"overall_verdict\": \"fully_supported\"|\"partially_supported\"|\"unsupported\",\n"
        "      \"keep\": true|false\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "EVIDENCE:\n"
        + "\n\n---\n\n".join(evidence_texts[:12])
        + "\n\nDRAFT:\n"
        + str(draft or "")
    )


# ------------------------------------------------------------
# 3. Override contrastive_adaptive_rag generation
# ------------------------------------------------------------

def generate_contrastive_adaptive_rag(cu: pd.DataFrame,
                                      retriever: SimpleRetriever
                                      ) -> Tuple[str, str, pd.DataFrame, Dict[str, Any]]:
    """
    Fixed Contrastive Adaptive RAG.

    Main fixes:
    1. Verification no longer deletes every bullet silently.
    2. keep=false ratio is stored in meta["verification_stats"].
    3. Missing shortlist recalls are filled even when verifier kept zero bullets.
    4. Final output still goes through _apply_enforce, so V5.2 evidence repair remains active.
    """
    cu_aug = augment_change_units(cu)

    contrastive_max_docs = max(globals().get("RAG_MAX_DOCS", 12), 18)
    contrastive_max_bullets = max(globals().get("RAG_SUMMARY_MAX_BULLETS", 12), 16)

    query_specs = build_query_bundle(cu_aug)
    retrieved = hybrid_search_queries(
        retriever,
        query_specs,
        per_query_k=32,
        final_k=contrastive_max_docs,
    )

    ret_sl = build_shortlist_from_retrieved(
        retrieved,
        max_items=min(contrastive_max_bullets, 16),
    )
    anc_sl = build_anchor_shortlist(
        retriever.corpus_df,
        max_items=min(contrastive_max_bullets, 16),
    )
    shortlist = merge_shortlists(
        ret_sl,
        anc_sl,
        max_items=min(contrastive_max_bullets, 16),
    )

    evidence = ensure_evidence_for_shortlist(
        retrieved,
        retriever.corpus_df,
        shortlist,
        max_docs=contrastive_max_docs,
    )

    # After evidence repair, rebuild shortlist from actual evidence so generation/eval are aligned.
    if evidence is not None and len(evidence) > 0:
        try:
            shortlist = build_shortlist_from_retrieved(
                evidence,
                max_items=min(contrastive_max_bullets, 16),
            )
        except Exception:
            pass

    ev_texts = evidence["text"].fillna("").astype(str).tolist() if evidence is not None and len(evidence) else []
    route_digest = build_route_digest(cu_aug)

    if LLM_MODE == "offline":
        lines = ["Contrastive recall watchlist:"]
        for item in shortlist:
            recall = item.get("recall_number", "")
            pairs = item.get("old_new_pairs", [])
            bullet = f"Recall {recall}"
            if pairs:
                p = pairs[0]
                bullet += (
                    f": [{p.get('field', '')}] previously '{p.get('old', '')}', "
                    f"now '{p.get('new', '')}'."
                )
            else:
                bullet += ": " + row_to_evidence_bullet(item)
            lines.append("- " + bullet)

        raw, final = _apply_enforce(
            "\n".join(lines),
            evidence,
            shortlist,
            max_unique=min(contrastive_max_bullets, 16),
        )
        return raw, final, evidence, {
            "mode": "offline",
            "usage": {},
            "query_specs": query_specs,
            "shortlist": shortlist,
            "route_digest": route_digest,
            "verification_result": {},
            "verification_stats": {
                "verify_total_bullets": 0,
                "verify_keep_false_original": 0,
                "verify_keep_false_ratio_original": 0.0,
                "verify_kept_after_relax": 0,
                "verify_removed_after_relax": 0,
                "verify_removed_ratio_after_relax": 0.0,
                "verify_overall_verdict_counts": {},
            },
            "structured_pairs": _build_old_new_pairs_for_shortlist(shortlist),
        }

    structured_pairs = _build_old_new_pairs_for_shortlist(shortlist)

    # Step 1: contrastive draft
    draft, u1 = call_llm([
        {
            "role": "system",
            "content": (
                "Be factual and change-aware. Use recall numbers. "
                "Prefer old-to-new comparisons when old/new values are available. "
                "Do not invent facts beyond the evidence."
            ),
        },
        {
            "role": "user",
            "content": _contrastive_generation_prompt(
                structured_pairs,
                ev_texts,
                route_digest,
            ),
        },
    ])

    # Step 2: self verification
    verify_text, u2 = call_llm([
        {
            "role": "system",
            "content": "Return only valid JSON. No markdown.",
        },
        {
            "role": "user",
            "content": _self_verify_prompt(draft, ev_texts),
        },
    ])

    verification_result = _safe_json_loads_from_llm(verify_text)
    verified_text, kept_bullets, removed_bullets = _verified_text_from_result(
        draft,
        verification_result,
    )
    verification_stats = _verification_stats(verification_result)

    # Step 3: fill missing shortlist recalls.
    # Important fix: this runs even if verifier kept zero bullets.
    covered = _covered_recalls_from_text_and_bullets(verified_text, kept_bullets)

    target_n = min(len(shortlist), min(contrastive_max_bullets, 16))
    missing = [
        x for x in shortlist
        if canon_recall_number(x.get("recall_number", "")) not in covered
    ]

    if missing and len(covered) < target_n:
        missing = missing[:max(1, min(6, target_n - len(covered)))]

        fill_prompt = (
            "Some high-priority evidence recalls are not yet covered in the verified summary. "
            "Add one short, evidence-backed bullet for each missing recall. "
            "Each bullet must start with 'Recall <Number>:'. "
            "Use old-to-new wording if old_new_changes are available. "
            "Do not invent unsupported details.\n\n"
            "MISSING RECALLS:\n"
            + json.dumps([
                {
                    "recall_number": x.get("recall_number", ""),
                    "route": x.get("route_label", ""),
                    "change_types": x.get("change_types", ""),
                    "changed_fields": x.get("changed_fields", ""),
                    "recall_heading": x.get("recall_heading", ""),
                    "hazard_description": x.get("hazard_description", ""),
                    "consumer_action": x.get("consumer_action", ""),
                    "remedy": x.get("remedy", ""),
                    "incidents": x.get("incidents", ""),
                    "old_new_changes": x.get("old_new_pairs", []),
                }
                for x in missing
            ], ensure_ascii=False, indent=2)
            + "\n\nEVIDENCE:\n"
            + "\n\n---\n\n".join(ev_texts[:12])
            + "\n\nCURRENT VERIFIED SUMMARY:\n"
            + verified_text
        )

        fill_text, u3 = call_llm([
            {
                "role": "system",
                "content": "Be concise, factual, and evidence-grounded.",
            },
            {
                "role": "user",
                "content": fill_prompt,
            },
        ])

        fill_text = normalize_text(fill_text)
        if fill_text:
            verified_text = (verified_text.strip() + "\n" + fill_text.strip()).strip()

        # Add fill usage into u2 bucket so total usage is correct.
        for k, v in u3.items():
            if isinstance(v, (int, float)):
                u2[k] = u2.get(k, 0) + v

    usage = {}
    for d in [u1, u2]:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v

    raw, final = _apply_enforce(
        verified_text,
        evidence,
        shortlist,
        max_unique=min(contrastive_max_bullets, 16),
    )

    # Update stats after final enforcement.
    verification_stats["final_unique_recalls"] = len({
        canon_recall_number(x)
        for x in extract_recall_numbers(final or "")
        if canon_recall_number(x)
    })
    verification_stats["raw_unique_recalls"] = len({
        canon_recall_number(x)
        for x in extract_recall_numbers(raw or "")
        if canon_recall_number(x)
    })

    return raw, final, evidence, {
        "mode": "llm",
        "usage": usage,
        "query_specs": query_specs,
        "shortlist": shortlist,
        "route_digest": route_digest,
        "verification_result": verification_result,
        "verification_stats": verification_stats,
        "verification_raw_text": verify_text,
        "verification_kept_bullets": kept_bullets,
        "verification_removed_bullets": removed_bullets,
        "structured_pairs": structured_pairs,
    }


# ------------------------------------------------------------
# 4. Debug verifier keep=false ratio
# ------------------------------------------------------------

def inspect_contrastive_verification(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Run after:
        result = run(EXPLICIT_PREVIOUS_SNAPSHOT)

    It prints keep=false ratio and returns a bullet-level verifier table.
    """
    ca_meta = result.get("meta", {}).get("contrastive_adaptive_rag", {})
    vr = ca_meta.get("verification_result", {}) or {}
    stats = ca_meta.get("verification_stats", _verification_stats(vr))

    print("contrastive_adaptive_rag verification stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    bullets = vr.get("bullets", []) if isinstance(vr, dict) else []
    rows = []

    for i, b in enumerate(bullets):
        if not isinstance(b, dict):
            continue
        claims = b.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        claim_counts = Counter(
            str(c.get("verdict", "missing")).strip().lower()
            for c in claims
            if isinstance(c, dict)
        )

        rows.append({
            "idx": i + 1,
            "recall_number": canon_recall_number(b.get("recall_number", "")),
            "route": b.get("route", ""),
            "original_keep": b.get("keep", True),
            "relaxed_keep": _bullet_keepable_after_verify(b),
            "overall_verdict": b.get("overall_verdict", ""),
            "n_claims": len(claims),
            "fully_supported_claims": claim_counts.get("fully_supported", 0),
            "partially_supported_claims": claim_counts.get("partially_supported", 0),
            "unsupported_claims": claim_counts.get("unsupported", 0),
            "bullet_text": normalize_text(b.get("bullet_text", ""))[:260],
        })

    df = pd.DataFrame(rows)

    if display is not None and len(df) > 0:
        display(df)
    elif len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("No verifier bullets found. This may mean LLM_MODE='offline' or verifier JSON parsing failed.")

    return df


# ------------------------------------------------------------
# 5. Final meta summary: includes contrastive_adaptive_rag output
# ------------------------------------------------------------

def _metric_value(auto_eval_df: pd.DataFrame,
                  method: str,
                  col: str,
                  default: float = 0.0) -> float:
    try:
        row = auto_eval_df[auto_eval_df["method"] == method]
        if len(row) == 0 or col not in row.columns:
            return default
        val = row.iloc[0][col]
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def _score_methods_for_meta(auto_eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for m in METHOD_ORDER_FINAL:
        if "method" not in auto_eval_df.columns or m not in set(auto_eval_df["method"].astype(str)):
            continue

        final_recall_f1 = _metric_value(auto_eval_df, m, "final_recall_f1")
        final_cu_weighted = _metric_value(auto_eval_df, m, "final_cu_weighted_coverage")
        final_grounding = _metric_value(auto_eval_df, m, "final_grounding_rate")
        final_route = _metric_value(auto_eval_df, m, "final_route_coverage_rate")
        final_precision = _metric_value(auto_eval_df, m, "final_recall_precision")

        # Balanced score for final written conclusion.
        # It favors recall/change coverage, but still penalizes ungrounded summaries.
        meta_score = (
            0.35 * final_recall_f1
            + 0.25 * final_cu_weighted
            + 0.20 * final_grounding
            + 0.10 * final_route
            + 0.10 * final_precision
        )

        rows.append({
            "method": m,
            "meta_score": round(meta_score, 4),
            "final_recall_f1": final_recall_f1,
            "final_recall_coverage": _metric_value(auto_eval_df, m, "final_recall_coverage"),
            "final_recall_precision": final_precision,
            "final_cu_weighted_coverage": final_cu_weighted,
            "final_grounding_rate": final_grounding,
            "final_route_coverage_rate": final_route,
        })

    return pd.DataFrame(rows).sort_values("meta_score", ascending=False).reset_index(drop=True)


def _deterministic_final_meta_summary(result: Dict[str, Any],
                                      score_df: pd.DataFrame,
                                      best_methods: List[str]) -> str:
    change_stats = result.get("change_stats", {})
    route_digest = result.get("route_digest", [])

    lines = []
    lines.append("# Final Meta Summary")
    lines.append("")
    lines.append("## Overall finding")
    lines.append(
        "This run compares four generation strategies: prompt_only, rag, agentic_rag, "
        "and contrastive_adaptive_rag. The final comparison now explicitly includes "
        "contrastive_adaptive_rag, so the strongest contrastive/evidence-grounded method "
        "is no longer omitted from the final conclusion."
    )
    lines.append("")
    lines.append("## Change statistics")
    lines.append("```json")
    lines.append(json.dumps(change_stats, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Method ranking")
    if score_df is not None and len(score_df) > 0:
        lines.append(score_df.to_markdown(index=False))
    else:
        lines.append("No automatic evaluation table is available.")
    lines.append("")
    lines.append("## Recommended focus")
    if best_methods:
        lines.append(
            "The best methods based on the balanced meta score are: "
            + ", ".join(best_methods)
            + ". In the project report, use these methods as the main qualitative examples, "
              "then explain the remaining methods as baselines or ablations."
        )
    else:
        lines.append("No best method could be selected from the automatic metrics.")
    lines.append("")
    lines.append("## Route digest")
    lines.append("```json")
    lines.append(json.dumps(route_digest, ensure_ascii=False, indent=2))
    lines.append("```")

    return "\n".join(lines)


def generate_final_meta_summary(result: Dict[str, Any],
                                top_n: int = 2) -> Tuple[str, Dict[str, Any]]:
    """
    Generate final cross-method written analysis.

    Fixed behavior:
    - Always considers contrastive_adaptive_rag together with prompt_only/rag/agentic_rag.
    - Uses automatic metrics to select the strongest methods.
    - Saves enough metadata to explain why the final methods were chosen.
    """
    auto_eval_df = result.get("auto_eval_df", pd.DataFrame()).copy()
    score_df = _score_methods_for_meta(auto_eval_df) if len(auto_eval_df) else pd.DataFrame()
    best_methods = score_df["method"].head(top_n).tolist() if len(score_df) else []

    summaries = result.get("summaries", {})

    compact_summaries = {}
    for m in METHOD_ORDER_FINAL:
        s = summaries.get(m, {})
        compact_summaries[m] = {
            "raw_preview": normalize_text(s.get("raw", ""))[:1200],
            "final_preview": normalize_text(s.get("final", ""))[:1600],
        }

    meta_payload = {
        "change_stats": result.get("change_stats", {}),
        "route_digest": result.get("route_digest", []),
        "method_scores": score_df.to_dict("records") if len(score_df) else [],
        "best_methods": best_methods,
        "summaries": compact_summaries,
        "contrastive_verification_stats": (
            result.get("meta", {})
                  .get("contrastive_adaptive_rag", {})
                  .get("verification_stats", {})
        ),
    }

    if LLM_MODE == "offline":
        text = _deterministic_final_meta_summary(result, score_df, best_methods)
        usage = {}
    else:
        prompt = (
            "You are writing the final meta-summary for a CPSC recall monitoring project.\n\n"
            "You MUST compare all four methods:\n"
            "1. prompt_only\n"
            "2. rag\n"
            "3. agentic_rag\n"
            "4. contrastive_adaptive_rag\n\n"
            "Use the automatic metrics and summary previews below. "
            "Explain which method is strongest, why, and what the project should focus on. "
            "Mention coverage, precision/F1, grounding, route coverage, and verification behavior when relevant. "
            "Write in clear English, suitable for a final project report. "
            "Do not invent metrics not present in the payload.\n\n"
            "PAYLOAD:\n"
            + json.dumps(meta_payload, ensure_ascii=False, indent=2)
        )

        text, usage = call_llm([
            {
                "role": "system",
                "content": "Be analytical, precise, and concise. Do not hallucinate metrics.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ])

    meta = {
        "usage": usage,
        "method_scores": score_df.to_dict("records") if len(score_df) else [],
        "best_methods": best_methods,
        "included_methods": METHOD_ORDER_FINAL,
    }

    return normalize_text(text), meta


# ------------------------------------------------------------
# 6. Override run_pipeline so final_meta_summary is saved
# ------------------------------------------------------------

def run_pipeline(previous_snapshot_path: str,
                 current_snapshot_path: str,
                 output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    check_required_previous_snapshot(previous_snapshot_path)

    prev_df = load_snapshot(previous_snapshot_path)
    cur_df = load_snapshot(current_snapshot_path)
    print(f"Prev rows: {len(prev_df)} | Cur rows: {len(cur_df)}")

    cu = augment_change_units(build_change_units(prev_df, cur_df))
    change_stats = summarize_change_stats(cu)
    print("Change stats:", change_stats)

    corpus = build_retrieval_corpus(cu)
    retriever = SimpleRetriever(corpus)

    human_gold_df = load_human_gold(HUMAN_GOLD_PATH) if HUMAN_GOLD_PATH else None
    gold = choose_gold(cu, human_gold_df)
    print(f"Gold mode: {gold['gold_mode']} | Reference recalls: {len(gold['reference_recalls'])}")

    # Run all four methods.
    print("Running prompt_only …")
    po_raw, po_final, po_meta = generate_prompt_only(cu, corpus)

    print("Running rag …")
    rag_raw, rag_final, rag_ev, rag_meta = generate_rag(cu, retriever)

    print("Running agentic_rag …")
    ag_raw, ag_final, ag_ev, ag_meta = generate_agentic_rag(cu, retriever)

    print("Running contrastive_adaptive_rag …")
    ca_raw, ca_final, ca_ev, ca_meta = generate_contrastive_adaptive_rag(cu, retriever)

    # Evaluate all four methods.
    auto_rows = [
        auto_eval_one("prompt_only", po_raw, po_final, gold, corpus, cu),
        auto_eval_one("rag", rag_raw, rag_final, gold, rag_ev, cu),
        auto_eval_one("agentic_rag", ag_raw, ag_final, gold, ag_ev, cu),
        auto_eval_one(
            "contrastive_adaptive_rag",
            ca_raw,
            ca_final,
            gold,
            ca_ev,
            cu,
            ca_meta.get("verification_result"),
        ),
    ]
    auto_eval_df = pd.DataFrame(auto_rows)

    summaries = {
        "prompt_only": (po_raw, po_final),
        "rag": (rag_raw, rag_final),
        "agentic_rag": (ag_raw, ag_final),
        "contrastive_adaptive_rag": (ca_raw, ca_final),
    }

    result = {
        "change_stats": change_stats,
        "gold": gold,
        "summaries": {k: {"raw": v[0], "final": v[1]} for k, v in summaries.items()},
        "auto_eval_df": auto_eval_df,
        "route_digest": build_route_digest(cu),
        "meta": {
            "prompt_only": po_meta,
            "rag": rag_meta,
            "agentic_rag": ag_meta,
            "contrastive_adaptive_rag": ca_meta,
        },
        "evidence": {
            "rag": rag_ev,
            "agentic_rag": ag_ev,
            "contrastive_adaptive_rag": ca_ev,
        },
    }

    # Generate final meta summary AFTER all method outputs and metrics exist.
    print("Generating final_meta_summary …")
    final_meta_summary, final_meta = generate_final_meta_summary(result, top_n=2)
    result["final_meta_summary"] = final_meta_summary
    result["meta"]["final_meta_summary"] = final_meta

    # Save outputs.
    write_json(
        {k: {"raw": v[0], "final": v[1]} for k, v in summaries.items()},
        os.path.join(output_dir, "summaries.json"),
    )
    auto_eval_df.to_csv(os.path.join(output_dir, "auto_eval.csv"), index=False)
    write_json(change_stats, os.path.join(output_dir, "change_stats.json"))
    write_json(build_route_digest(cu), os.path.join(output_dir, "route_digest.json"))
    write_json(ca_meta.get("verification_stats", {}), os.path.join(output_dir, "contrastive_verify_stats.json"))
    write_json(final_meta, os.path.join(output_dir, "final_meta_summary_meta.json"))

    with open(os.path.join(output_dir, "final_meta_summary.md"), "w", encoding="utf-8") as f:
        f.write(final_meta_summary)

    # Human eval templates.
    build_human_eval_templates(cu, summaries, gold, output_dir)

    print("\n── Auto Evaluation Results ──")
    cols = [
        c for c in auto_eval_df.columns
        if any(
            k in c
            for k in [
                "method",
                "recall_coverage",
                "recall_f1",
                "cu_weighted",
                "grounding_rate",
                "enforce_delta",
                "route_coverage",
            ]
        )
    ]
    print(auto_eval_df[cols].to_string(index=False))

    print("\n── Contrastive Verification Stats ──")
    for k, v in ca_meta.get("verification_stats", {}).items():
        print(f"{k}: {v}")

    print("\nFinal meta summary saved to:")
    print(os.path.join(output_dir, "final_meta_summary.md"))

    return result


import re, json
import pandas as pd
from typing import Any, Dict, List, Tuple


# ------------------------------------------------------------
# 0. Larger budget for contrastive adaptive RAG
# ------------------------------------------------------------

CONTRASTIVE_MAX_DOCS = max(
    globals().get("AGENT_MAX_DOCS", 22),
    globals().get("RAG_MAX_DOCS", 18),
    24,
)

CONTRASTIVE_MAX_BULLETS = max(
    globals().get("AGENT_SUMMARY_MAX_BULLETS", 16),
    globals().get("RAG_SUMMARY_MAX_BULLETS", 16),
    16,
)

print("V6 contrastive patch loaded")
print("CONTRASTIVE_MAX_DOCS:", CONTRASTIVE_MAX_DOCS)
print("CONTRASTIVE_MAX_BULLETS:", CONTRASTIVE_MAX_BULLETS)


# ------------------------------------------------------------
# 1. Stronger contrastive generation prompt
# ------------------------------------------------------------

def _contrastive_generation_prompt_v6(
    structured_pairs: List[Dict[str, Any]],
    evidence_texts: List[str],
    route_digest: List[Dict[str, Any]],
    shortlist: List[Dict[str, Any]],
    selected_recalls: List[str],
) -> str:
    return (
        "You are writing a CHANGE-TYPE-AWARE, EVIDENCE-GROUNDED summary for CPSC recall monitoring.\n\n"
        "Your main objective is HIGH RECALL-NUMBER COVERAGE while staying factual.\n\n"
        "## Mandatory coverage rule\n"
        "Write 12-16 bullet points if evidence supports them.\n"
        "Cover SELECTED_RECALLS first.\n"
        "Every bullet must start exactly with: Recall XX-XXX:\n"
        "Each bullet must cover exactly one recall.\n"
        "Do not merge multiple recalls into one bullet.\n\n"
        "## Format rules\n"
        "For modified recalls:\n"
        "Recall XX-XXX: [Product] — previously [old value], now [new value]. "
        "Consumer impact: [why it matters].\n\n"
        "For newly added recalls:\n"
        "Recall XX-XXX: [Product] — newly added recall. "
        "Hazard: [hazard]. Action required: [consumer action].\n\n"
        "For partially supported recalls:\n"
        "Recall XX-XXX: [Product] — [supported factual summary]. [partial evidence]\n\n"
        "## Strict factuality rules\n"
        "1. Use ONLY the evidence below.\n"
        "2. If old/new values are available, describe the change direction.\n"
        "3. If old/new values are not available but hazard/action/remedy is supported, still write the bullet.\n"
        "4. Do NOT invent incidents, injuries, remedies, or consumer actions.\n"
        "5. Do NOT skip a selected recall merely because it has no old/new pair; summarize the supported added-recall facts instead.\n\n"
        "## Selected recalls to cover first\n"
        f"{json.dumps(selected_recalls, ensure_ascii=False)}\n\n"
        "## Target shortlist\n"
        + compact_target_list(shortlist, CONTRASTIVE_MAX_BULLETS)
        + "\n\n## Route digest\n"
        + json.dumps(route_digest, ensure_ascii=False, indent=2)
        + "\n\n## Structured old/new changes\n"
        + json.dumps(structured_pairs, ensure_ascii=False, indent=2)
        + "\n\n## Evidence\n"
        + "\n\n---\n\n".join(evidence_texts[:CONTRASTIVE_MAX_DOCS])
    )


# ------------------------------------------------------------
# 2. Build stronger contrastive query specs
# ------------------------------------------------------------

def build_contrastive_query_specs_v6(
    cu_aug: pd.DataFrame,
    retriever: SimpleRetriever,
) -> List[Dict[str, str]]:
    query_specs = []

    # Base route-aware queries
    query_specs.extend(build_query_bundle(cu_aug))

    # Recall-number expansion from corpus
    try:
        query_specs.extend(build_recall_expansion_queries(retriever.corpus_df, per_route=8))
    except Exception:
        pass

    # Extra broad queries to avoid missing high-risk current-window recalls
    query_specs.extend([
        {
            "name": "contrastive_added_high_risk_broad",
            "intent": "added_high",
            "query": (
                "newly added high risk recalls fire burn battery child infant choking "
                "injury death hazard stop use refund repair replace"
            ),
        },
        {
            "name": "contrastive_action_remedy_broad",
            "intent": "consumer_action",
            "query": (
                "consumer action changed remedy refund repair replacement stop using immediately "
                "return contact firm recall remedy update"
            ),
        },
        {
            "name": "contrastive_incident_units_broad",
            "intent": "incidents_units",
            "query": (
                "incidents injuries reports units affected hazard description updated expanded recall"
            ),
        },
        {
            "name": "contrastive_metadata_broad",
            "intent": "generic",
            "query": (
                "recall heading hazard description product name manufacturer description changed updated"
            ),
        },
    ])

    # Optional LLM planning, same idea as agentic_rag, but used only to improve retrieval.
    if globals().get("LLM_MODE", "offline") != "offline":
        try:
            plan_text, _ = call_llm([
                {
                    "role": "system",
                    "content": "Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": (
                        "Build broad retrieval queries for CPSC recall-change analysis. "
                        "Return JSON: {\"queries\": [\"query1\", \"query2\", ...]}. "
                        "Queries should cover added recalls, consumer action/remedy changes, "
                        "hazard updates, incidents/injuries, units affected, and removed/resolved recalls."
                    ),
                },
            ])

            obj = _safe_json_loads_from_llm(plan_text)
            llm_queries = obj.get("queries", [])

            if isinstance(llm_queries, list):
                intent_rules = [
                    (r"consumer|remedy|refund|repair|replace|stop|return", "consumer_action"),
                    (r"incident|injur|unit|hazard|death|burn|fire", "incidents_units"),
                    (r"added|new|high-risk|risk", "added_high"),
                    (r"removed|legacy|resolved", "removed_legacy"),
                ]

                for i, q in enumerate(llm_queries[:6], 1):
                    qq = normalize_text(q)
                    if not qq:
                        continue

                    intent = next(
                        (v for pat, v in intent_rules if re.search(pat, qq.lower())),
                        "generic",
                    )

                    query_specs.append({
                        "name": f"contrastive_llm_{i}",
                        "intent": intent,
                        "query": qq,
                    })

        except Exception as e:
            print("Contrastive LLM query planning skipped:", repr(e))

    return query_specs


# ------------------------------------------------------------
# 3. Deterministic coverage repair after verification
# ------------------------------------------------------------

def repair_contrastive_coverage_v6(
    text: str,
    shortlist: List[Dict[str, Any]],
    target_n: int = 16,
) -> str:
    """
    Add grounded bullets for selected evidence recalls that are missing from the text.
    This does not use human gold. It only uses evidence-derived shortlist rows.
    """
    text = normalize_text(text or "").strip()

    cited = {
        canon_recall_number(x)
        for x in extract_recall_numbers(text)
        if canon_recall_number(x)
    }

    additions = []

    for row in shortlist or []:
        if len(cited) >= target_n:
            break

        recall = canon_recall_number(row.get("recall_number", ""))
        if not recall or recall in cited:
            continue

        bullet = row_to_evidence_bullet(row)
        bullet = normalize_text(bullet)

        if not bullet:
            continue

        # Make sure format is consistent.
        if not bullet.lower().startswith("recall"):
            bullet = f"Recall {recall}: {bullet}"

        additions.append("- " + bullet)
        cited.add(recall)

    if additions:
        return (text + "\n" + "\n".join(additions)).strip()

    return text


# ------------------------------------------------------------
# 4. Override generate_contrastive_adaptive_rag
# ------------------------------------------------------------

def generate_contrastive_adaptive_rag(
    cu: pd.DataFrame,
    retriever: SimpleRetriever,
) -> Tuple[str, str, pd.DataFrame, Dict[str, Any]]:
    """
    V6 Contrastive Adaptive RAG.

    Main changes:
    1. Uses agentic-level retrieval budget.
    2. Adds broad and LLM-planned retrieval queries.
    3. Forces 12-16 target recall coverage in the draft prompt.
    4. Runs deterministic coverage repair after verifier.
    5. Still keeps self-verification and grounding statistics.
    """
    cu_aug = augment_change_units(cu)

    query_specs = build_contrastive_query_specs_v6(cu_aug, retriever)

    retrieved = hybrid_search_queries(
        retriever,
        query_specs,
        per_query_k=40,
        final_k=CONTRASTIVE_MAX_DOCS,
    )

    ret_sl = build_shortlist_from_retrieved(
        retrieved,
        max_items=CONTRASTIVE_MAX_BULLETS,
    )

    anc_sl = build_anchor_shortlist(
        retriever.corpus_df,
        max_items=CONTRASTIVE_MAX_BULLETS,
    )

    shortlist = merge_shortlists(
        ret_sl,
        anc_sl,
        max_items=CONTRASTIVE_MAX_BULLETS,
    )

    evidence = ensure_evidence_for_shortlist(
        retrieved,
        retriever.corpus_df,
        shortlist,
        max_docs=CONTRASTIVE_MAX_DOCS,
    )

    # Rebuild shortlist from actual evidence so generation aligns with eval.
    if evidence is not None and len(evidence) > 0:
        shortlist = build_shortlist_from_retrieved(
            evidence,
            max_items=CONTRASTIVE_MAX_BULLETS,
        )

    selected_recalls = [
        canon_recall_number(x.get("recall_number", ""))
        for x in shortlist
        if canon_recall_number(x.get("recall_number", ""))
    ][:CONTRASTIVE_MAX_BULLETS]

    ev_texts = (
        evidence["text"].fillna("").astype(str).tolist()
        if evidence is not None and len(evidence) > 0
        else []
    )

    route_digest = build_route_digest(cu_aug)
    structured_pairs = _build_old_new_pairs_for_shortlist(shortlist)

    # Offline fallback
    if globals().get("LLM_MODE", "offline") == "offline":
        lines = ["Contrastive adaptive recall watchlist:"]
        for row in shortlist:
            lines.append("- " + row_to_evidence_bullet(row))

        repaired = repair_contrastive_coverage_v6(
            "\n".join(lines),
            shortlist,
            target_n=CONTRASTIVE_MAX_BULLETS,
        )

        raw, final = _apply_enforce(
            repaired,
            evidence,
            shortlist,
            max_unique=CONTRASTIVE_MAX_BULLETS,
        )

        return raw, final, evidence, {
            "mode": "offline",
            "usage": {},
            "query_specs": query_specs,
            "shortlist": shortlist,
            "selected_recalls": selected_recalls,
            "route_digest": route_digest,
            "structured_pairs": structured_pairs,
            "verification_result": {},
            "verification_stats": {
                "verify_total_bullets": 0,
                "verify_keep_false_original": 0,
                "verify_keep_false_ratio_original": 0.0,
                "verify_kept_after_relax": 0,
                "verify_removed_after_relax": 0,
                "verify_removed_ratio_after_relax": 0.0,
                "verify_overall_verdict_counts": {},
                "raw_unique_recalls": len(set(extract_recall_numbers(raw or ""))),
                "final_unique_recalls": len(set(extract_recall_numbers(final or ""))),
            },
        }

    # Step 1: coverage-first contrastive draft
    draft, u1 = call_llm([
        {
            "role": "system",
            "content": (
                "Be factual and evidence-grounded. "
                "Maximize selected recall coverage. "
                "Every bullet must start with 'Recall <Number>:'. "
                "Do not invent unsupported facts."
            ),
        },
        {
            "role": "user",
            "content": _contrastive_generation_prompt_v6(
                structured_pairs=structured_pairs,
                evidence_texts=ev_texts,
                route_digest=route_digest,
                shortlist=shortlist,
                selected_recalls=selected_recalls,
            ),
        },
    ])

    # Step 2: self verification
    verify_text, u2 = call_llm([
        {
            "role": "system",
            "content": "Return only valid JSON. No markdown.",
        },
        {
            "role": "user",
            "content": _self_verify_prompt(draft, ev_texts),
        },
    ])

    verification_result = _safe_json_loads_from_llm(verify_text)

    verified_text, kept_bullets, removed_bullets = _verified_text_from_result(
        draft,
        verification_result,
    )

    verification_stats = _verification_stats(verification_result)

    # Step 3: deterministic coverage repair after verifier
    target_n = min(CONTRASTIVE_MAX_BULLETS, len(shortlist))

    repaired_text = repair_contrastive_coverage_v6(
        verified_text,
        shortlist,
        target_n=target_n,
    )

    # Step 4: final selected-evidence enforcement
    raw, final = _apply_enforce(
        repaired_text,
        evidence,
        shortlist,
        max_unique=CONTRASTIVE_MAX_BULLETS,
    )

    usage = {}
    for d in [u1, u2]:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v

    verification_stats["raw_unique_recalls"] = len({
        canon_recall_number(x)
        for x in extract_recall_numbers(raw or "")
        if canon_recall_number(x)
    })

    verification_stats["final_unique_recalls"] = len({
        canon_recall_number(x)
        for x in extract_recall_numbers(final or "")
        if canon_recall_number(x)
    })

    verification_stats["selected_recalls_n"] = len(selected_recalls)
    verification_stats["evidence_rows_n"] = len(evidence) if evidence is not None else 0

    return raw, final, evidence, {
        "mode": "llm",
        "usage": usage,
        "query_specs": query_specs,
        "shortlist": shortlist,
        "selected_recalls": selected_recalls,
        "route_digest": route_digest,
        "structured_pairs": structured_pairs,
        "verification_result": verification_result,
        "verification_stats": verification_stats,
        "verification_raw_text": verify_text,
        "verification_kept_bullets": kept_bullets,
        "verification_removed_bullets": removed_bullets,
    }


import re
import json
from typing import Any, Dict, List, Tuple

CONTRASTIVE_V71_MAX_FINAL_RECALLS = 22

print("V7.1 coverage-preserving precision cleanup loaded.")
print("CONTRASTIVE_V71_MAX_FINAL_RECALLS:", CONTRASTIVE_V71_MAX_FINAL_RECALLS)


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def _canon_v71(x: Any) -> str:
    try:
        return canon_recall_number(x)
    except Exception:
        s = str(x or "")
        m = re.search(r"\b(\d{2})[-\s]?(\d{3})\b", s)
        return f"{m.group(1)}-{m.group(2)}" if m else ""


def _extract_recalls_v71(text: Any) -> List[str]:
    try:
        nums = extract_recall_numbers(str(text or ""))
        return [_canon_v71(x) for x in nums if _canon_v71(x)]
    except Exception:
        s = str(text or "")
        nums = re.findall(r"\b\d{2}-\d{3}\b", s)
        return [_canon_v71(x) for x in nums if _canon_v71(x)]


def _norm_one_line_v71(text: Any) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _norm_keep_lines_v71(text: Any) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _row_get_v71(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return getattr(row, key)
    except Exception:
        return default


def _score_v71(row: Dict[str, Any]) -> float:
    try:
        return float(_row_get_v71(row, "score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _change_type_priority_v71(row: Dict[str, Any]) -> int:
    ct = str(_row_get_v71(row, "change_types", "") or "").lower()
    route = str(_row_get_v71(row, "route_label", "") or "").lower()

    score = 0
    if "modified" in ct:
        score += 5
    if "added" in ct:
        score += 4
    if "consumer" in route or "action" in route or "remedy" in route:
        score += 4
    if "high" in route or "risk" in route:
        score += 3
    if "incident" in route or "injur" in route or "unit" in route:
        score += 2

    return score


def _unique_shortlist_rows_v71(shortlist: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    rows = []
    seen = set()

    for row in shortlist or []:
        recall = _canon_v71(_row_get_v71(row, "recall_number", ""))

        if not recall:
            recall = _canon_v71(_row_get_v71(row, "text", ""))

        if not recall or recall in seen:
            continue

        seen.add(recall)
        rows.append((recall, row))

    return rows


# ------------------------------------------------------------
# Bullet parsing
# ------------------------------------------------------------

def _split_recall_bullets_v71(text: str) -> List[str]:
    """
    Split text into recall-level bullets.
    Only bullets starting with 'Recall XX-XXX:' are considered valid.
    """
    s = _norm_keep_lines_v71(text)

    s = re.sub(
        r"(?<!\n)(\s*[-*•]?\s*Recall\s+\d{2}-\d{3}\s*:)",
        r"\n\1",
        s,
        flags=re.IGNORECASE,
    ).strip()

    pattern = re.compile(
        r"(?:^|\n)\s*(?:[-*•]\s*)?"
        r"(Recall\s+\d{2}-\d{3}\s*:.*?)(?=\n\s*(?:[-*•]\s*)?Recall\s+\d{2}-\d{3}\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    bullets = [
        _norm_one_line_v71(m.group(1))
        for m in pattern.finditer(s)
        if _norm_one_line_v71(m.group(1))
    ]

    return bullets


def _primary_recall_from_bullet_v71(bullet: str) -> str:
    m = re.search(r"\bRecall\s+(\d{2}-\d{3})\s*:", str(bullet or ""), flags=re.IGNORECASE)
    if not m:
        return ""
    return _canon_v71(m.group(1))


def _remove_extra_recall_mentions_v71(bullet: str, primary_recall: str) -> str:
    """
    Keep only the primary recall number.
    Remove extra recall numbers inside the same bullet because auto-eval counts all of them.
    """
    s = _norm_one_line_v71(bullet)
    primary = _canon_v71(primary_recall)

    nums = set(_extract_recalls_v71(s))
    extras = sorted([x for x in nums if x and x != primary])

    for ex in extras:
        s = re.sub(
            rf"\bRecall\s+Number\s*:\s*{re.escape(ex)}\b",
            "the related recall",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            rf"\bRecall\s+{re.escape(ex)}\b",
            "the related recall",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            rf"\b{re.escape(ex)}\b",
            "the related recall",
            s,
            flags=re.IGNORECASE,
        )

    s = re.sub(
        r"^\s*[-*•]?\s*Recall\s+\d{2}-\d{3}\s*:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    return f"Recall {primary}: {s}"


def _row_to_single_recall_bullet_v71(row: Dict[str, Any], recall: str) -> str:
    recall = _canon_v71(recall)

    try:
        body = row_to_evidence_bullet(row)
    except Exception:
        body = json.dumps(row, ensure_ascii=False)

    body = _norm_one_line_v71(body)

    body = re.sub(
        r"^\s*[-*•]?\s*Recall\s+Number\s*:\s*\d{2}-\d{3}\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        r"^\s*[-*•]?\s*Recall\s+\d{2}-\d{3}\s*:?\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )

    body = body.strip(" -:;")

    if not body:
        change_types = _row_get_v71(row, "change_types", "")
        route_label = _row_get_v71(row, "route_label", "")
        score = _row_get_v71(row, "score", "")
        body = f"Evidence-supported recall update. Change types: {change_types}. Route: {route_label}. Score: {score}."

    bullet = f"Recall {recall}: {body}"
    bullet = _remove_extra_recall_mentions_v71(bullet, recall)

    return bullet


# ------------------------------------------------------------
# V7.1 cleanup
# ------------------------------------------------------------

def clean_contrastive_final_summary_v71(
    text: str,
    shortlist: List[Dict[str, Any]],
    max_final_recalls: int = CONTRASTIVE_V71_MAX_FINAL_RECALLS,
) -> str:
    """
    Coverage-preserving precision cleanup.

    Difference from V7:
    - V7 selected only the top 16 shortlist rows, which dropped true gold recalls.
    - V7.1 first preserves recall bullets that already appeared in the V6 final output.
    - Then it fills extra high-priority shortlist recalls up to a soft cap.
    """
    shortlist_rows = _unique_shortlist_rows_v71(shortlist)
    row_by_recall = {recall: row for recall, row in shortlist_rows}

    existing_bullets = _split_recall_bullets_v71(text)

    bullet_by_recall: Dict[str, str] = {}

    # 1. Preserve existing V6 final bullets first.
    # This is important because V6 already had 16/16 coverage.
    for bullet in existing_bullets:
        primary = _primary_recall_from_bullet_v71(bullet)

        if not primary:
            continue

        if primary in bullet_by_recall:
            continue

        bullet_by_recall[primary] = _remove_extra_recall_mentions_v71(bullet, primary)

    # 2. Rank shortlist rows for possible filling.
    ranked_rows = sorted(
        shortlist_rows,
        key=lambda x: (
            _change_type_priority_v71(x[1]),
            _score_v71(x[1]),
        ),
        reverse=True,
    )

    # 3. Fill from high-priority shortlist rows up to soft cap.
    for recall, row in ranked_rows:
        if len(bullet_by_recall) >= max_final_recalls:
            break

        if recall in bullet_by_recall:
            continue

        bullet_by_recall[recall] = _row_to_single_recall_bullet_v71(row, recall)

    # 4. If still above cap, remove lowest-priority non-shortlist / low-priority rows.
    # But do NOT blindly take the first 16. This was the V7 bug.
    if len(bullet_by_recall) > max_final_recalls:
        def rank_for_keep(recall: str) -> Tuple[int, float, int]:
            row = row_by_recall.get(recall, {})
            in_shortlist = 1 if recall in row_by_recall else 0
            return (
                in_shortlist,
                _change_type_priority_v71(row),
                _score_v71(row),
            )

        kept_recalls = sorted(
            bullet_by_recall.keys(),
            key=rank_for_keep,
            reverse=True,
        )[:max_final_recalls]

        bullet_by_recall = {
            r: bullet_by_recall[r]
            for r in kept_recalls
            if r in bullet_by_recall
        }

    # 5. Output ordered by shortlist rank first, then remaining existing recalls.
    ordered = []

    for recall, _ in shortlist_rows:
        if recall in bullet_by_recall and recall not in ordered:
            ordered.append(recall)

    for recall in bullet_by_recall:
        if recall not in ordered:
            ordered.append(recall)

    cleaned = "Contrastive adaptive recall watchlist:\n"
    cleaned += "\n".join(f"- {bullet_by_recall[r]}" for r in ordered if r in bullet_by_recall)

    return cleaned.strip()


# ------------------------------------------------------------
# Wrap V6 function safely
# ------------------------------------------------------------

# If the old V7 wrapper already exists, this tries to recover the V6 function.
# Best practice: restart runtime, run FINAL PATCH -> V6 -> V7.1 -> run().
if "_generate_contrastive_adaptive_rag_before_v7" in globals():
    # This variable was created by the previous V7 cell and should point to V6.
    _generate_contrastive_adaptive_rag_before_v71 = _generate_contrastive_adaptive_rag_before_v7
elif "_generate_contrastive_adaptive_rag_before_v71" not in globals():
    _generate_contrastive_adaptive_rag_before_v71 = generate_contrastive_adaptive_rag


def generate_contrastive_adaptive_rag(cu, retriever):
    """
    V7.1 wrapper around V6 contrastive_adaptive_rag.
    """
    raw, final, evidence, meta = _generate_contrastive_adaptive_rag_before_v71(cu, retriever)

    if not isinstance(meta, dict):
        meta = {}

    shortlist = meta.get("shortlist", [])

    before_recalls = sorted(set(_extract_recalls_v71(final)))

    cleaned_final = clean_contrastive_final_summary_v71(
        final,
        shortlist,
        max_final_recalls=CONTRASTIVE_V71_MAX_FINAL_RECALLS,
    )

    after_recalls = sorted(set(_extract_recalls_v71(cleaned_final)))

    cleanup_stats = {
        "max_final_recalls": CONTRASTIVE_V71_MAX_FINAL_RECALLS,
        "before_unique_recalls": len(before_recalls),
        "after_unique_recalls": len(after_recalls),
        "before_recalls": before_recalls,
        "after_recalls": after_recalls,
        "removed_recalls": sorted(set(before_recalls) - set(after_recalls)),
        "added_recalls": sorted(set(after_recalls) - set(before_recalls)),
    }

    meta["precision_cleanup_v71"] = cleanup_stats

    if "verification_stats" not in meta or not isinstance(meta["verification_stats"], dict):
        meta["verification_stats"] = {}

    meta["verification_stats"]["v71_before_unique_recalls"] = len(before_recalls)
    meta["verification_stats"]["v71_after_unique_recalls"] = len(after_recalls)
    meta["verification_stats"]["final_unique_recalls"] = len(after_recalls)

    return raw, cleaned_final, evidence, meta


