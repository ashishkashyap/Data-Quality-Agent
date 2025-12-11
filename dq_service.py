import json
import os
import datetime as dt
from typing import Dict, Any, List, Tuple

import psycopg2
from psycopg2 import errors as psycopg2_errors
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query, Form
from starlette.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. FastAPI app
# ============================================================

app = FastAPI(title="NITCO DataTrust Agent - DQ Service")

# Directory where configs live, e.g. configs/sales_fact.json
DQ_CONFIG_DIR = os.getenv("DQ_CONFIG_DIR", "configs")

def get_available_tables() -> List[str]:
    """Get list of available table IDs from config files."""
    if not os.path.exists(DQ_CONFIG_DIR):
        return []
    return sorted([
        f.replace(".json", "") 
        for f in os.listdir(DQ_CONFIG_DIR) 
        if f.endswith(".json")
    ])

# ============================================================
# 2. Config & DB Connection
# ============================================================

def get_config_path(table_id: str) -> str:
    """
    Returns the full path to the JSON config for the given table_id.
    """
    fname = f"{table_id}.json"
    path = os.path.join(DQ_CONFIG_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Config file not found for table_id={table_id} at {path}"
        )
    return path


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_connection():
    """
    Connect to Supabase Postgres via environment variables:

    SUPABASE_HOST
    SUPABASE_DB
    SUPABASE_USER
    SUPABASE_PASSWORD
    SUPABASE_PORT
    """
    import ssl

    host = os.getenv("SUPABASE_HOST")
    dbname = os.getenv("SUPABASE_DB", "postgres")
    user = os.getenv("SUPABASE_USER", "postgres")
    password = os.getenv("SUPABASE_PASSWORD")
    port = int(os.getenv("SUPABASE_PORT", "5432"))

    if not host or not password:
        raise HTTPException(
            status_code=500,
            detail="Database connection not configured. Please set SUPABASE_HOST and SUPABASE_PASSWORD."
        )

    # Supabase generally requires SSL
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port,
        sslmode="require",
        sslrootcert=None,
    )
    return conn


def quote_identifier(col: str) -> str:
    """
    Very small helper to quote identifiers safely.
    """
    col = col.replace('"', '""')
    return f'"{col}"'

# ============================================================
# 3. Historical row count helper (for row count rules)
# ============================================================

def get_historical_row_counts(conn, cfg: Dict[str, Any], days: int = 7) -> List[Tuple[dt.date, int]]:
    """
    Returns [(date, row_count), ...] for the last `days` worth of data
    based on the business_date_column.
    """
    table = cfg["db_table_name"]
    biz_col = cfg["business_date_column"]
    cur = conn.cursor()
    cur.execute(f"""
        SELECT {quote_identifier(biz_col)}::date AS d, COUNT(*) 
        FROM {table}
        WHERE {quote_identifier(biz_col)} >= CURRENT_DATE - INTERVAL %s
        GROUP BY d
        ORDER BY d
    """, (f"{days} day",))
    rows = cur.fetchall()
    return [(r[0], r[1]) for r in rows]

# ============================================================
# 4. Window filters & base metrics
# ============================================================

def build_window_filter(biz_col_quoted: str, window: str) -> str:
    """
    Build a WHERE clause for different time windows based on the business date column.

    window:
      - "today"    -> last 1 day (current behavior)
      - "prev_day" -> 1-day slice ending yesterday
      - "last_7d"  -> last 7 days
      - "last_30d" -> last 30 days
    """
    if window == "today":
        return f"{biz_col_quoted} >= (CURRENT_DATE - INTERVAL '1 day')"
    elif window == "prev_day":
        return (
            f"{biz_col_quoted} >= (CURRENT_DATE - INTERVAL '2 day') "
            f"AND {biz_col_quoted} < (CURRENT_DATE - INTERVAL '1 day')"
        )
    elif window == "last_7d":
        return f"{biz_col_quoted} >= (CURRENT_DATE - INTERVAL '7 day')"
    elif window == "last_30d":
        return f"{biz_col_quoted} >= (CURRENT_DATE - INTERVAL '30 day')"
    else:
        raise ValueError(f"Unknown window: {window}")


def get_total_row_count(conn, cfg: Dict[str, Any]) -> int:
    """Get total row count from the entire table."""
    table = cfg["db_table_name"]
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]


def get_metrics_for_window(conn, cfg: Dict[str, Any], where_clause: str) -> Dict[str, Any]:
    """
    Generic metrics for a given WHERE clause:
    - row_count
    - max_business_date (overall)
    - null % per critical column in that window
    """
    table = cfg["db_table_name"]
    biz_col = cfg["business_date_column"]
    critical_cols = cfg["critical_columns"]

    cur = conn.cursor()

    # Row count in this window
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_clause}")
    row_count = cur.fetchone()[0]

    # Max business date (overall, not window-limited)
    cur.execute(f"SELECT MAX({quote_identifier(biz_col)}) FROM {table}")
    max_date = cur.fetchone()[0]

    # Null % per critical column in this window
    null_pcts: Dict[str, Any] = {}
    for col in critical_cols:
        savepoint_name = f"sp_{hash(col) % 1000000}"
        try:
            cur.execute(f"SAVEPOINT {savepoint_name}")
            quoted_col = quote_identifier(col)
            cur.execute(f"""
                SELECT 
                    (SUM(CASE WHEN {quoted_col} IS NULL THEN 1 ELSE 0 END)::float 
                     / NULLIF(COUNT(*),0)) * 100.0
                FROM {table}
                WHERE {where_clause}
            """)
            val = cur.fetchone()[0]
            null_pcts[col] = float(val or 0.0)
            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        except (psycopg2.ProgrammingError, psycopg2.OperationalError) as e:
            try:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            except Exception:
                try:
                    conn.rollback()
                    # re-run basic queries if needed
                    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_clause}")
                    row_count = cur.fetchone()[0]
                    cur.execute(f"SELECT MAX({quote_identifier(biz_col)}) FROM {table}")
                    max_date = cur.fetchone()[0]
                except Exception:
                    pass
            print(f"Warning: Error checking column '{col}': {e}. Skipping.")
            null_pcts[col] = None

    return {
        "row_count": row_count,
        "max_business_date": max_date.isoformat() if max_date else None,
        "null_pcts": null_pcts
    }


def get_last_day_filter(biz_col: str) -> str:
    """
    Backwards-compatible wrapper: 'today' window.
    """
    return build_window_filter(biz_col, "today")


def get_current_metrics(conn, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Metrics for the 'today' window (last 1 day), used for rule evaluation.
    """
    biz_col = cfg["business_date_column"]
    where_clause = build_window_filter(quote_identifier(biz_col), "today")
    window_metrics = get_metrics_for_window(conn, cfg, where_clause)

    return {
        "row_count_last_day": window_metrics["row_count"],
        "max_business_date": window_metrics["max_business_date"],
        "null_pcts_last_day": window_metrics["null_pcts"]
    }

# ============================================================
# 5. Uniqueness, boolean checks, FK checks
# ============================================================

def compute_uniqueness(conn, cfg: Dict[str, Any], where_clause: str | None = None) -> Dict[str, Any]:
    table = cfg["db_table_name"]
    biz_col = cfg["business_date_column"]
    pk_cols = cfg.get("primary_keys", [])
    if where_clause is None:
        where_clause = get_last_day_filter(quote_identifier(biz_col))

    if not pk_cols:
        return {"duplicate_pct": 0.0}

    pk_expr = ", ".join([quote_identifier(col) for col in pk_cols])
    cur = conn.cursor()
    cur.execute(f"""
        SELECT COUNT(*) AS total,
               COUNT(DISTINCT ({pk_expr})) AS distinct_cnt
        FROM {table}
        WHERE {where_clause}
    """)
    total, distinct_cnt = cur.fetchone()
    dup_pct = 0.0
    if total and distinct_cnt is not None:
        dup_pct = max(0.0, (total - distinct_cnt) * 100.0 / total)

    return {
        "total_rows": total,
        "distinct_rows": distinct_cnt,
        "duplicate_pct": dup_pct
    }


def compute_boolean_checks(
    conn,
    table: str,
    biz_col: str,
    checks: Dict[str, str],
    where_clause: str | None = None
) -> Dict[str, float]:
    """
    Generic % passing for each boolean expression over a given window.
    """
    if not checks:
        return {}

    if where_clause is None:
        where_clause = get_last_day_filter(quote_identifier(biz_col))

    results: Dict[str, float] = {}
    cur = conn.cursor()
    for name, expr in checks.items():
        cur.execute(f"""
            SELECT 
                100.0 * SUM(CASE WHEN {expr} THEN 1 ELSE 0 END)::float 
                / NULLIF(COUNT(*),0)
            FROM {table}
            WHERE {where_clause}
        """)
        val = cur.fetchone()[0]
        results[name] = float(val or 0.0)
    return results


def compute_fk_integrity(
    conn,
    table: str,
    biz_col: str,
    fk_constraints: Dict[str, Dict[str, str]],
    where_clause: str | None = None
) -> Dict[str, float]:
    """
    For each FK constraint, compute % of rows whose FK has a match in ref table
    over a given window.
    """
    if not fk_constraints:
        return {}

    if where_clause is None:
        where_clause = get_last_day_filter(quote_identifier(biz_col))

    results: Dict[str, float] = {}
    cur = conn.cursor()

    for name, fk in fk_constraints.items():
        fk_col = fk["fk_column"]
        ref_table = fk["ref_table"]
        ref_col = fk["ref_column"]

        cur.execute(f"""
            SELECT 
                100.0 * SUM(
                    CASE WHEN EXISTS (
                        SELECT 1 FROM {ref_table} r 
                        WHERE r.{ref_col} = f.{fk_col}
                    ) THEN 1 ELSE 0 END
                )::float / NULLIF(COUNT(*),0)
            FROM {table} f
            WHERE {where_clause}
        """)
        val = cur.fetchone()[0]
        results[name] = float(val or 0.0)

    return results

# ============================================================
# 6. DQ profile computation for a window
# ============================================================

def compute_dq_profile(
    conn,
    cfg: Dict[str, Any],
    base_metrics: Dict[str, Any],
    where_clause: str | None = None
) -> Dict[str, Any]:
    table = cfg["db_table_name"]
    biz_col = cfg["business_date_column"]
    dq_cfg = cfg.get("dq_checks", {})

    # Uniqueness
    uniq = compute_uniqueness(conn, cfg, where_clause=where_clause)

    # Accuracy / Validity / Consistency / Integrity
    accuracy_checks = dq_cfg.get("accuracy_checks", {})
    validity_checks = dq_cfg.get("validity_checks", {})
    consistency_checks = dq_cfg.get("consistency_checks", {})
    fk_constraints = dq_cfg.get("fk_constraints", {})

    accuracy_results = compute_boolean_checks(conn, table, biz_col, accuracy_checks, where_clause=where_clause)
    validity_results = compute_boolean_checks(conn, table, biz_col, validity_checks, where_clause=where_clause)
    consistency_results = compute_boolean_checks(conn, table, biz_col, consistency_checks, where_clause=where_clause)
    integrity_results = compute_fk_integrity(conn, table, biz_col, fk_constraints, where_clause=where_clause)

    # Completeness: based on null pct of critical columns
    null_pcts = base_metrics["null_pcts_last_day"]
    if null_pcts:
        valid_null_pcts = {k: v for k, v in null_pcts.items() if v is not None}
        if valid_null_pcts:
            completeness_score = max(
                0.0,
                100.0 - sum(valid_null_pcts.values()) / len(valid_null_pcts)
            )
        else:
            completeness_score = 100.0
    else:
        completeness_score = 100.0

    # Uniqueness score
    uniq_dup_pct = uniq.get("duplicate_pct", 0.0)
    uniqueness_score = max(0.0, 100.0 - uniq_dup_pct)

    def avg_pct(d: Dict[str, float]) -> float:
        return sum(d.values()) / len(d) if d else 100.0

    accuracy_score = avg_pct(accuracy_results)
    validity_score = avg_pct(validity_results)
    consistency_score = avg_pct(consistency_results)
    integrity_score = avg_pct(integrity_results)

    overall = round(
        (completeness_score + uniqueness_score + accuracy_score +
         validity_score + consistency_score + integrity_score) / 6.0,
        2
    )

    return {
        "snapshot_date": dt.date.today().isoformat(),
        "overall_score": overall,
        "dimensions": {
            "completeness": {
                "score": round(completeness_score, 2),
                "null_pcts": null_pcts
            },
            "uniqueness": {
                "score": round(uniqueness_score, 2),
                "duplicate_pct": uniq_dup_pct
            },
            "accuracy": {
                "score": round(accuracy_score, 2),
                "checks": accuracy_results
            },
            "validity": {
                "score": round(validity_score, 2),
                "checks": validity_results
            },
            "consistency": {
                "score": round(consistency_score, 2),
                "checks": consistency_results
            },
            "integrity": {
                "score": round(integrity_score, 2),
                "fk_checks": integrity_results
            }
        }
    }


def compute_profile_for_window(conn, cfg: Dict[str, Any], window: str) -> Dict[str, Any]:
    """
    Build a full DQ profile for a given time window:
      - 'today'
      - 'prev_day'
      - 'last_7d'
      - 'last_30d'
    """
    biz_col = cfg["business_date_column"]
    where_clause = build_window_filter(quote_identifier(biz_col), window)
    window_metrics = get_metrics_for_window(conn, cfg, where_clause)

    base_metrics_for_profile = {
        "row_count_last_day": window_metrics["row_count"],
        "max_business_date": window_metrics["max_business_date"],
        "null_pcts_last_day": window_metrics["null_pcts"]
    }

    return compute_dq_profile(conn, cfg, base_metrics_for_profile, where_clause=where_clause)

# ============================================================
# 7. DQ profiles storage (optional, for audit)
# ============================================================

def ensure_dq_profiles_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.dq_profiles (
            table_id       text NOT NULL,
            snapshot_date  date NOT NULL,
            profile        jsonb NOT NULL,
            PRIMARY KEY (table_id, snapshot_date)
        )
    """)
    conn.commit()


def save_profile(conn, table_id: str, profile: Dict[str, Any]):
    cur = conn.cursor()
    snapshot_date = profile.get("snapshot_date", dt.date.today().isoformat())
    cur.execute("""
        INSERT INTO public.dq_profiles(table_id, snapshot_date, profile)
        VALUES (%s, %s, %s)
        ON CONFLICT (table_id, snapshot_date)
        DO UPDATE SET profile = EXCLUDED.profile
    """, (table_id, snapshot_date, json.dumps(profile)))
    conn.commit()

# ============================================================
# 8. Rule evaluation
# ============================================================

def evaluate_rules(
    cfg: Dict[str, Any],
    base_metrics: Dict[str, Any],
    hist_counts: List[Tuple[dt.date, int]]
) -> Dict[str, Any]:
    """
    Evaluate generic rules from config["rules"].
    Returns severity + rule_results (list).
    """
    rules_cfg = cfg.get("rules", {})
    rule_results: List[Dict[str, Any]] = []

    # Helper to escalate severity
    severity_order = ["info", "low", "medium", "high"]
    current_severity_idx = 0

    def bump_severity(level: str):
        nonlocal current_severity_idx
        try:
            idx = severity_order.index(level)
        except ValueError:
            return
        current_severity_idx = max(current_severity_idx, idx)

    # Row count rule
    row_count = base_metrics.get("row_count_last_day", 0)
    if "min_row_count" in rules_cfg:
        min_row = rules_cfg["min_row_count"]
        if row_count < min_row:
            rule_results.append({
                "rule": "min_row_count",
                "status": "FAIL",
                "severity": "high",
                "detail": f"Row count last day {row_count} < minimum {min_row}"
            })
            bump_severity("high")
        else:
            rule_results.append({
                "rule": "min_row_count",
                "status": "PASS",
                "severity": "info",
                "detail": f"Row count last day {row_count} >= minimum {min_row}"
            })

    # Row count vs 7d average rule
    if "row_count_vs_7d_percent_drop_alert" in rules_cfg and hist_counts:
        # hist_counts is [(date, count), ...] for last 7 days
        # compute average excluding today if present
        today = dt.date.today()
        vals = [c for d, c in hist_counts if d < today]
        avg_7d = sum(vals) / len(vals) if vals else 0
        if avg_7d > 0:
            drop_pct = (avg_7d - row_count) * 100.0 / avg_7d
        else:
            drop_pct = 0.0
        threshold = rules_cfg["row_count_vs_7d_percent_drop_alert"]
        if drop_pct > threshold:
            rule_results.append({
                "rule": "row_count_vs_7d_average",
                "status": "FAIL",
                "severity": "medium",
                "detail": f"Row count last day {row_count}, 7d avg {avg_7d:.1f}, drop {drop_pct:.1f}% > threshold {threshold}%"
            })
            bump_severity("medium")
        else:
            rule_results.append({
                "rule": "row_count_vs_7d_average",
                "status": "PASS",
                "severity": "info",
                "detail": f"Row count last day {row_count}, 7d avg {avg_7d:.1f}, drop {drop_pct:.1f}% <= threshold {threshold}%"
            })

    # Freshness rule
    max_date_str = base_metrics.get("max_business_date")
    if "freshness_lag_hours" in rules_cfg and max_date_str:
        freshness_hours = rules_cfg["freshness_lag_hours"]
        try:
            max_date = dt.datetime.fromisoformat(max_date_str)
        except Exception:
            # if it's a date only, treat as midnight
            try:
                max_date = dt.datetime.fromisoformat(max_date_str + "T00:00:00")
            except Exception:
                max_date = None
        if max_date:
            # Ensure both datetimes are timezone-aware or both are naive
            # If max_date is timezone-aware, make utcnow aware too
            if max_date.tzinfo is not None:
                # max_date is timezone-aware, use timezone-aware utcnow
                from datetime import timezone
                now = dt.datetime.now(timezone.utc)
            else:
                # max_date is naive, use naive utcnow
                now = dt.datetime.utcnow()
            lag_hours = (now - max_date).total_seconds() / 3600.0
            if lag_hours > freshness_hours:
                rule_results.append({
                    "rule": "freshness_lag_hours",
                    "status": "FAIL",
                    "severity": "medium",
                    "detail": f"Data max business_date {max_date_str} is {lag_hours:.1f} hours old (> {freshness_hours}h)"
                })
                bump_severity("medium")
            else:
                rule_results.append({
                    "rule": "freshness_lag_hours",
                    "status": "PASS",
                    "severity": "info",
                    "detail": f"Data max business_date {max_date_str} is {lag_hours:.1f} hours old (<= {freshness_hours}h)"
                })

    # Max null % per column
    max_null_cfg = rules_cfg.get("max_null_pct", {})
    null_pcts = base_metrics.get("null_pcts_last_day", {}) or {}
    for col, limit in max_null_cfg.items():
        val = null_pcts.get(col)
        if val is None:
            # skip if no data
            continue
        if val > limit:
            rule_results.append({
                "rule": f"max_null_pct_{col}",
                "status": "FAIL",
                "severity": "medium",
                "detail": f"Null % for column '{col}' is {val:.2f}% > limit {limit}%"
            })
            bump_severity("medium")
        else:
            rule_results.append({
                "rule": f"max_null_pct_{col}",
                "status": "PASS",
                "severity": "info",
                "detail": f"Null % for column '{col}' is {val:.2f}% <= limit {limit}%"
            })

    severity = severity_order[current_severity_idx]
    return {
        "severity": severity,
        "metrics": base_metrics,
        "rule_results": rule_results
    }

# ============================================================
# 9. LLM summary
# ============================================================

def generate_llm_summary(
    cfg: Dict[str, Any],
    eval_result: Dict[str, Any],
    dq_profile: Dict[str, Any],
    trends: Dict[str, Any]
) -> str:
    table_id = cfg["table_id"]
    severity = eval_result["severity"]
    metrics = eval_result["metrics"]
    rules = eval_result["rule_results"]
    dimensions = dq_profile["dimensions"]

    rules_text = "\n".join(
        [f"- {r['rule']}: {r['status']} ({r['detail']})" for r in rules]
    )

    # Flatten trends for prompt readability
    trend_lines = []
    for dim, vals in trends.items():
        trend_lines.append(
            f"{dim}: today={vals['today']:.2f}, prev_day={vals['prev_day']:.2f}, "
            f"7d_avg={vals['avg_7d']:.2f}, 30d_avg={vals['avg_30d']:.2f}"
        )
    trends_text = "\n".join(trend_lines)

    dim_scores = {dim: dimensions[dim]["score"] for dim in dimensions}

    prompt = f"""
You are a senior data quality engineer.

A data quality agent just ran on table '{table_id}' in Supabase.

Overall rule-based severity: {severity}

Last-day base metrics:
- Row count (last day): {metrics['row_count_last_day']}
- Max business date in table: {metrics['max_business_date']}
- Null percentages (last day): {metrics['null_pcts_last_day']}

Dimension scores for last day (0-100, higher is better):
{dim_scores}

Rule evaluations:
{rules_text}

Historical trends per dimension (today vs previous day, 7d avg, 30d avg):
{trends_text}

TASK:
1. Produce a Markdown table of the dimension trends with columns:
   | Dimension | Today | Prev Day | 7d Avg | 30d Avg | Trend |
   Use arrows (↑, ↓, →) and short commentary in the Trend column.
2. Write a concise narrative (5–8 sentences) explaining:
   - Where data quality is improving
   - Where it is degrading
   - Which issues are most risky to the business
   - What short list of actions (3–5 bullets) the team should take next.
3. Assume the reader is a business-savvy data leader, not a low-level engineer.
Keep it concise but insightful. Use clear, non-technical language.
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: simple heuristic summary if no API key configured
        return (
            "LLM summary is not available (OPENAI_API_KEY is not set).\n"
            f"Overall severity: {severity}. Dimension scores: {dim_scores}. "
            "Please review the rule evaluations and trends above."
        )

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            "Error calling OpenAI for summary: " + str(e) +
            f"\nFallback summary: severity={severity}, dimension scores={dim_scores}."
        )

# ============================================================
# 10. Core orchestration function
# ============================================================

def run_dq_check_for_table(table_id: str) -> Dict[str, Any]:
    config_path = get_config_path(table_id)
    cfg = load_config(config_path)
    conn = get_connection()

    try:
        # 1) Today metrics (used for rules & severity)
        base_metrics_today = get_current_metrics(conn, cfg)
        # Get total row count for the entire table
        total_row_count = get_total_row_count(conn, cfg)
        base_metrics_today["total_row_count"] = total_row_count
        hist_counts = get_historical_row_counts(conn, cfg, days=7)
        eval_result = evaluate_rules(cfg, base_metrics_today, hist_counts)

        # 2) DQ profiles for different time windows
        dq_today = compute_profile_for_window(conn, cfg, "today")
        dq_prev_day = compute_profile_for_window(conn, cfg, "prev_day")
        dq_last_7d = compute_profile_for_window(conn, cfg, "last_7d")
        dq_last_30d = compute_profile_for_window(conn, cfg, "last_30d")

        # (Optional) still store today's profile for audit/history
        ensure_dq_profiles_table(conn)
        save_profile(conn, cfg["table_id"], dq_today)

        # 3) Build trends directly from these window profiles
        trends: Dict[str, Any] = {}
        for dim in dq_today["dimensions"].keys():
            trends[dim] = {
                "today": dq_today["dimensions"][dim]["score"],
                "prev_day": dq_prev_day["dimensions"][dim]["score"],
                "avg_7d": dq_last_7d["dimensions"][dim]["score"],
                "avg_30d": dq_last_30d["dimensions"][dim]["score"]
            }

        # 4) LLM summary using the new trend structure
        llm_summary = generate_llm_summary(cfg, eval_result, dq_today, trends)

        return {
            "table_id": cfg["table_id"],
            "snapshot_date": dq_today["snapshot_date"],
            "severity": eval_result["severity"],
            "metrics": base_metrics_today,
            "dq_profile": dq_today,
            "trends": trends,
            "rule_results": eval_result["rule_results"],
            "llm_summary_markdown": llm_summary
        }
    finally:
        conn.close()

# ============================================================
# 10. Format results for display
# ============================================================

def format_markdown_to_html(markdown_text: str) -> str:
    """Convert markdown text to HTML with proper formatting."""
    if not markdown_text:
        return "<p><em>No analysis available.</em></p>"
    
    import re
    
    html = markdown_text
    
    # First, protect code blocks and tables from other processing
    code_blocks = []
    def protect_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"
    
    html = re.sub(r'```[\s\S]*?```', protect_code, html)
    
    # Handle markdown tables (more robust pattern)
    table_pattern = r'\|(.+)\|\s*\n\|[-\s\|:]+\|\s*\n((?:\|.+\|\s*\n?)+)'
    def replace_table(match):
        header_row = match.group(1)
        data_rows = match.group(2)
        
        headers = [h.strip() for h in header_row.split('|') if h.strip()]
        rows = []
        for row in data_rows.strip().split('\n'):
            if row.strip() and '|' in row:
                cells = [c.strip() for c in row.split('|') if c.strip() or c == '']
                # Remove empty first/last if they exist
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                if cells:
                    rows.append(cells)
        
        if not headers or not rows:
            return match.group(0)  # Return original if parsing failed
        
        html_table = '<table class="markdown-table">\n<thead>\n<tr>'
        for header in headers:
            html_table += f'<th>{header}</th>'
        html_table += '</tr>\n</thead>\n<tbody>'
        for row in rows:
            html_table += '<tr>'
            for i, cell in enumerate(row):
                # Pad with empty cells if needed
                if i >= len(headers):
                    break
                html_table += f'<td>{cell}</td>'
            # Add empty cells if row is shorter than headers
            while len([c for c in row if c]) < len(headers):
                html_table += '<td></td>'
            html_table += '</tr>'
        html_table += '</tbody>\n</table>'
        return html_table
    
    html = re.sub(table_pattern, replace_table, html, flags=re.MULTILINE)
    
    # Convert headers (order matters - do ## before #)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2 class="summary-h2">\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1 class="summary-h1">\1</h1>', html, flags=re.MULTILINE)
    
    # Convert bold (do before italic to avoid conflicts)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
    
    # Convert italic (avoid matching bold)
    html = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', html)
    
    # Convert bullet lists (handle both - and *)
    lines = html.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Check for list item
        if re.match(r'^[-*]\s+', stripped):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            content = re.sub(r'^[-*]\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        elif re.match(r'^\d+\.\s+', stripped):
            if not in_list:
                result_lines.append('<ol>')
                in_list = True
            content = re.sub(r'^\d+\.\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                result_lines.append('</ul>' if '<ol>' not in '\n'.join(result_lines[-10:]) else '</ol>')
                in_list = False
            result_lines.append(line)
    
    if in_list:
        result_lines.append('</ul>')
    
    html = '\n'.join(result_lines)
    
    # Restore code blocks
    for i, code_block in enumerate(code_blocks):
        html = html.replace(f"__CODE_BLOCK_{i}__", f'<pre><code>{code_block}</code></pre>')
    
    # Convert line breaks to paragraphs (but preserve lists and tables)
    paragraphs = html.split('\n\n')
    formatted_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para:
            # Don't wrap if it's already HTML tags
            if not (para.startswith('<') and para.endswith('>')):
                # Check if it contains block elements
                if '<ul>' in para or '<ol>' in para or '<table' in para or '<h' in para or '<pre>' in para:
                    formatted_paragraphs.append(para)
                else:
                    formatted_paragraphs.append(f'<p>{para}</p>')
            else:
                formatted_paragraphs.append(para)
    
    html = '\n'.join(formatted_paragraphs)
    
    # Clean up any double paragraphs
    html = re.sub(r'</p>\s*<p>', '\n', html)
    
    return html


def format_dq_results_html(result: Dict[str, Any]) -> str:
    """Format DQ check results as a professional, client-ready HTML page."""
    
    severity = result.get("severity", "UNKNOWN")
    severity_config = {
        "CRITICAL": {"color": "#e74c3c", "bg": "#fee", "icon": "🔴"},
        "HIGH": {"color": "#e67e22", "bg": "#fff4e6", "icon": "🟠"},
        "MEDIUM": {"color": "#f39c12", "bg": "#fffbf0", "icon": "🟡"},
        "LOW": {"color": "#3498db", "bg": "#ebf5fb", "icon": "🔵"},
        "OK": {"color": "#27ae60", "bg": "#eafaf1", "icon": "🟢"},
        "INFO": {"color": "#95a5a6", "bg": "#f8f9fa", "icon": "ℹ️"}
    }.get(severity.upper(), {"color": "#95a5a6", "bg": "#f8f9fa", "icon": "ℹ️"})
    
    table_id = result.get("table_id", "N/A")
    snapshot_date = result.get("snapshot_date", "N/A")
    metrics = result.get("metrics", {})
    dq_profile = result.get("dq_profile", {})
    trends = result.get("trends", {})
    rule_results = result.get("rule_results", [])
    llm_summary = result.get("llm_summary_markdown", "")
    
    dimensions = dq_profile.get("dimensions", {})
    
    # Format markdown summary
    formatted_summary = format_markdown_to_html(llm_summary)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Quality Report - {table_id}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 40px 20px;
                color: #2c3e50;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                background: white;
                border-radius: 16px;
                padding: 40px;
                margin-bottom: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                border-top: 5px solid {severity_config["color"]};
            }}
            .header-top {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 25px;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .header-title {{
                flex: 1;
            }}
            .header h1 {{
                font-size: 36px;
                font-weight: 700;
                color: #1a202c;
                margin-bottom: 8px;
                letter-spacing: -0.5px;
            }}
            .header-subtitle {{
                font-size: 16px;
                color: #718096;
                font-weight: 500;
            }}
            .severity-badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 24px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
                background: {severity_config["bg"]};
                color: {severity_config["color"]};
                border: 2px solid {severity_config["color"]};
            }}
            .header-meta {{
                display: flex;
                gap: 30px;
                flex-wrap: wrap;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
            }}
            .meta-item {{
                display: flex;
                flex-direction: column;
            }}
            .meta-label {{
                font-size: 12px;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
                margin-bottom: 4px;
            }}
            .meta-value {{
                font-size: 18px;
                font-weight: 700;
                color: #1a202c;
            }}
            .section {{
                background: white;
                border-radius: 16px;
                padding: 35px;
                margin-bottom: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .section:hover {{
                transform: translateY(-2px);
                box-shadow: 0 15px 50px rgba(0,0,0,0.15);
            }}
            .section-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 3px solid #e2e8f0;
            }}
            .section h2 {{
                font-size: 24px;
                font-weight: 700;
                color: #1a202c;
                margin: 0;
            }}
            .section-icon {{
                font-size: 28px;
            }}
            table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                margin-top: 20px;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            thead {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            th {{
                padding: 16px 20px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: white;
            }}
            tbody tr {{
                transition: background-color 0.2s;
            }}
            tbody tr:hover {{
                background-color: #f7fafc;
            }}
            tbody tr:last-child td {{
                border-bottom: none;
            }}
            td {{
                padding: 16px 20px;
                border-bottom: 1px solid #e2e8f0;
                font-size: 14px;
            }}
            .score-good {{
                color: #27ae60;
                font-weight: 700;
                font-size: 15px;
            }}
            .score-warning {{
                color: #f39c12;
                font-weight: 700;
                font-size: 15px;
            }}
            .score-bad {{
                color: #e74c3c;
                font-weight: 700;
                font-size: 15px;
            }}
            .status-pass {{
                color: #27ae60;
                font-weight: 600;
            }}
            .status-fail {{
                color: #e74c3c;
                font-weight: 600;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                padding: 25px;
                border-radius: 12px;
                border-left: 4px solid #667eea;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .summary-card:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            }}
            .summary-label {{
                font-size: 12px;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 600;
                margin-bottom: 8px;
            }}
            .summary-value {{
                font-size: 32px;
                font-weight: 700;
                color: #1a202c;
                line-height: 1.2;
            }}
            .summary-unit {{
                font-size: 14px;
                color: #718096;
                font-weight: 500;
                margin-left: 4px;
            }}
            .ai-summary {{
                background: linear-gradient(135deg, #f7fafc 0%, #ffffff 100%);
                padding: 35px;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }}
            .ai-summary h1, .ai-summary h2, .ai-summary h3 {{
                color: #1a202c;
                margin-top: 25px;
                margin-bottom: 15px;
                font-weight: 700;
            }}
            .ai-summary h1 {{
                font-size: 28px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            .ai-summary h2 {{
                font-size: 22px;
                color: #667eea;
            }}
            .ai-summary h3 {{
                font-size: 18px;
                color: #4a5568;
            }}
            .ai-summary p {{
                margin-bottom: 16px;
                color: #4a5568;
                line-height: 1.8;
                font-size: 15px;
            }}
            .ai-summary ul, .ai-summary ol {{
                margin: 20px 0;
                padding-left: 30px;
            }}
            .ai-summary li {{
                margin-bottom: 10px;
                color: #4a5568;
                line-height: 1.7;
            }}
            .ai-summary strong {{
                color: #1a202c;
                font-weight: 600;
            }}
            .markdown-table {{
                width: 100%;
                margin: 25px 0;
                border-collapse: separate;
                border-spacing: 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .markdown-table th {{
                background: #667eea;
                color: white;
                padding: 12px 16px;
                font-weight: 600;
                font-size: 13px;
            }}
            .markdown-table td {{
                padding: 12px 16px;
                border-bottom: 1px solid #e2e8f0;
            }}
            .markdown-table tr:last-child td {{
                border-bottom: none;
            }}
            .trend-up {{
                color: #27ae60;
                font-weight: 600;
            }}
            .trend-down {{
                color: #e74c3c;
                font-weight: 600;
            }}
            .trend-stable {{
                color: #718096;
                font-weight: 500;
            }}
            .chart-container {{
                position: relative;
                height: 300px;
                margin: 25px 0;
                padding: 20px;
                background: #f7fafc;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 25px;
                margin-top: 25px;
            }}
            .chart-card {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .chart-title {{
                font-size: 16px;
                font-weight: 600;
                color: #1a202c;
                margin-bottom: 15px;
                text-align: center;
            }}
            .back-button {{
                position: fixed;
                top: 20px;
                left: 20px;
                background: white;
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                cursor: pointer;
                font-weight: 600;
                color: #667eea;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                z-index: 1000;
                font-family: inherit;
                text-decoration: none;
                display: inline-block;
            }}
            .back-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            }}
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                .section {{
                    page-break-inside: avoid;
                }}
                .back-button {{
                    display: none;
                }}
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    </head>
    <body>
        <a href="/" class="back-button">← Back to Selection</a>
        <div class="container">
            <div class="header">
                <div class="header-top">
                    <div class="header-title">
                        <h1>Data Quality Assessment Report</h1>
                        <div class="header-subtitle">Comprehensive analysis and monitoring dashboard</div>
                    </div>
                    <div class="severity-badge">
                        <span>{severity_config["icon"]}</span>
                        <span>Status: {severity}</span>
                    </div>
                </div>
                <div class="header-meta">
                    <div class="meta-item">
                        <div class="meta-label">Table</div>
                        <div class="meta-value">{table_id}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Snapshot Date</div>
                        <div class="meta-value">{snapshot_date}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Overall Score</div>
                        <div class="meta-value">{dq_profile.get("overall_score", 0):.1f}</div>
                    </div>
                </div>
            </div>
    """
    
    # Summary Metrics
    row_count = metrics.get("row_count_last_day", 0)
    total_row_count = metrics.get("total_row_count", 0)
    max_date = metrics.get("max_business_date", "N/A")
    overall_score = dq_profile.get("overall_score", 0)
    
    html += f"""
            <div class="section">
                <div class="section-header">
                    <span class="section-icon">📊</span>
                    <h2>Executive Summary</h2>
                </div>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-label">Total Records</div>
                        <div class="summary-value">{total_row_count:,}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Row Count (Last Day)</div>
                        <div class="summary-value">{row_count:,}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Max Business Date</div>
                        <div class="summary-value">{max_date}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Overall DQ Score</div>
                        <div class="summary-value">{overall_score:.1f}<span class="summary-unit">/ 100</span></div>
                    </div>
                </div>
            </div>
    """
    
    # Dimension Scores Table
    html += """
            <div class="section">
                <div class="section-header">
                    <span class="section-icon">📈</span>
                    <h2>Data Quality Dimensions</h2>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Dimension</th>
                            <th>Score</th>
                            <th>Status</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for dim_name, dim_data in dimensions.items():
        score = dim_data.get("score", 0)
        status_class = "score-good" if score >= 80 else "score-warning" if score >= 60 else "score-bad"
        status_text = "✓ Good" if score >= 80 else "⚠ Warning" if score >= 60 else "✗ Poor"
        
        # Get details based on dimension type
        details = []
        if "null_pcts" in dim_data:
            null_pcts = dim_data["null_pcts"]
            if null_pcts:
                high_nulls = [k for k, v in null_pcts.items() if v is not None and v > 5]
                if high_nulls:
                    details.append(f"High nulls: {', '.join(high_nulls[:3])}")
        if "duplicate_pct" in dim_data:
            dup_pct = dim_data.get("duplicate_pct", 0)
            if dup_pct > 0:
                details.append(f"Duplicates: {dup_pct:.1f}%")
        
        details_str = "; ".join(details) if details else "No issues"
        
        html += f"""
                    <tr>
                        <td><strong>{dim_name.title()}</strong></td>
                        <td class="{status_class}">{score:.2f}</td>
                        <td>{status_text}</td>
                        <td style="font-size: 12px; color: #666;">{details_str}</td>
                    </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
    """
    
    # Trends Table
    if trends:
        html += """
            <div class="section">
                <div class="section-header">
                    <span class="section-icon">📉</span>
                    <h2>Historical Trends (Last 30 Days)</h2>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Dimension</th>
                            <th>Today</th>
                            <th>Prev Day</th>
                            <th>7d Avg</th>
                            <th>30d Avg</th>
                            <th>Trend</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for dim_name, trend_data in trends.items():
            today = trend_data.get("today", 0)
            prev_day = trend_data.get("prev_day")
            avg_7d = trend_data.get("avg_7d")
            avg_30d = trend_data.get("avg_30d")
            
            # Format values properly (handle None)
            prev_day_str = f"{prev_day:.2f}" if prev_day is not None else "N/A"
            avg_7d_str = f"{avg_7d:.2f}" if avg_7d is not None else "N/A"
            avg_30d_str = f"{avg_30d:.2f}" if avg_30d is not None else "N/A"
            
            # Determine trend
            if prev_day is not None:
                if today > prev_day + 2:
                    trend = '<span class="trend-up">↑ Improving</span>'
                elif today < prev_day - 2:
                    trend = '<span class="trend-down">↓ Declining</span>'
                else:
                    trend = '<span class="trend-stable">→ Stable</span>'
            else:
                trend = '<span class="trend-stable">→ No data</span>'
            
            html += f"""
                        <tr>
                            <td><strong>{dim_name.title()}</strong></td>
                            <td>{today:.2f}</td>
                            <td>{prev_day_str}</td>
                            <td>{avg_7d_str}</td>
                            <td>{avg_30d_str}</td>
                            <td>{trend}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        """
        
        # Add bar charts for trends
        html += """
            <div class="section">
                <div class="section-header">
                    <span class="section-icon">📊</span>
                    <h2>Trend Visualization</h2>
                </div>
                <div class="chart-grid">
        """
        
        # Create a chart for each dimension
        chart_id = 0
        for dim_name, trend_data in trends.items():
            today = trend_data.get("today", 0) or 0
            prev_day = trend_data.get("prev_day") or 0
            avg_7d = trend_data.get("avg_7d") or 0
            avg_30d = trend_data.get("avg_30d") or 0
            
            chart_id += 1
            canvas_id = f"chart_{chart_id}"
            
            html += f"""
                    <div class="chart-card">
                        <div class="chart-title">{dim_name.title()}</div>
                        <div class="chart-container">
                            <canvas id="{canvas_id}"></canvas>
                        </div>
                    </div>
            """
        
        html += """
                </div>
            </div>
        """
        
        # Add JavaScript to render charts
        html += """
        <script>
            // Chart.js configuration
            Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
            Chart.defaults.font.size = 12;
            Chart.defaults.color = '#4a5568';
            
            // Color palette
            const colors = {
                today: '#667eea',
                prevDay: '#764ba2',
                avg7d: '#f093fb',
                avg30d: '#4facfe'
            };
            
            // Create charts for each dimension
        """
        
        chart_id = 0
        for dim_name, trend_data in trends.items():
            today = trend_data.get("today", 0) or 0
            prev_day = trend_data.get("prev_day") or 0
            avg_7d = trend_data.get("avg_7d") or 0
            avg_30d = trend_data.get("avg_30d") or 0
            
            chart_id += 1
            canvas_id = f"chart_{chart_id}"
            
            html += f"""
            (function() {{
                const ctx_{chart_id} = document.getElementById('{canvas_id}');
                if (ctx_{chart_id}) {{
                    new Chart(ctx_{chart_id}, {{
                        type: 'bar',
                        data: {{
                            labels: ['Today', 'Prev Day', '7d Avg', '30d Avg'],
                            datasets: [{{
                                label: 'Score',
                                data: [{today:.2f}, {prev_day:.2f}, {avg_7d:.2f}, {avg_30d:.2f}],
                                backgroundColor: [
                                    colors.today,
                                    colors.prevDay,
                                    colors.avg7d,
                                    colors.avg30d
                                ],
                                borderColor: [
                                    colors.today,
                                    colors.prevDay,
                                    colors.avg7d,
                                    colors.avg30d
                                ],
                                borderWidth: 2,
                                borderRadius: 6,
                                borderSkipped: false
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    display: false
                                }},
                                tooltip: {{
                                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                    padding: 12,
                                    titleFont: {{
                                        size: 14,
                                        weight: 'bold'
                                    }},
                                    bodyFont: {{
                                        size: 13
                                    }},
                                    callbacks: {{
                                        label: function(context) {{
                                            return 'Score: ' + context.parsed.y.toFixed(2);
                                        }}
                                    }}
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    max: 100,
                                    ticks: {{
                                        stepSize: 20,
                                        font: {{
                                            size: 11
                                        }},
                                        color: '#718096'
                                    }},
                                    grid: {{
                                        color: '#e2e8f0',
                                        drawBorder: false
                                    }}
                                }},
                                x: {{
                                    ticks: {{
                                        font: {{
                                            size: 11
                                        }},
                                        color: '#718096'
                                    }},
                                    grid: {{
                                        display: false
                                    }}
                                }}
                            }}
                        }}
                    }});
                }}
            }})();
            """
        
        html += """
        </script>
        """
    
    # Rule Results Table
    if rule_results:
        html += """
            <div class="section">
                <div class="section-header">
                    <span class="section-icon">✅</span>
                    <h2>Rule Evaluations</h2>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Rule</th>
                            <th>Status</th>
                            <th>Severity</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for rule in rule_results:
            status = rule.get("status", "UNKNOWN")
            severity = rule.get("severity", "info")
            detail = rule.get("detail", "")
            rule_name = rule.get("rule", "Unknown")
            
            status_class = "status-pass" if status == "PASS" else "status-fail"
            status_icon = "✓" if status == "PASS" else "✗"
            
            html += f"""
                        <tr>
                            <td><strong>{rule_name.replace('_', ' ').title()}</strong></td>
                            <td class="{status_class}">{status_icon} {status}</td>
                            <td>{severity.upper()}</td>
                            <td style="font-size: 12px; color: #666;">{detail}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        """
    
    # LLM Summary
    if formatted_summary:
        html += f"""
            <div class="section">
                <div class="section-header">
                    <span class="section-icon">🤖</span>
                    <h2>AI-Powered Analysis & Recommendations</h2>
                </div>
                <div class="ai-summary">
                    {formatted_summary}
                </div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


def extract_css_and_body(full_html: str) -> tuple:
    """Extract CSS from <style> tag, scripts, and body content from full HTML."""
    import re
    # Extract CSS from <style> tag
    style_match = re.search(r'<style[^>]*>(.*?)</style>', full_html, re.DOTALL | re.IGNORECASE)
    css = style_match.group(1).strip() if style_match else ""
    
    # Extract scripts
    script_matches = re.findall(r'<script[^>]*>(.*?)</script>', full_html, re.DOTALL | re.IGNORECASE)
    scripts = [s.strip() for s in script_matches if s.strip()]
    
    # Extract content between <body> and </body>
    body_match = re.search(r'<body[^>]*>(.*?)</body>', full_html, re.DOTALL | re.IGNORECASE)
    body_content = body_match.group(1).strip() if body_match else full_html
    
    return css, body_content, scripts


def format_multi_table_results_html(results: Dict[str, Dict[str, Any]]) -> str:
    """Format multiple DQ check results in a tabbed interface."""
    
    # Collect all CSS and scripts from successful results
    all_css = set()
    all_scripts = []  # Store scripts per table
    table_contents = {}
    table_scripts = {}  # Store scripts for each table
    tab_data = []
    
    for table_id, result in results.items():
        if "error" in result:
            # Error case
            table_contents[table_id] = f"""
                <div class="error-section" style="padding: 40px; text-align: center;">
                    <h2 style="color: #e74c3c; margin-bottom: 20px;">Error analyzing {table_id}</h2>
                    <p style="color: #c0392b; padding: 20px; background: #fee; border-radius: 8px; display: inline-block;">
                        {result["error"]}
                    </p>
                </div>
            """
            table_scripts[table_id] = []
            tab_data.append({
                "id": table_id,
                "label": table_id,
                "severity": "ERROR",
                "score": 0
            })
        else:
            # Normal result - use existing formatter and extract CSS, scripts, and body content
            full_html = format_dq_results_html(result)
            css, body_content, scripts = extract_css_and_body(full_html)
            if css:
                all_css.add(css)
            
            # Extract just the container content (remove outer container div)
            import re
            container_match = re.search(r'<div class="container">(.*?)</div>\s*</body>', body_content, re.DOTALL | re.IGNORECASE)
            if container_match:
                body_content = container_match.group(1).strip()
            
            # Make chart IDs unique by prefixing with table_id to avoid conflicts
            # Replace chart IDs in the content (canvas elements and script references)
            chart_id_pattern = r'chart_(\d+)'
            def replace_chart_id(match):
                return f'chart_{table_id}_{match.group(1)}'
            
            # Update chart IDs in body content (canvas elements)
            body_content = re.sub(chart_id_pattern, replace_chart_id, body_content)
            
            # Update chart IDs in scripts (getElementById calls and variable names)
            updated_scripts = []
            for script in scripts:
                updated_script = script
                # Replace getElementById('chart_X') with getElementById('chart_table_id_X')
                updated_script = re.sub(r"getElementById\('chart_(\d+)'\)", 
                                       lambda m: f"getElementById('chart_{table_id}_{m.group(1)}')", 
                                       updated_script)
                # Replace ctx_X variable names to avoid conflicts
                updated_script = re.sub(r'ctx_(\d+)', 
                                       lambda m: f'ctx_{table_id}_{m.group(1)}', 
                                       updated_script)
                updated_scripts.append(updated_script)
            
            table_scripts[table_id] = updated_scripts
            
            table_contents[table_id] = body_content
            severity = result.get("severity", "UNKNOWN")
            score = result.get("dq_profile", {}).get("overall_score", 0)
            tab_data.append({
                "id": table_id,
                "label": table_id,
                "severity": severity,
                "score": score
            })
    
    # Combine all CSS
    combined_css = "\n".join(all_css) if all_css else ""
    
    # Create tab navigation
    tabs_html = ""
    tab_panels_html = ""
    
    for i, tab in enumerate(tab_data):
        tab_id = tab["id"]
        is_active = "active" if i == 0 else ""
        severity_badge = f'<span class="tab-severity-badge severity-{tab["severity"].lower()}">{tab["severity"]}</span>' if tab["severity"] != "ERROR" else ""
        score_badge = f'<span class="tab-score-badge">{tab["score"]:.1f}</span>' if tab["score"] > 0 else ""
        
        tabs_html += f'''
            <button class="tab-button {is_active}" onclick="switchTab('{tab_id}')" data-tab="{tab_id}">
                <span class="tab-label">{tab["label"]}</span>
                {severity_badge}
                {score_badge}
            </button>
        '''
        
        # Include scripts for this table
        scripts_html = ""
        if tab_id in table_scripts and table_scripts[tab_id]:
            scripts_html = f'<script>{"".join(table_scripts[tab_id])}</script>'
        
        tab_panels_html += f'''
            <div id="tab-{tab_id}" class="tab-panel {is_active}">
                <div class="container">
                    {table_contents[tab_id]}
                </div>
                {scripts_html}
            </div>
        '''
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Quality Report - Multiple Tables</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .tabs-container {{
                max-width: 1600px;
                margin: 0 auto;
            }}
            .tabs-header {{
                background: white;
                border-radius: 16px 16px 0 0;
                padding: 20px 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: flex;
                gap: 10px;
                overflow-x: auto;
                flex-wrap: wrap;
            }}
            .tabs-content {{
                background: transparent;
            }}
            .tab-panel {{
                display: none;
            }}
            .tab-panel.active {{
                display: block;
            }}
            .tab-panel .container {{
                background: transparent;
                padding: 0;
            }}
            .tab-panel .header {{
                border-radius: 0;
                margin-bottom: 0;
            }}
            .tab-panel .section:first-child {{
                border-radius: 0 0 16px 16px;
            }}
            .tab-button {{
                background: #f7fafc;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 14px;
                font-weight: 600;
                color: #4a5568;
                display: flex;
                align-items: center;
                gap: 8px;
                white-space: nowrap;
                font-family: inherit;
            }}
            .tab-button:hover {{
                background: #edf2f7;
                border-color: #cbd5e0;
                transform: translateY(-2px);
            }}
            .tab-button.active {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-color: #667eea;
                color: white;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            .tab-label {{
                font-weight: 600;
            }}
            .tab-severity-badge {{
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .tab-button.active .tab-severity-badge {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
            }}
            .tab-button:not(.active) .tab-severity-badge.severity-critical {{
                background: #fee;
                color: #c0392b;
            }}
            .tab-button:not(.active) .tab-severity-badge.severity-high {{
                background: #fff4e6;
                color: #d68910;
            }}
            .tab-button:not(.active) .tab-severity-badge.severity-medium {{
                background: #fffbf0;
                color: #f39c12;
            }}
            .tab-button:not(.active) .tab-severity-badge.severity-low {{
                background: #ebf5fb;
                color: #2980b9;
            }}
            .tab-button:not(.active) .tab-severity-badge.severity-ok {{
                background: #eafaf1;
                color: #27ae60;
            }}
            .tab-score-badge {{
                font-size: 12px;
                padding: 4px 10px;
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.2);
                color: white;
                font-weight: 700;
            }}
            .tab-button:not(.active) .tab-score-badge {{
                background: #e2e8f0;
                color: #4a5568;
            }}
            {combined_css}
            .back-button {{
                position: fixed;
                top: 20px;
                left: 20px;
                background: white;
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                cursor: pointer;
                font-weight: 600;
                color: #667eea;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                z-index: 1000;
                font-family: inherit;
            }}
            .back-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <a href="/" class="back-button">← Back to Selection</a>
        <div class="tabs-container">
            <div class="tabs-header">
                {tabs_html}
            </div>
            <div class="tabs-content">
                {tab_panels_html}
            </div>
        </div>
        <script>
            function switchTab(tabId) {{
                // Hide all panels
                document.querySelectorAll('.tab-panel').forEach(panel => {{
                    panel.classList.remove('active');
                }});
                
                // Remove active class from all buttons
                document.querySelectorAll('.tab-button').forEach(button => {{
                    button.classList.remove('active');
                }});
                
                // Show selected panel
                const selectedPanel = document.getElementById('tab-' + tabId);
                selectedPanel.classList.add('active');
                
                // Add active class to selected button
                document.querySelector(`[data-tab="${{tabId}}"]`).classList.add('active');
                
                // Re-initialize charts in the active tab (in case they weren't rendered)
                // Chart.js will handle this automatically if canvas elements exist
                setTimeout(() => {{
                    // Force chart resize if needed
                    if (typeof Chart !== 'undefined') {{
                        Chart.helpers.each(Chart.instances, function(instance) {{
                            if (instance.canvas.closest('#tab-' + tabId)) {{
                                instance.resize();
                            }}
                        }});
                    }}
                }}, 100);
            }}
            
            // Initialize charts when page loads
            document.addEventListener('DOMContentLoaded', function() {{
                // Charts will be initialized by the scripts in each tab panel
            }});
        </script>
    </body>
    </html>
    """
    
    return html


# ============================================================
# 11. FastAPI endpoints
# ============================================================

def create_landing_page(available_tables: List[str], error: str = None) -> str:
    """Create a professional landing page for table selection."""
    error_html = f"""
        <div class="error-message" style="background: #fee; border: 1px solid #e74c3c; color: #c0392b; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <strong>⚠️ Error:</strong> {error}
        </div>
    """ if error else ""
    
    tables_checkboxes = ""
    for table in available_tables:
        tables_checkboxes += f'''
            <label class="checkbox-label">
                <input type="checkbox" name="table_id" value="{table}" class="checkbox-input">
                <span class="checkbox-custom"></span>
                <span class="checkbox-text">{table}</span>
            </label>
        '''
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Quality Assessment - NITCO DataTrust</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .container {{
                max-width: 600px;
                width: 100%;
            }}
            .card {{
                background: white;
                border-radius: 20px;
                padding: 50px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                animation: slideUp 0.5s ease-out;
            }}
            @keyframes slideUp {{
                from {{
                    opacity: 0;
                    transform: translateY(30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            .logo {{
                text-align: center;
                margin-bottom: 40px;
            }}
            .logo h1 {{
                font-size: 36px;
                font-weight: 700;
                color: #1a202c;
                margin-bottom: 10px;
                letter-spacing: -0.5px;
            }}
            .logo .subtitle {{
                font-size: 16px;
                color: #718096;
                font-weight: 500;
            }}
            .form-group {{
                margin-bottom: 25px;
            }}
            .form-label {{
                display: block;
                font-size: 14px;
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .form-input {{
                width: 100%;
                padding: 16px 20px;
                font-size: 16px;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                transition: all 0.3s ease;
                background: #f7fafc;
                color: #1a202c;
                font-family: inherit;
            }}
            .form-input:focus {{
                outline: none;
                border-color: #667eea;
                background: white;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            }}
            .form-select {{
                width: 100%;
                padding: 16px 20px;
                font-size: 16px;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                transition: all 0.3s ease;
                background: #f7fafc;
                color: #1a202c;
                font-family: inherit;
                cursor: pointer;
            }}
            .form-select:focus {{
                outline: none;
                border-color: #667eea;
                background: white;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            }}
            .submit-btn {{
                width: 100%;
                padding: 18px;
                font-size: 18px;
                font-weight: 600;
                color: white;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: 10px;
            }}
            .submit-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }}
            .submit-btn:active {{
                transform: translateY(0);
            }}
            .submit-btn:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
            }}
            .info-box {{
                background: #ebf5fb;
                border-left: 4px solid #3498db;
                padding: 15px;
                border-radius: 8px;
                margin-top: 25px;
            }}
            .info-box p {{
                font-size: 14px;
                color: #2c3e50;
                line-height: 1.6;
            }}
            .info-box strong {{
                color: #1a202c;
            }}
            .checkbox-container {{
                max-height: 300px;
                overflow-y: auto;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 15px;
                background: #f7fafc;
            }}
            .checkbox-label {{
                display: flex;
                align-items: center;
                padding: 12px;
                margin-bottom: 8px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            .checkbox-label:hover {{
                background: #edf2f7;
            }}
            .checkbox-input {{
                display: none;
            }}
            .checkbox-custom {{
                width: 22px;
                height: 22px;
                border: 2px solid #cbd5e0;
                border-radius: 6px;
                margin-right: 12px;
                position: relative;
                transition: all 0.2s ease;
                flex-shrink: 0;
            }}
            .checkbox-input:checked + .checkbox-custom {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-color: #667eea;
            }}
            .checkbox-input:checked + .checkbox-custom::after {{
                content: '✓';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 14px;
                font-weight: bold;
            }}
            .checkbox-text {{
                font-size: 15px;
                color: #2d3748;
                font-weight: 500;
            }}
            .select-all-container {{
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 2px solid #e2e8f0;
            }}
            .select-all-btn {{
                background: transparent;
                border: 2px solid #667eea;
                color: #667eea;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 600;
                transition: all 0.2s ease;
            }}
            .select-all-btn:hover {{
                background: #667eea;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="logo">
                    <h1>🔍 Data Quality Assessment Agent</h1>
                    <div class="subtitle">NITCO DataTrust Probe</div>
                </div>
                
                {error_html}
                
                <form method="POST" action="/dq/analyze" id="analysisForm">
                    <div class="form-group">
                        <label class="form-label">Select Tables to Analyze</label>
                        <div class="select-all-container">
                            <button type="button" class="select-all-btn" onclick="toggleAll()">Select / Deselect All</button>
                        </div>
                        <div class="checkbox-container">
                            {tables_checkboxes}
                        </div>
                    </div>
                    
                    <button type="submit" class="submit-btn" id="submitBtn">
                        Analyze Data Quality
                    </button>
                </form>
                
                <script>
                    function toggleAll() {{
                        const checkboxes = document.querySelectorAll('input[name="table_id"]');
                        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
                        checkboxes.forEach(cb => cb.checked = !allChecked);
                        updateSubmitButton();
                    }}
                    
                    function updateSubmitButton() {{
                        const checked = document.querySelectorAll('input[name="table_id"]:checked').length;
                        const btn = document.getElementById('submitBtn');
                        if (checked === 0) {{
                            btn.disabled = true;
                            btn.textContent = 'Please select at least one table';
                        }} else {{
                            btn.disabled = false;
                            btn.textContent = `Analyze Data Quality (${{checked}} table${{checked > 1 ? 's' : ''}} selected)`;
                        }}
                    }}
                    
                    document.querySelectorAll('input[name="table_id"]').forEach(cb => {{
                        cb.addEventListener('change', updateSubmitButton);
                    }});
                    
                    document.getElementById('analysisForm').addEventListener('submit', function(e) {{
                        const checked = document.querySelectorAll('input[name="table_id"]:checked').length;
                        if (checked === 0) {{
                            e.preventDefault();
                            alert('Please select at least one table to analyze.');
                        }}
                    }});
                    
                    updateSubmitButton();
                </script>
                
                <div class="info-box">
                    <p>
                        <strong>What this tool does:</strong><br>
                        This application performs comprehensive data quality analysis including completeness, 
                        uniqueness, accuracy, validity, consistency, and integrity checks. Results include 
                        trend analysis and AI-powered recommendations.
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


@app.get("/")
def landing_page():
    """Landing page with table selection form."""
    available_tables = get_available_tables()
    html_content = create_landing_page(available_tables)
    return HTMLResponse(content=html_content)


@app.post("/dq/analyze")
async def analyze_table_post(request: Request):
    """Handle form submission with multiple table selections."""
    try:
        form_data = await request.form()
        table_ids = form_data.getlist("table_id")
        
        if not table_ids:
            available_tables = get_available_tables()
            error_msg = "Please select at least one table to analyze."
            html_content = create_landing_page(available_tables, error=error_msg)
            return HTMLResponse(content=html_content, status_code=400)
        
        # Validate all table_ids exist
        available_tables = get_available_tables()
        invalid_tables = [tid for tid in table_ids if tid not in available_tables]
        if invalid_tables:
            error_msg = f"Invalid tables: {', '.join(invalid_tables)}. Available tables: {', '.join(available_tables)}"
            html_content = create_landing_page(available_tables, error=error_msg)
            return HTMLResponse(content=html_content, status_code=400)
        
        # Redirect to the results page with multiple table IDs
        table_ids_param = "&".join([f"table_id={tid}" for tid in table_ids])
        return RedirectResponse(url=f"/dq/run?{table_ids_param}&format=html", status_code=302)
    except Exception as e:
        available_tables = get_available_tables()
        error_msg = f"Error: {str(e)}"
        html_content = create_landing_page(available_tables, error=error_msg)
        return HTMLResponse(content=html_content, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dq/run")
def dq_run(
    request: Request,
    table_id: str = Query(None, description="Single table_id (for backward compatibility)"),
    format: str = Query("html", description="Output format: 'json' or 'html'")
):
    try:
        # Handle multiple table_ids from query parameters
        table_ids = request.query_params.getlist("table_id")
        if not table_ids and table_id:
            table_ids = [table_id]
        
        # If no table_ids provided, return error
        if not table_ids or (len(table_ids) == 1 and not table_ids[0]):
            available_tables = get_available_tables()
            error_msg = "Please select at least one table to analyze."
            html_content = create_landing_page(available_tables, error=error_msg)
            return HTMLResponse(content=html_content, status_code=400)
        
        # Remove duplicates and empty values
        table_ids = list(set([tid for tid in table_ids if tid]))
        
        # Run DQ checks for all tables
        results = {}
        for tid in table_ids:
            try:
                results[tid] = run_dq_check_for_table(tid)
            except Exception as e:
                results[tid] = {"error": str(e)}
        
        if format.lower() == "html":
            if len(results) == 1:
                # Single table - use existing format
                html_content = format_dq_results_html(list(results.values())[0])
            else:
                # Multiple tables - use tabbed format
                html_content = format_multi_table_results_html(results)
            return HTMLResponse(content=html_content)
        else:
            # JSON format - return all results
            return JSONResponse(content=results)
    except FileNotFoundError as e:
        # Return to landing page with error
        available_tables = get_available_tables()
        error_msg = str(e)
        html_content = create_landing_page(available_tables, error=error_msg)
        return HTMLResponse(content=html_content, status_code=404)
    except HTTPException as e:
        raise e
    except psycopg2_errors.UndefinedTable as e:
        raise HTTPException(status_code=500, detail=f"Table not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running DQ check: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dq_service:app", host="0.0.0.0", port=8001, reload=True)
