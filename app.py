from __future__ import annotations
import os
import io
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote_plus


# =============================
# Utilities
# =============================

# --- GitHub config (EDIT THESE TO MATCH YOUR REPO) ---
GITHUB_OWNER = "<your-org-or-username>"
GITHUB_REPO = "<your-repo>"
GITHUB_FILE_PATH = "CFB Rankings Upload.xlsm"  # path inside the repo
GITHUB_REF = "main"  # branch, tag, or commit sha
# Optional: if the repo is private, set env var GITHUB_TOKEN in your deployment settings

import requests
import base64


def fetch_github_bytes(owner: str, repo: str, path: str, ref: str = "main", token: str | None = None) -> bytes:
    """Fetch a file's raw bytes from GitHub. Tries raw URL first, then Contents API (base64)."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    # 1) Try raw.githubusercontent.com (works for public and private with token when using API download)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    r = requests.get(raw_url, headers=headers, timeout=30)
    if r.status_code == 200 and r.content:
        return r.content

    # 2) Fallback: GitHub Contents API (base64-encoded)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r2 = requests.get(api_url, headers=headers, timeout=30)
    if r2.status_code == 200:
        payload = r2.json()
        if isinstance(payload, dict) and payload.get("encoding") == "base64" and payload.get("content"):
            return base64.b64decode(payload["content"])

    raise RuntimeError(f"Failed to fetch file from GitHub: {owner}/{repo}@{ref}:{path} (status {r.status_code} / {r2.status_code if 'r2' in locals() else 'no-contents'})")

def read_excel_flexible(file_like, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read an Excel(xls/xlsx/xlsm) into a DataFrame using openpyxl if available.
    Tries pandas default engine first, falls back to openpyxl.
    If sheet_name is None, reads the first visible sheet.
    """
    try:
        if sheet_name is None:
            return pd.read_excel(file_like)
        return pd.read_excel(file_like, sheet_name=sheet_name)
    except Exception:
        try:
            if sheet_name is None:
                return pd.read_excel(file_like, engine="openpyxl")
            return pd.read_excel(file_like, engine="openpyxl", sheet_name=sheet_name)
        except Exception as e:
            raise e


def load_data(sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load rankings data directly from GitHub (no uploads)."""
    token = os.getenv("GITHUB_TOKEN", None)
    try:
        blob = fetch_github_bytes(GITHUB_OWNER, GITHUB_REPO, GITHUB_FILE_PATH, ref=GITHUB_REF, token=token)
    except Exception as e:
        raise ValueError(
            "Could not download Excel from GitHub.
"
            f"Repo: {GITHUB_OWNER}/{GITHUB_REPO}
Path: {GITHUB_FILE_PATH}
Ref: {GITHUB_REF}
"
            f"Detail: {e}"
        )

    # Read the workbook bytes into a DataFrame
    df = read_excel_flexible(io.BytesIO(blob), sheet_name=sheet_name)

    if df is None or (isinstance(df, dict) and all((not isinstance(v, pd.DataFrame) or v.empty) for v in df.values())):
        raise ValueError("The GitHub workbook has no non-empty sheets.")

    if isinstance(df, dict):
        # pick the first non-empty sheet deterministically by name
        for name in sorted(df.keys()):
            sub = df[name]
            if isinstance(sub, pd.DataFrame) and not sub.empty:
                df = sub
                break

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_team_series(df: pd.DataFrame) -> pd.Series:
    """Return a string Series with team names, from one of common columns or the index."""
    candidates = ["Team Name", "Team", "School", "Name"]
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(str)
            s.name = c
            return s
    s = pd.Series(df.index.astype(str), index=df.index, name="__index_team__")
    return s


def get_image_series(df: pd.DataFrame) -> pd.Series:
    """Return a URL Series for logos if present; otherwise a None series."""
    if "Image URL" in df.columns:
        return df["Image URL"]
    return pd.Series([None] * len(df), index=df.index, name="Image URL")


def team_logo_anchor(url: Optional[str], team_name: str) -> str:
    """Return HTML for a small clickable logo that links to the Team Dashboards tab for the given team."""
    if not url or (isinstance(url, float) and np.isnan(url)):
        return ""
    q_team = quote_plus(str(team_name))
    return f'<a href="?tab=Team%20Dashboards&team={q_team}" title="Open Team Dashboard for {team_name}"><img src="{url}" width="18"></a>'


# =============================
# App Config
# =============================

st.set_page_config(page_title="CFB Rankings & Team Dashboards", layout="wide")

# Read query params (Streamlit >=1.30: st.query_params; fallback to experimental)
try:
    qp = st.query_params  # type: ignore[attr-defined]
    requested_tab = qp.get("tab", "Rankings")
    requested_team = qp.get("team", None)
except Exception:
    try:
        qp2 = st.experimental_get_query_params()  # deprecated fallback
        requested_tab = qp2.get("tab", ["Rankings"])[0]
        requested_team = qp2.get("team", [None])[0]
    except Exception:
        requested_tab, requested_team = "Rankings", None


# =============================
# Load Data
# =============================

try:
    df = load_data()
except Exception as e:
    st.error("❌ Unable to load data from GitHub. Check the repo/name/path/ref and (if private) set GITHUB_TOKEN.")
    st.exception(e)
    st.stop()

# Build robust team + image series
team_s = get_team_series(df)
img_s = get_image_series(df)

# Guarantee an identifier column for display
if team_s.name not in df.columns:
    # If team names come from index, create a visible column
    df = df.copy()
    df["Team"] = team_s.values
    team_col_name = "Team"
else:
    team_col_name = team_s.name

# Create clickable logo column
logo_col = [team_logo_anchor(u, t) for u, t in zip(img_s, team_s)]

# Compose a display DataFrame for Rankings
# Prefer to show Logo + Team + common metric columns if they exist
common_metrics = [
    "Rk", "Pre Rk", "Pwr Rtg", "Off Rtg", "Def Rtg",
    "W", "L", "Proj W", "Proj Conf W", "Sched Diff"
]
show_cols: list[str] = []
if "Team Logo" not in df.columns:
    df = df.copy()
    df["Team Logo"] = logo_col
show_cols.append("Team Logo")
if team_col_name in df.columns:
    show_cols.append(team_col_name)
for c in common_metrics:
    if c in df.columns:
        show_cols.append(c)

rankings_df = df[show_cols].copy()

# =============================
# UI — Tabs
# =============================

tabs = st.tabs(["Rankings", "Team Dashboards"])  # Streamlit can't preselect, so we JS-click below

# Auto-click the dashboard tab if query params request it
if requested_tab == "Team Dashboards":
    st.markdown(
        """
        <script>
        (function(){
          const want = "Team Dashboards";
          const go = () => {
            const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
            for (const t of tabs) { if (t.innerText.trim() === want) { t.click(); break; } }
          };
          window.addEventListener('load', () => setTimeout(go, 60));
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Rankings Tab
# -----------------------------
with tabs[0]:
    st.markdown("## 🏈 College Football Rankings")

    # Basic sorting if a rank column exists
    if "Rk" in rankings_df.columns:
        rankings_df = rankings_df.sort_values("Rk", ascending=True)

    # Render with HTML to show clickable images
    html_table = (
        rankings_df.to_html(escape=False, index=False)
        .replace("<table border=\"1\" class=\"dataframe\">", "<table class=\"dataframe\" style=\"width:100%;\">")
    )

    st.write(html_table, unsafe_allow_html=True)

    # Optional: double-click anywhere on a team logo to navigate (single-click via <a> already works)
    st.markdown(
        """
        <script>
        document.addEventListener('dblclick', function(e) {
          const img = e.target.closest('img');
          if (!img) return;
          const parent = img.closest('a');
          if (parent && parent.href) { window.location = parent.href; }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Team Dashboards Tab
# -----------------------------
with tabs[1]:
    st.markdown("## 📊 Team Dashboards")

    # All unique team names (strings)
    teams = sorted(pd.unique(team_s.astype(str)))

    # Determine default
    default_team = None
    if requested_team is not None and str(requested_team) in teams:
        default_team = str(requested_team)
    elif teams:
        default_team = teams[0]

    # Selector
    if teams:
        idx = teams.index(default_team) if default_team in teams else 0
        selected_team = st.selectbox("Team", teams, index=idx, key="team_dashboard_select")
    else:
        st.warning("No teams found in the dataset.")
        st.stop()

    # Filter the rows for the selected team whether team is a column or index
    if team_s.name in df.columns:
        filtered = df[df[team_s.name].astype(str) == str(selected_team)]
    else:
        filtered = df[df.index.astype(str) == str(selected_team)]

    if filtered.empty:
        st.info("No rows found for the selected team.")
    else:
        # KPIs from the first row
        row = filtered.iloc[0]
        kpi_cols = [c for c in [
            "Rk", "Pre Rk", "Pwr Rtg", "Off Rtg", "Def Rtg",
            "W", "L", "Proj W", "Proj Conf W", "Sched Diff"
        ] if c in filtered.columns]

        if kpi_cols:
            cols = st.columns(min(4, max(1, len(kpi_cols))))
            for i, c in enumerate(kpi_cols[:4]):
                val = row[c]
                if isinstance(val, (int, float)) and c not in ["Rk", "Pre Rk", "W", "L"]:
                    cols[i].metric(c, f"{val:.1f}")
                else:
                    cols[i].metric(c, f"{val}")

        st.markdown("#### Team Detail")
        # Hide the raw team column if it exists; keep everything else (you can also hide the logo col if desired)
        hide_cols = set()
        if team_s.name in filtered.columns:
            hide_cols.add(team_s.name)
        # hide_cols.add("Team Logo")  # uncomment to hide the small logo column in the detail table

        detail_cols = [c for c in filtered.columns if c not in hide_cols]
        st.dataframe(filtered[detail_cols], use_container_width=True)

        # Placeholder for expansion: add charts, schedules, player stats, etc.
        # Example: Show numeric columns as a quick profile
        num_cols = [c for c in detail_cols if pd.api.types.is_numeric_dtype(filtered[c])]
        if num_cols:
            st.markdown("#### Numeric Profile (first row)")
            st.bar_chart(filtered.iloc[[0]][num_cols].T)


# =============================
# Footer / Help
# =============================
with st.expander("Help & Tips"):
    st.markdown(
        """
        **Navigation**  
        • Click a team logo on the *Rankings* tab to jump to that team's dashboard.  
        • The URL will include `?tab=Team%20Dashboards&team=<Team>` so you can bookmark/share direct links.  

        **Data expectations**  
        • A team identifier is required in one of these: `Team Name`, `Team`, `School`, `Name`, or the dataframe index.  
        • Logos are optional in the `Image URL` column.  
        • Common metric columns (shown if present): `Rk`, `Pre Rk`, `Pwr Rtg`, `Off Rtg`, `Def Rtg`, `W`, `L`, `Proj W`, `Proj Conf W`, `Sched Diff`.  

        **Customization**  
        • Add charts, schedules, and deeper stats inside the Team Dashboards tab where indicated.  
        • If your sheet uses different column names, update `get_team_series()` and the `common_metrics` list accordingly.  
        """
    )
