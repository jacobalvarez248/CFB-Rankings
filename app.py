import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from urllib.parse import unquote, quote
import streamlit.components.v1 as components
import numpy as np

st.set_page_config(page_title="CFB Rankings", layout="wide", initial_sidebar_state="expanded")

import urllib.parse
import streamlit as st

# ---- Read query params and stash into session_state for Comparison ----
qp = st.query_params
if qp.get("view") == "comparison":
    st.session_state["comp_home"] = qp.get("home", "")
    st.session_state["comp_away"] = qp.get("away", "")
    st.session_state["comp_neutral"] = (qp.get("neutral", "0") == "1")

# ---- Optional: auto-switch to the Comparison tab when view=comparison ----
st.markdown("""
<script>
(function(){
  const params = new URLSearchParams(window.location.search);
  if (params.get('view') === 'comparison') {
    const target = 'Comparison';
    function clickTab(){
      const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
      for (const t of tabs) {
        if ((t.innerText || '').trim().toLowerCase() === target.toLowerCase()) {
          t.click();
          return;
        }
      }
      setTimeout(clickTab, 60);
    }
    clickTab();
  }
})();
</script>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    expected_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    metrics_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Metrics', header=0)
    
    # Fix column naming issue
    metrics_df.reset_index(inplace=True)
    return expected_df, logos_df, metrics_df

df, logos_df, metrics_df = load_data()

def deduplicate_columns(columns):
    seen = {}
    out = []
    for c in columns:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

df.columns = deduplicate_columns(df.columns)
df = df.loc[:, ~df.columns.str.contains(r'\.(1|2|3|4)$')]
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')
if 'Current Rank' in df.columns:
    df['Current Rank'] = df['Current Rank'].astype('Int64')
df['Team Name'] = df['Team']
df.set_index('Team Name', inplace=True)

conf_logo_map = logos_df.set_index('Team')['Image URL'].to_dict()
df['Conference Logo'] = df.get('Conference', pd.NA).apply(
    lambda conf: f'<img src="{conf_logo_map.get(conf, '')}" width="15">' if conf_logo_map.get(conf) else (conf if pd.notna(conf) else '')
)
df['Conf Name'] = df.get('Conference', pd.NA)

df.drop(columns=[
    "Conference", "Image URL", "Vegas Win Total",
    "Projected Overall Wins", "Projected Overall Losses",
    "Projected Conference Wins", "Projected Conference Losses",
    "Schedule Difficulty Rank", "Column1", "Column3", "Column5"
], errors='ignore', inplace=True)

df.rename(columns={
    "Preseason Rank": "Pre Rk",
    "Current Rank": "Rk",
    "Power Rating": "Pwr Rtg",
    "Offensive Rating": "Off Rtg",
    "Defensive Rating": "Def Rtg",
    "Current Wins": "W",
    "Current Losses": "L",
    "Schedule Difficulty": "Sched Diff",
    "Conference Logo": "Conf"
}, inplace=True)

first_cols = ["Pre Rk", "Rk", "Team", "Conf"]
existing = [c for c in df.columns if c not in first_cols]
df = df[[c for c in first_cols if c in df.columns] + existing]

query_params = st.query_params
preselect_team = unquote(query_params.get("selected_team", ""))
if 'selected_team' not in st.session_state:
    st.session_state['selected_team'] = preselect_team if preselect_team else df.index[0]

# --- Build a clean team-level frame from Metrics for ranks + ratings
def build_team_frame(df_expected, df_metrics, df_logos):
    # keep only what we need + attach logos
    base = df_metrics.copy()
    # Ratings expected to exist in Metrics sheet
    # 'Team', 'Power Rating', 'Offensive Rating', 'Defensive Rating', plus per-metric cols (Off./Def. …)
    needed = ['Team', 'Power Rating', 'Offensive Rating', 'Defensive Rating']
    missing = [c for c in needed if c not in base.columns]
    if missing:
        raise ValueError(f"Missing columns on Metrics sheet: {missing}")

    logos_map = df_logos.set_index('Team')['Image URL'].to_dict()
    base['Logo'] = base['Team'].map(logos_map).fillna('')

    # Global ranks across ALL teams
    base['Pwr Rank'] = base['Power Rating'].rank(ascending=False, method='min').astype(int)
    base['Off Rank'] = base['Offensive Rating'].rank(ascending=False, method='min').astype(int)
    base['Def Rank'] = base['Defensive Rating'].rank(ascending=True,  method='min').astype(int)  # lower is better on defense

    return base

# --- Metric map: Offense column vs matching Defense column on the Metrics sheet
COMPARISON_METRICS = [
    # Yards / Game
    ("Yards/Game",           "Off. Yds/Game",           "Def. Yds/Game"),
    ("Pass Yards/Game",      "Off. Pass Yds/Game",      "Def. Pass Yds/Game"),
    ("Rush Yards/Game",      "Off. Rush Yds/Game",      "Def. Rush Yds/Game"),
    ("Points/Game",          "Off. Points/Game",        "Def. Points/Game"),
    # Yards / Play
    ("Yards/Play",           "Off. Yds/Play",           "Def. Yds/Play"),
    ("Pass Yards/Play",      "Off. Pass Yds/Play",      "Def. Pass Yds/Play"),
    ("Rush Yards/Play",      "Off. Rush Yds/Play",      "Def. Rush Yds/Play"),
    ("Points/Play",          "Off. Points/Play",        "Def. Points/Play"),
    # EPA / Play
    ("EPA/Play",             "Off. EPA/Play",           "Def. EPA/Play"),
    ("Pass EPA/Play",        "Off. Pass EPA/Play",      "Def. Pass EPA/Play"),
    ("Rush EPA/Play",        "Off. Rush EPA/Play",      "Def. Rush EPA/Play"),
    ("Pts/Scoring Opp.",     "Off. Points/Scoring Opp.", "Def. Points/Scoring Opp."),
    # Success Rate (percent)
    ("Success Rate",         "Off. Success Rate",       "Def. Success Rate"),
    ("Pass Success Rate",    "Off. Pass Success Rate",  "Def. Pass Success Rate"),
    ("Rush Success Rate",    "Off. Rush Success Rate",  "Def. Rush Success Rate"),
    # Explosiveness (rate)
    ("Explosiveness",        "Off. Explosiveness",      "Def. Explosiveness"),
    ("Pass Explosiveness",   "Off. Pass Explosivenes",  "Def. Pass Explosivenes"),
    ("Rush Explosiveness",   "Off. Rush Explosiveness", "Def. Rush Explosiveness"),
]

def build_rank_tables(team_df, home, away):
    """
    Returns:
      when_home_has_ball: home OFF vs away DEF
      when_away_has_ball: away OFF vs home DEF
    """

    # <<< Key fix: set index to Team for all rank lookups >>>
    tdf = team_df.set_index('Team', drop=False)
    nteams = len(tdf)

    def rank_series(col, higher_is_better):
        s = tdf[col]
        return s.rank(ascending=not higher_is_better, method='min').astype(int)

    # Precompute all ranks once (offense high→good, defense low→good)
    ranks = {}
    for label, off_col, def_col in COMPARISON_METRICS:
        if off_col in tdf.columns:
            ranks[off_col] = rank_series(off_col, higher_is_better=True)
        if def_col in tdf.columns:
            ranks[def_col] = rank_series(def_col, higher_is_better=False)

    def one_side(off_team, def_team):
        rows = []
        for label, off_col, def_col in COMPARISON_METRICS:
            # Skip missing metric columns gracefully
            if off_col not in ranks or def_col not in ranks:
                continue
            if off_team not in ranks[off_col].index or def_team not in ranks[def_col].index:
                continue
            off_rank = int(ranks[off_col].loc[off_team])
            def_rank = int(ranks[def_col].loc[def_team])
            diff = abs(off_rank - def_rank)
            rows.append((label, off_rank, def_rank, diff))
        out = pd.DataFrame(rows, columns=['Metric', 'Off Rank', 'Def Rank', 'Δ Rank'])
        out['N'] = nteams
        return out

    return one_side(home, away), one_side(away, home)


def projected_score(team_df, home, away, neutral: bool):
    """
    Project scores using Metrics sheet columns:
      - OFF from column 'off' (Excel DN)
      - DEF from column 'def' (Excel DO)
    Falls back to 'Offensive Rating' / 'Defensive Rating' if needed.
    """
    # pick columns robustly
    OFF_CANDIDATES = ['off', 'Off', 'OFF', 'Offensive Rating']
    DEF_CANDIDATES = ['def', 'Def', 'DEF', 'Defensive Rating']

    def pick_col(candidates):
        for c in candidates:
            if c in team_df.columns:
                return c
        raise KeyError(f"None of these columns were found: {candidates}")

    off_col = pick_col(OFF_CANDIDATES)
    def_col = pick_col(DEF_CANDIDATES)

    tdf = team_df.set_index('Team', drop=False)
    h = tdf.loc[home]
    a = tdf.loc[away]

    # TOTAL = avg(home OFF vs away DEF, away OFF vs home DEF)
    total = (h[off_col] + a[def_col]) / 2.0 + (a[off_col] + h[def_col]) / 2.0

    # Projected difference = home PWR - away PWR + 2.5 (if not neutral)
    projected_diff = (h['Power Rating'] - a['Power Rating']) + (0 if neutral else 2.5)

    home_score = total / 2.0 + projected_diff / 2.0
    away_score = total / 2.0 - projected_diff / 2.0
    return float(total), float(home_score), float(away_score)


def style_rank_table(df):
    """Blue gradient where rank #1 is darkest; also badge the 5 closest and 5 furthest by Δ Rank."""
    n = int(df['N'].iloc[0])
    closest_idx = df.nsmallest(5, 'Δ Rank').index
    furthest_idx = df.nlargest(5, 'Δ Rank').index

    def highlight_rows(row):
        if row.name in closest_idx:
            return ['background-color: #e6f0ff'] * len(row)
        if row.name in furthest_idx:
            return ['background-color: #ffe9cc'] * len(row)
        return [''] * len(row)

    from matplotlib.colors import LinearSegmentedColormap
    cmap_blue = LinearSegmentedColormap.from_list('white_to_darknavy', ['#ffffff', '#002060'])
    cmap_blue_r = cmap_blue.reversed()

    styled = (
        df.drop(columns=['N'])
          .style
          .apply(highlight_rows, axis=1)
          .background_gradient(cmap=cmap_blue_r, subset=['Off Rank'], vmin=1, vmax=n)
          .background_gradient(cmap=cmap_blue_r, subset=['Def Rank'], vmin=1, vmax=n)
          .format({'Off Rank': '{:d}', 'Def Rank': '{:d}', 'Δ Rank': '{:d}'})
          .hide(axis="index")                  # 👈 hides the left index column
    )
    return styled

# Tab selector with auto-switch logic
query_params = st.query_params
selected_team = query_params.get("selected_team", "")

# 👇 NEW: if URL says view=comparison, make that the default
if query_params.get("view") == "comparison":
    default_tab = "🤝 Comparison"
elif selected_team:
    default_tab = "📊 Team Dashboards"
else:
    default_tab = "🏆 Rankings"

tabs = ["🏆 Rankings", "📈 Metrics", "📊 Team Dashboards", "🤝 Comparison"]
default_index = tabs.index(default_tab)

tab_choice = st.radio(
    " ",
    tabs,
    horizontal=True,
    label_visibility="collapsed",
    index=default_index,)

#-----------------------------------------------------RANKINGS TAB------------------------------------------------
if tab_choice == "🏆 Rankings":
    with st.sidebar:
        st.header("Filters & Sort")
        team_query = st.text_input("Team contains", value="")
        conf_options = sorted([c for c in df['Conf Name'].dropna().unique()])
        conf_selected = st.multiselect("Conference", conf_options)
        sortable_cols = [c for c in df.columns if c not in ['Team', 'Conf']] + ['Team Name', 'Conf Name']
        primary_sort = st.selectbox("Sort by", options=sortable_cols, index=sortable_cols.index('Rk') if 'Rk' in sortable_cols else 0)
        sort_ascending = st.checkbox("Ascending", value=True)

    st.markdown("## 🏈 College Football Rankings")

    view = df.copy()
    if team_query:
        view = view[view.index.str.contains(team_query, case=False, na=False)]
    if conf_selected:
        view = view[view['Conf Name'].isin(conf_selected)]
    if primary_sort in view.columns:
        view = view.sort_values(by=primary_sort, ascending=sort_ascending, kind="mergesort")
    elif primary_sort == 'Team Name':
        view = view.sort_values(by=view.index.name, ascending=sort_ascending, kind="mergesort")
    elif primary_sort == 'Conf Name':
        view = view.sort_values(by='Conf Name', ascending=sort_ascending, kind="mergesort")

    visible_cols = [c for c in view.columns if c != 'Conf Name']
    view = view[visible_cols]

    from urllib.parse import quote  # make sure this import is at the top
    view['Team'] = view.apply(
        lambda row: (
            f'<a href="?selected_team={quote(row.name)}#📊%20Team%20Dashboards">'
            f'<img src="{logos_df.set_index("Team").at[row.name, "Image URL"]}" width="15"></a>'
        ) if row.name in logos_df.set_index('Team').index else '',
        axis=1
    )

    numeric_cols = [c for c in view.columns if pd.api.types.is_numeric_dtype(view[c])]
    fmt = {c: '{:.1f}' for c in numeric_cols}
    for col in ['Pre Rk', 'Rk', 'W', 'L']:
        if col in view.columns:
            fmt[col] = '{:.0f}'

    styled = view.style.format(fmt).hide(axis='index')

    cmap_blue = LinearSegmentedColormap.from_list('white_to_darknavy', ['#ffffff', '#002060'])
    cmap_blue_r = cmap_blue.reversed()
    cmap_green = LinearSegmentedColormap.from_list('white_to_darkgreen', ['#ffffff', '#006400'])
    cmap_gold = LinearSegmentedColormap.from_list('darkgold_to_white', ['#b8860b', '#ffffff'])

    def text_contrast(series, invert=False):
        vmin = float(series.min(skipna=True))
        vmax = float(series.max(skipna=True))
        rng = (vmax - vmin) if vmax != vmin else 1.0
        norm = (series - vmin) / rng
        if invert:
            norm = 1 - norm
        return ['color: white' if (x >= 0.6) else 'color: black' for x in norm.fillna(0)]

    for col in ['Pwr Rtg', 'Off Rtg']:
        if col in view.columns:
            styled = styled.background_gradient(cmap=cmap_blue, subset=[col], vmin=view[col].min(), vmax=view[col].max()).apply(text_contrast, subset=[col])
    for col in ['Proj W', 'Proj Conf W']:
        if col in view.columns:
            styled = styled.background_gradient(cmap=cmap_green, subset=[col], vmin=view[col].min(), vmax=view[col].max()).apply(text_contrast, subset=[col])
    for col in ['Def Rtg']:
        if col in view.columns:
            styled = styled.background_gradient(cmap=cmap_blue_r, subset=[col], vmin=view[col].min(), vmax=view[col].max()).apply(lambda s: text_contrast(s, invert=True), subset=[col])
    for col in ['Sched Diff']:
        if col in view.columns:
            styled = styled.background_gradient(cmap=cmap_gold, subset=[col], vmin=view[col].min(), vmax=view[col].max()).apply(lambda s: text_contrast(s, invert=True), subset=[col])

    st.markdown("""
    <style>
    .block-container { padding-left: .5rem !important; padding-right: .5rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 11px; border-collapse: collapse; }
    td, th { padding: 4px !important; text-align: center !important; vertical-align: middle !important; font-size: 11px; }
    thead th {
      background-color: #002060 !important;
      color: #ffffff !important;
      font-weight: 500 !important;
      font-size: 10px !important;
      padding: 1px 3px !important;
    }
    thead th:nth-child(3), thead th:nth-child(4),
    tbody td:nth-child(3), tbody td:nth-child(4) {
      text-align: center !important;
      vertical-align: middle !important;
      width: 50px; min-width: 55px; max-width: 75px;
      overflow: hidden;
    }
    td img { display: block; margin: 0 auto; }
    tbody td { padding-left: 2px !important; padding-right: 2px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.write(styled.to_html(escape=False), unsafe_allow_html=True)

#-----------------------------------------------------METRICS TAB------------------------------------------------
if tab_choice == "📈 Metrics":
    import html
    from urllib.parse import quote

    st.markdown("## 📈 Metrics")

    # === Controls (no sort dropdown) ===
    c1, c2 = st.columns(2)
    with c1:
        unit_choice = st.selectbox("Unit", ["Offense", "Defense"], key="metrics_unit")
    with c2:
        metric_choice = st.selectbox(
            "Metric",
            ["Yards/Game", "Yards/Play", "EPA/Play", "Pts/Scoring Opp.", "Success Rate", "Explosiveness"],
            key="metrics_metric",
        )

    # Ensure clean headers
    df.columns = df.columns.str.strip()
    metrics_df.columns = metrics_df.columns.str.strip()
    logos_df.columns = logos_df.columns.str.strip()

    # Merge core team/rating info with the Metrics sheet
    core_cols = ['Team', 'Rk', 'Pwr Rtg', 'Off Rtg', 'Def Rtg', 'Conf Name']
    df_core = df[[c for c in core_cols if c in df.columns]].copy()
    merged_df = pd.merge(df_core, metrics_df, on='Team', how='inner')

    # Helpers to tolerate header variations (Explosivenes vs Explosiveness)
    def _col(name: str):
        for cand in (name, name.replace("Explosiveness", "Explosivenes")):
            if cand in merged_df.columns:
                return cand
        return None

    METRIC_FAMILIES = {
        "Yards/Game":       ["Yds/Game", "Pass Yds/Game", "Rush Yds/Game", "Points/Game"],
        "Yards/Play":       ["Yds/Play", "Pass Yds/Play", "Rush Yds/Play", "Points/Play"],
        "EPA/Play":         ["EPA/Play", "Pass EPA/Play", "Rush EPA/Play"],
        "Pts/Scoring Opp.": ["Points/Scoring Opp."],
        "Success Rate":     ["Success Rate", "Pass Success Rate", "Rush Success Rate"],
        "Explosiveness":    ["Explosiveness", "Pass Explosiveness", "Rush Explosiveness"],
    }

    prefix = "Off." if unit_choice == "Offense" else "Def."
    family_cols = []
    for short in METRIC_FAMILIES[metric_choice]:
        c = _col(f"{prefix} {short}")
        if c:
            family_cols.append(c)

    if not family_cols:
        st.info("No matching metric columns found for your selection.")
        st.stop()

    # Working frame
    filt_df = merged_df.copy()

    # --- Build TEAM logo map (from logos_df) ---
    url_col_candidates = ["Image URL", "Logo URL", "URL", "Logo", "Image"]
    team_logo_map = {}
    if "Team" in logos_df.columns:
        url_col_team = next((c for c in url_col_candidates if c in logos_df.columns), None)
        if url_col_team:
            team_logo_map = (
                logos_df.set_index("Team")[url_col_team]
                .dropna().astype(str).to_dict()
            )

    # --- Build CONFERENCE logo map (from same logos_df) ---
    conf_logo_map = {}
    url_col_any = next((c for c in url_col_candidates if c in logos_df.columns), None)

    # 1) Direct "Conf Name" column
    if "Conf Name" in logos_df.columns and url_col_any:
        conf_logo_map = logos_df.set_index("Conf Name")[url_col_any].dropna().astype(str).to_dict()

    # 2) If missing, try rows marked as conferences in a "Type" column
    if not conf_logo_map and "Type" in logos_df.columns and url_col_any:
        conf_rows = logos_df[logos_df["Type"].str.contains("conf", case=False, na=False)]
        key_col = "Name" if "Name" in conf_rows.columns else ("Team" if "Team" in conf_rows.columns else None)
        if key_col:
            conf_logo_map = conf_rows.set_index(key_col)[url_col_any].dropna().astype(str).to_dict()

    # 3) Last resort: treat logos_df["Team"] rows that equal conf names as conf logos
    if not conf_logo_map and "Team" in logos_df.columns and url_col_any:
        possible = logos_df[logos_df["Team"].isin(filt_df["Conf Name"].unique())]
        if not possible.empty:
            conf_logo_map = possible.set_index("Team")[url_col_any].dropna().astype(str).to_dict()

    # --- Numeric ranks (tie-aware). Offense: higher=better; Defense: lower=better. ---
    rank_cols = []
    for col in family_cols:
        hib = (prefix == "Off.")
        rk = filt_df[col].rank(ascending=not hib, method="min")
        rk_col = f"{col}__rk"
        filt_df[rk_col] = rk.astype("Int64")
        rank_cols.append(rk_col)

    # Base view (numeric under the hood)
    base_cols = [c for c in ['Rk', 'Team', 'Conf Name', 'Pwr Rtg', 'Off Rtg', 'Def Rtg'] if c in filt_df.columns]
    view = filt_df[base_cols + family_cols + rank_cols].copy()

    # --- Team logos only (click → Team Dashboards) | ASCII anchor to avoid parser issues ---
    def team_logo_link(team: str) -> str:
        url = team_logo_map.get(team, "")
        if not url:
            return ""
        safe_title = html.escape(str(team))
        return (
            f'<a href="?selected_team={quote(team)}#Team%20Dashboards" title="{safe_title}">'
            f'<img src="{html.escape(url)}" width="22" style="vertical-align:middle;border-radius:3px;"></a>'
        )
    view["Team"] = filt_df["Team"].reindex(view.index).map(team_logo_link)

    # --- Conference logos only; fallback to text if not found ---
    def conf_logo_cell(conf_name: str) -> str:
        url = conf_logo_map.get(conf_name, "")
        if url:
            safe_conf = html.escape(str(conf_name))
            return f'<img src="{html.escape(url)}" width="20" style="vertical-align:middle;border-radius:3px;" title="{safe_conf}">'
        return html.escape(conf_name) if isinstance(conf_name, str) else ""
    if "Conf Name" in view.columns:
        view["Conf Name"] = filt_df["Conf Name"].reindex(view.index).map(conf_logo_cell)

    # --- Default sort: by Rk ascending ---
    if "Rk" in view.columns:
        view = view.sort_values("Rk", ascending=True, kind="mergesort")

    # --- Display formatting ---
    # Headers (blank Team header to avoid odd glyphs)
    rename_dict = {
        "Rk": "Rk",
        "Team": "Team",               # logo-only
        "Conf Name": "Conf",      # logo-only
        "Pwr Rtg": "Pwr", "Off Rtg": "Off", "Def Rtg": "Def",
        "Off. Yds/Game": "Y/G", "Off. Pass Yds/Game": "P Y/G", "Off. Rush Yds/Game": "R Y/G", "Off. Points/Game": "Pts/G",
        "Off. Yds/Play": "Y/Pl", "Off. Pass Yds/Play": "P Y/Pl", "Off. Rush Yds/Play": "R Y/Pl", "Off. Points/Play": "Pts/Pl",
        "Off. EPA/Play": "EPA", "Off. Pass EPA/Play": "P EPA", "Off. Rush EPA/Play": "R EPA", "Off. Points/Scoring Opp.": "Pts/ScOpp",
        "Off. Success Rate": "Succ%", "Off. Pass Success Rate": "P Succ%", "Off. Rush Success Rate": "R Succ%",
        "Off. Explosiveness": "Expl", "Off. Pass Explosivenes": "P Expl", "Off. Rush Explosiveness": "R Expl",
        "Def. Yds/Game": "Y/G", "Def. Pass Yds/Game": "P Y/G", "Def. Rush Yds/Game": "R Y/G", "Def. Points/Game": "Pts/G",
        "Def. Yds/Play": "Y/Pl", "Def. Pass Yds/Play": "P Y/Pl", "Def. Rush Yds/Play": "R Y/Pl", "Def. Points/Play": "Pts/Pl",
        "Def. EPA/Play": "EPA", "Def. Pass EPA/Play": "P EPA", "Def. Rush EPA/Play": "R EPA", "Def. Points/Scoring Opp.": "Pts/ScOpp",
        "Def. Success Rate": "Succ%", "Def. Pass Success Rate": "P Succ%", "Def. Rush Success Rate": "R Succ%",
        "Def. Explosiveness": "Expl", "Def. Pass Explosivenes": "P Expl", "Def. Rush Explosiveness": "R Expl",
    }

    display_view = view.copy()

    # Metrics: "value (rank)" with 1 decimal; rates 1 decimal %
    def _fmt_metric(col_name: str) -> pd.Series:
        s = filt_df[col_name].reindex(display_view.index)
        rk = filt_df.get(f"{col_name}__rk", None)
        rk = rk.reindex(display_view.index) if rk is not None else None

        is_rate = ("Success Rate" in col_name) and ("Explosiveness" not in col_name)
        if is_rate:
            val_txt = s.map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
        else:
            val_txt = s.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        rk_txt = rk.map(lambda x: "" if (rk is None or pd.isna(x)) else f" ({int(x)})") if rk is not None else ""
        return (val_txt.fillna("") + rk_txt.fillna("")).str.strip()

    for col in family_cols:
        if col in display_view.columns:
            display_view[col] = _fmt_metric(col)

    # Ratings as TEXT with exactly TWO decimals (format BEFORE renaming)
    for raw_col in ("Pwr Rtg", "Off Rtg", "Def Rtg"):
        if raw_col in display_view.columns and raw_col in filt_df.columns:
            series_numeric = pd.to_numeric(filt_df[raw_col].reindex(display_view.index), errors="coerce")
            display_view[raw_col] = series_numeric.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    # Drop helper rank columns and rename headers
    display_view = display_view[[c for c in display_view.columns if not c.endswith("__rk")]]
    display_view.rename(columns=rename_dict, inplace=True)

    # === Gradients (Blues: darker = better; white text on dark cells) ===
    BLUE_CMAP = "Blues"
    def _goodness_series(raw: pd.Series, higher_is_better: bool) -> pd.Series:
        vals = pd.to_numeric(raw, errors="coerce")
        if not higher_is_better:
            vals = vals.max() - vals
        return vals

    def _text_contrast_from_series(raw: pd.Series, higher_is_better: bool):
        vals = pd.to_numeric(raw, errors="coerce")
        if not higher_is_better:
            vals = vals.max() - vals
        rng = (vals.max() - vals.min()) or 1.0
        norm = (vals - vals.min()) / rng
        return ['color: white' if v >= 0.6 else 'color: black' for v in norm]

    styled = display_view.style.hide(axis="index")

    # Ratings shading: Pwr/Off higher=better; Def lower=better
    ratings_cfg = [("Pwr Rtg", "Pwr", True), ("Off Rtg", "Off", True), ("Def Rtg", "Def", False)]
    for raw_col, disp_col, hib in ratings_cfg:
        if disp_col in display_view.columns and raw_col in filt_df.columns:
            gmap_vals = _goodness_series(filt_df[raw_col].reindex(view.index), hib)
            styled = styled.background_gradient(cmap=BLUE_CMAP, subset=[disp_col], gmap=gmap_vals)
            styled = styled.apply(lambda s, rc=raw_col, h=hib:
                                  _text_contrast_from_series(filt_df[rc].reindex(s.index), h),
                                  subset=[disp_col])

    # Metric family shading: offense metrics higher=better; defense metrics lower=better
    for raw_col in family_cols:
        disp_col = rename_dict.get(raw_col, raw_col)
        if disp_col not in display_view.columns:
            continue
        hib = not raw_col.startswith("Def.")
        gmap_vals = _goodness_series(filt_df[raw_col].reindex(view.index), hib)
        styled = styled.background_gradient(cmap=BLUE_CMAP, subset=[disp_col], gmap=gmap_vals)
        styled = styled.apply(lambda s, rc=raw_col, h=hib:
                              _text_contrast_from_series(filt_df[rc].reindex(s.index), h),
                              subset=[disp_col])

    # === Mobile-friendly render (no overlap) ===
    # Build per-cell classes so CSS controls wrapping responsively
    cell_classes = pd.DataFrame("", index=display_view.index, columns=display_view.columns)

    rating_cols = [c for c in ["Pwr", "Off", "Def"] if c in display_view.columns]
    metric_cols = []
    for raw_col in family_cols:
        disp = rename_dict.get(raw_col, raw_col)
        if disp in display_view.columns:
            metric_cols.append(disp)

    for c in rating_cols:
        cell_classes[c] = (cell_classes[c] + " num rating nowrap").str.strip()
    for c in metric_cols:
        cell_classes[c] = (cell_classes[c] + " num metric nowrap").str.strip()

    styled = styled.set_td_classes(cell_classes)

    TABLE_CSS = """
    <style>
    .metrics-table-wrapper { overflow-x: auto; } /* allow horizontal scroll when needed */
    .metrics-table { width: 100%; border-collapse: collapse; table-layout: fixed; }

    .metrics-table th, .metrics-table td {
      padding: 2px 4px; line-height: 1.15; font-size: 12px;
    }
    .metrics-table th { text-align: center; }
    .metrics-table img { display: inline-block; vertical-align: middle; }

    /* Numeric cells */
    .metrics-table td.num { text-align: right; }

    /* Small min-widths so numbers don't crash into borders on desktop */
    .metrics-table td.rating { min-width: 6.5ch; }  /* e.g., 26.40 */
    .metrics-table td.metric { min-width: 9ch; }    /* e.g., 610.0 (4) */

    /* Keep numbers on one line on desktop */
    .metrics-table td.nowrap { white-space: nowrap; }

    /* On phones, let columns breathe and wrap */
    @media (max-width: 640px) {
      .metrics-table { table-layout: auto; }
      .metrics-table th, .metrics-table td { font-size: 11px; padding: 2px 3px; }
      .metrics-table td.nowrap { white-space: normal; }  /* allow wrapping to prevent overlap */
      .metrics-table td.rating, .metrics-table td.metric { min-width: 0; }
    }
    </style>
    """
    st.markdown(TABLE_CSS, unsafe_allow_html=True)

    html_table = styled.set_table_attributes('class="metrics-table"').to_html()
    st.markdown(f'<div class="metrics-table-wrapper">{html_table}</div>', unsafe_allow_html=True)

#---------------------------------------------------------Team Dashboards--------------------------------------------------------
if tab_choice == "📊 Team Dashboards":
    st.markdown("## 📊 Team Dashboards")

    # Re-load the small subset we need directly from the workbook
    @st.cache_data
    def _team_basics():
        exp = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
        logos = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)[['Team','Image URL']]
        # Compute ranks (Power/Off high=good; Def low=good)
        exp['Pwr Rank'] = exp['Power Rating'].rank(ascending=False, method='min').astype(int)
        exp['Off Rank'] = exp['Offensive Rating'].rank(ascending=False, method='min').astype(int)
        exp['Def Rank'] = exp['Defensive Rating'].rank(ascending=True,  method='min').astype(int)

        # Keep only what we need
        keep = [
            'Team','Power Rating','Offensive Rating','Defensive Rating',
            'Pwr Rank','Off Rank','Def Rank',
            'Current Wins','Current Losses',
            'Projected Overall Wins','Projected Overall Losses'
        ]
        exp = exp[keep].copy()
        exp = exp.merge(logos, on='Team', how='left')
        exp.set_index('Team', inplace=True)
        return exp

    basics = _team_basics()

    all_teams = sorted(basics.index.tolist())
    # Respect preselected team from query params or session (keeps your existing behavior)
    pre_sel = st.session_state.get('selected_team')
    default_idx = all_teams.index(pre_sel) if pre_sel in all_teams else 0
    selected_team = st.selectbox("Select a Team", options=all_teams, index=default_idx)
    st.session_state['selected_team'] = selected_team

    row = basics.loc[selected_team]

    # Pretty strings
    record_txt = f"{int(row['Current Wins'])}-{int(row['Current Losses'])}"
    exp_record_txt = f"{row['Projected Overall Wins']:.1f}-{row['Projected Overall Losses']:.1f}"

    logo_url = row.get('Image URL', '') or ''
    team_name_html = f"""
    <div style="display:flex;align-items:center;gap:12px;">
        <div style="flex:0 0 56px;height:56px;border:1px solid #ddd;border-radius:6px;
                    display:flex;align-items:center;justify-content:center;overflow:hidden;background:#fff;">
            {'<img src="'+logo_url+'" style="max-width:100%;max-height:100%;">' if logo_url else ''}
        </div>
        <div style="line-height:1.15;">
            <div style="font-size:18px;font-weight:700;">{selected_team}</div>
            <div style="font-size:12px;color:#333;">
                Rank: {row['Pwr Rank']} &nbsp;&nbsp; Off Rank: {row['Off Rank']} &nbsp;&nbsp; Def Rank: {row['Def Rank']}
            </div>
            <div style="font-size:12px;color:#333;">Record: {record_txt}</div>
            <div style="font-size:12px;color:#333;">Expected Record: {exp_record_txt}</div>
        </div>
    </div>
    """

    st.markdown(team_name_html, unsafe_allow_html=True)

    # (Optional) faint divider
    st.markdown("<hr style='opacity:.2;'>", unsafe_allow_html=True)

    # -------------------------- Team Schedule (no scroll, blue headers, neutral→vs) --------------------------
    @st.cache_data
    def _team_schedule_df(team_name: str):
        sch = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Schedule', header=0)
        sch.columns = [str(c).strip() for c in sch.columns]
    
        def _pick(df, *cands):
            lower = {c.lower(): c for c in df.columns}
            for c in cands:
                if c.lower() in lower: return lower[c.lower()]
            raise KeyError(f"Missing expected column; looked for {cands}")
    
        team_col   = _pick(sch, 'Team', 'Team Name')
        game_col   = _pick(sch, 'Game')
        loc_col    = _pick(sch, 'Location', 'Loc')
        opp_col    = _pick(sch, 'Opponent', 'Opp')
        spread_col = _pick(sch, 'Spread')
        wp_col     = _pick(sch, 'Win Prob', 'Win Probability', 'WP')
    
        s = sch[sch[team_col].astype(str) == team_name].copy()
    
        # ---- Location helpers ----
        def _is_neutral(loc_raw: str) -> bool:
            t = (str(loc_raw) or '').strip().lower()
            return t.startswith('neutral')  # adapt if you have a specific value
        def _is_away(loc_raw: str) -> bool:
            t = (str(loc_raw) or '').strip().lower()
            return t in ('@', 'at', 'away')
        
        s['Away'] = s[loc_col].apply(_is_away)

        def _loc_prefix(loc_raw: str) -> str:
            t = (str(loc_raw) or '').strip().lower()
            if t.startswith('neutral'):
                return 'vs'            # display
            if t in ('@', 'at', 'away'):
                return 'at'
            return 'vs'                # home/unknown → 'vs'
    
        s['Opp Name'] = s[opp_col].astype(str).str.strip()
        s['Neutral']  = s[loc_col].apply(_is_neutral)
        s['Opponent'] = s.apply(lambda r: f"{_loc_prefix(r[loc_col])} {r['Opp Name']}".strip(), axis=1)
    
        # ---- Win Prob to 0..100 float ----
        def _to_prob_pct(x):
            if pd.isna(x): return np.nan
            if isinstance(x, str):
                xs = x.strip().replace('%', '')
                try:
                    v = float(xs); return v if v > 1 else v * 100.0
                except: return np.nan
            try:
                v = float(x); return v if v > 1 else v * 100.0
            except: return np.nan
    
        s['Win Prob'] = s[wp_col].apply(_to_prob_pct)
    
        # ---- Spread formatting (your existing logic) ----
        def _round_half(v):
            fv = float(v); return round(fv * 2) / 2.0
    
        def _invert_and_format(v):
            if pd.isna(v) or str(v).strip().lower() in ('none', ''): return ''
            txt = str(v).strip(); up = txt.upper()
            if 'WIN' in up:  return '🟢 WIN'
            if 'LOSS' in up or 'LOSE' in up: return '🔴 LOSS'
            try:
                val = float(txt.replace('+',''))
                val = -1.0 * _round_half(val)
                if abs(val) < 0.05: val = 0.0
                sign = '+' if val > 0 else ''
                return f"{sign}{val:.1f}"
            except:
                return txt
    
        s['Spread'] = s[spread_col].apply(_invert_and_format)
    
        out = (
            s.rename(columns={game_col: 'Game'})
             [['Game', 'Opponent', 'Spread', 'Win Prob', 'Opp Name', 'Neutral', 'Away']]
             .reset_index(drop=True)
        )
        return out
    
        
    sched_df = _team_schedule_df(selected_team)
    
    st.markdown("#### Schedule")

    # ---------- CSS (compact text, blue headers, pill progress bars) ----------
    st.markdown("""
    <style>
    .block-container { overflow-x: hidden !important; }
    .schedule-table {
      table-layout: fixed;
      width: 100% !important;
      border-collapse: collapse;
      font-size: 13px;
      line-height: 1.25;
    }
    .schedule-table thead th {
      background: #002060;
      color: #ffffff;
      font-weight: 600;
      padding: 6px 8px;
      border-bottom: 1px solid #c9d8ff;
      text-align: left;
    }
    .schedule-table tbody td {
      padding: 6px 8px;
      vertical-align: top;
      word-break: break-word;
      font-size: 13px;
    }
    
    /* Column sizing + alignment */
    .schedule-table thead th:nth-child(1),
    .schedule-table tbody td:nth-child(1) { /* Game */
      width: 64px; white-space: nowrap; text-align: center;
    }
    .schedule-table thead th:nth-child(2),
    .schedule-table tbody td:nth-child(2) { /* Opponent */
      width: 100%;                  /* let Opponent flex wider */
    }
    .schedule-table thead th:nth-child(3),
    .schedule-table tbody td:nth-child(3) { /* Spread */
      width: 84px; white-space: nowrap; text-align: center;
    }
    .schedule-table thead th:nth-child(4),
    .schedule-table tbody td:nth-child(4) { /* Win Prob */
      width: 110px; white-space: nowrap; text-align: center;
    }
    
    /* Win Prob cell: percent above pill bar */
    .wp-cell { display: flex; flex-direction: column; gap: 4px; align-items: center; }
    .wp-pct  { font-weight: 600; font-size: 12px; }
    .wp-track {
      width: 100%;
      height: 10px;
      background: #e7eefb;
      border-radius: 9999px;
      overflow: hidden;
    }
    .wp-fill {
      height: 100%;
      background: #1f6fe5;
      border-radius: 9999px;
    }
    /* Make schedule links look like plain text */
    .schedule-table a.sched-link {
      color: inherit !important;
      text-decoration: none !important;
      cursor: pointer;
    }
    .schedule-table a.sched-link:hover {
      text-decoration: underline; /* or remove this line if you want zero hover styling */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ---------- Data prep + HTML rendering ----------
    sched_df = _team_schedule_df(selected_team)
    
    if sched_df.empty:
        st.info("No schedule rows found for this team on the **Schedule** sheet.")
    else:
        df = sched_df.copy()
    
        if "Game" in df.columns:
            df["Game"] = df["Game"].astype("int64")
    
        if "Win Prob" in df.columns:
            wp = df["Win Prob"].astype(float)
            df["Win Prob"] = (wp * 100) if wp.max() <= 1.0 else wp
    
            def _wp_cell(x):
                pct = f"{x:.1f}%"
                return (
                    f'<div class="wp-cell">'
                    f'<div class="wp-pct">{pct}</div>'
                    f'<div class="wp-track"><div class="wp-fill" style="width:{x:.1f}%"></div></div>'
                    f'</div>'
                )
            df["Win Prob"] = df["Win Prob"].map(_wp_cell)
    
        if "Spread" in df.columns:
            df["Spread"] = df["Spread"].astype(str)
        
        from urllib.parse import urlencode, quote

        if "Opponent" in df.columns and {"Opp Name","Neutral","Away"}.issubset(df.columns):
            def _opp_link_row(row):
                # If this is an away game for the selected team, opponent is home
                is_away = bool(row["Away"])
                home = row["Opp Name"] if is_away else selected_team
                away = selected_team if is_away else row["Opp Name"]
                qs = urlencode({
                    "view": "comparison",
                    "home": home,
                    "away": away,
                    "neutral": "1" if bool(row["Neutral"]) else "0",
                }, quote_via=quote)
                return f'<a class="sched-link" href="?{qs}">{row["Opponent"]}</a>'
        
            df["Opponent"] = df.apply(_opp_link_row, axis=1)
        
        # Hide helper cols
        render_cols = [c for c in ["Game","Opponent","Spread","Win Prob"] if c in df.columns]
        df = df[render_cols]


        html = df.to_html(escape=False, index=False, classes="schedule-table")
        st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------- COMPARISON TAB ------------------------------------------------
if tab_choice == "🤝 Comparison":
    st.markdown("## 🤝 Comparison")

    # Build unified frame once
    team_frame = build_team_frame(df, metrics_df, logos_df)
    all_teams = sorted(team_frame['Team'].tolist())

    # ---- Defaults from URL/session (if present), else your existing fallbacks ----
    # Home default: comp_home > selected_team (if valid) > first team
    h_default = (
        st.session_state.get("comp_home")
        or (st.session_state.get("selected_team") if st.session_state.get("selected_team") in all_teams else None)
        or all_teams[0]
    )

    # Away default: comp_away > first team that's not home
    # (handles edge case if comp_away == h_default)
    a_default = st.session_state.get("comp_away")
    if not a_default or a_default not in all_teams or a_default == h_default:
        a_default = next(t for t in all_teams if t != h_default)

    neutral_default = st.session_state.get("comp_neutral", False)

    # ---- Controls (now honoring defaults from URL/session) ----
    csel1, csel2, csel3 = st.columns([2, 2, 1])
    with csel1:
        home_team = st.selectbox("Home Team", all_teams, index=all_teams.index(h_default))
    with csel2:
        away_team = st.selectbox("Away Team", all_teams, index=all_teams.index(a_default))
    with csel3:
        neutral = st.checkbox("Neutral site?", value=neutral_default)

    # Keep URL in sync when user changes the controls manually
    st.query_params.update({
        "view": "comparison",
        "home": home_team,
        "away": away_team,
        "neutral": "1" if neutral else "0",
    })

    # Also update session_state so subsequent navigations pick up latest choices
    st.session_state["comp_home"] = home_team
    st.session_state["comp_away"] = away_team
    st.session_state["comp_neutral"] = neutral


    # --- SIDE-BY-SIDE TEAM CARDS (always fit) + SCORE BELOW ---
    th = team_frame.set_index('Team').loc[home_team]
    ta = team_frame.set_index('Team').loc[away_team]
    total, home_score, away_score = projected_score(team_frame, home_team, away_team, neutral)

    home_logo = th['Logo'] or ""
    away_logo = ta['Logo'] or ""

    def team_html(team, logo, pwr, off, deff):
        return f"""
          <div class="team-card">
            <div class="team-head">
              <img src="{logo}" alt="logo" class="team-logo"/>
              <h3 class="team-name">{team}</h3>
            </div>
            <div class="badges"><span class="badge">Pwr</span><span class="val">{pwr}</span></div>
            <div class="badges"><span class="badge">Off</span><span class="val">{off}</span></div>
            <div class="badges"><span class="badge">Def</span><span class="val">{deff}</span></div>
          </div>
        """

    home_html = team_html(home_team, home_logo, th['Pwr Rank'], th['Off Rank'], th['Def Rank'])
    away_html = team_html(away_team, away_logo, ta['Pwr Rank'], ta['Off Rank'], ta['Def Rank'])

    st.markdown(
        f"""
        <div class="team-row">
          {home_html}
          {away_html}
        </div>

        <div class="score-block">
          <div class="score-label">Projected Score</div>
          <div class="score-main">{home_score:.2f} — {away_score:.2f}</div>
          <div class="score-sub">TOTAL {total:.2f}</div>
        </div>

        <style>
          .team-row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 16px;
            margin-bottom: 8px;
          }}
          .team-card {{
            display: grid;
            grid-template-rows: 112px 28px 28px 28px;
            flex: 1 1 160px;
            max-width: 260px;
            background: #f8f9fb;
            border-radius: 14px;
            padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
          }}
          .team-head {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
          }}
          .team-logo {{
            width: 64px; height: 64px; object-fit: contain; margin-bottom: 4px;
          }}
          .team-name {{
            margin: 0; font-size: 16px; text-align: center; line-height: 1.15;
            display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
            max-width: 200px;
          }}
          .badges {{
            display: grid;
            grid-template-columns: 42px 1fr;
            align-items: center;
            justify-content: center;
            column-gap: 8px;
            margin: 0;
          }}
          .badge {{
            background:#002060; color:#fff; border-radius:6px; padding:2px 5px; font-size:10px;
            text-align: center;
          }}
          .val {{
            font-weight: 600; font-size: 13px; text-align: left;
          }}
          .score-block {{
            text-align:center; margin: 10px 0;
          }}
          .score-label {{ font-size:12px; color:#444; }}
          .score-main  {{ font-size: 30px; font-weight: 700; }}
          .score-sub   {{ font-size: 11px; color: #666; }}
          @media (max-width: 480px) {{
            .team-card {{ grid-template-rows: 100px 26px 26px 26px; }}
            .team-logo {{ width: 50px; height: 50px; }}
            .team-name {{ font-size: 14px; }}
            .score-main {{ font-size: 24px; }}
          }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Two match-up tables
    home_ball_df, away_ball_df = build_rank_tables(team_frame, home_team, away_team)

    # Scoped CSS only for comparison tables
    st.markdown("""
    <style>
    .comparison-table table th,
    .comparison-table table td {
      white-space: nowrap !important;
      text-overflow: ellipsis;
      overflow: hidden;
      font-size: 11px !important;
      padding: 4px 6px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"### When **{home_team}** has the ball …")
    st.caption("Offensive rank (left) vs opponent defensive rank (right).")
    st.markdown(
        f"<div class='comparison-table'>{style_rank_table(home_ball_df).to_html(escape=False)}</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"### When **{away_team}** has the ball …")
    st.caption("Offensive rank (left) vs opponent defensive rank (right).")
    st.markdown(
        f"<div class='comparison-table'>{style_rank_table(away_ball_df).to_html(escape=False)}</div>",
        unsafe_allow_html=True
    )

    # Small legend
    st.markdown("""
    <div style="font-size:11px;color:#555;margin-top:.5rem">
      <b>Color scale:</b> rank #1 darkest blue across all teams; higher rank numbers lighter<br>
      <b>Parity:</b> 5 smallest |Δ Rank| across both tables highlighted light blue<br>
      <b>Advantage:</b> 5 largest |Δ Rank| across both tables highlighted light orange<br>
      <b>Expected Points (EP):</b> Each yardline is assigned a point value and measures the number of points that would be expected to be scored based on down, distance, and field position<br>
      <b>Expected Points Added (EPA):</b> Takes the EP from before a play and subtracts it from the EP after the play<br>
      <b>Explosiveness:</b> Measures the average EPA on plays which were marked as successful<br>
      <b>Success Rate:</b> Determines the success of a play. Successful plays meet one of the following criteria: <br>
      1st downs which gain at least 50% of the yards to go <br>
      2nd downs which gain at least 70% of the yards go <br>
      3rd and 4th downs which gain at least 100% of the yards to go<br>
      <b>Scoring Opportunity:</b> All offensive drives that cross the opponent's 40-yard line
      
    </div>
    """, unsafe_allow_html=True)

