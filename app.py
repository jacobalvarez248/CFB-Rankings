import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from urllib.parse import unquote, quote
import streamlit.components.v1 as components
import numpy as np

st.set_page_config(page_title="CFB Rankings", layout="wide", initial_sidebar_state="expanded")

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
default_tab = "📊 Team Dashboards" if selected_team else "🏆 Rankings"

tab_choice = st.radio(
    " ",
    ["🏆 Rankings", "📈 Metrics", "📊 Team Dashboards", "🤝 Comparison"],
    horizontal=True,
    label_visibility="collapsed",
    index=0 if default_tab == "🏆 Rankings" else (3 if default_tab == "🤝 Comparison" else (2 if default_tab == "📊 Team Dashboards" else 1)))

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
    from urllib.parse import quote
    st.markdown("## 📈 Metrics")

    # === Controls ===
    c1, c2 = st.columns(2)
    with c1:
        unit_choice = st.selectbox("Unit", ["Offense", "Defense"], key="metrics_unit")
    with c2:
        metric_choice = st.selectbox(
            "Metric",
            ["Yards/Game", "Yards/Play", "EPA/Play", "Pts/Scoring Opp.", "Success Rate", "Explosiveness"],
            key="metrics_metric",
        )

    # Clean column names
    df.columns = df.columns.str.strip()
    metrics_df.columns = metrics_df.columns.str.strip()

    # Merge core team/rating info with the Metrics sheet
    core_cols = ['Team', 'Rk', 'Pwr Rtg', 'Off Rtg', 'Def Rtg', 'Conf Name']
    df_core = df[[c for c in core_cols if c in df.columns]].copy()
    merged_df = pd.merge(df_core, metrics_df, on='Team', how='inner')

    # Which metrics to show?
    def _col(name: str):
        candidates = [name, name.replace("Explosiveness", "Explosivenes")]
        for c in candidates:
            if c in merged_df.columns:
                return c
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
        if c: family_cols.append(c)

    if not family_cols:
        st.info("No matching metric columns found for your selection.")
        st.stop()

    # Working frame
    filt_df = merged_df.copy()

    # Ranks: Offense higher=better, Defense lower=better
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

    # --- Team logos only (clickable → Team Dashboards) ---
    logos_map = logos_df.set_index("Team")["Image URL"].to_dict()
    def team_logo_link(team: str) -> str:
        url = logos_map.get(team, "")
        if not url:
            return ""
        return (
            f'<a href="?selected_team={quote(team)}#📊%20Team%20Dashboards" title="{team}">'
            f'<img src="{url}" width="22" style="vertical-align:middle;border-radius:3px;"></a>'
        )
    view["Team"] = filt_df["Team"].reindex(view.index).map(team_logo_link)

    # --- Conference logos only (fallback blank) ---
    # Expect conf_logos_df with ["Conf Name","Image URL"] OR a dict CONF_LOGOS = {"SEC": "...png", ...}
    try:
        conf_logos_map = conf_logos_df.set_index("Conf Name")["Image URL"].to_dict()
    except Exception:
        conf_logos_map = globals().get("CONF_LOGOS", {})
    def conf_logo_cell(conf_name: str) -> str:
        url = conf_logos_map.get(conf_name, "")
        if not url:
            return ""
        return f'<img src="{url}" width="20" style="vertical-align:middle;border-radius:3px;" title="{conf_name}">'
    if "Conf Name" in view.columns:
        view["Conf Name"] = filt_df["Conf Name"].reindex(view.index).map(conf_logo_cell)

    # Display headers
    rename_dict = {
        "Rk": "Rk",
        "Team": "",               # logo-only column → blank header looks cleaner
        "Conf Name": "Conf",      # logo-only conf column
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
    reverse_rename = {v: k for k, v in rename_dict.items()}

    # --- Sorting UI ---
    display_cols = [rename_dict.get(c, c) for c in base_cols + family_cols]
    sort_target_display = st.selectbox(
        "Sort by",
        options=display_cols,
        index=display_cols.index("Rk") if "Rk" in display_cols else 0,
        key="metrics_sort_by",
    )
    asc_box = st.checkbox("Ascending", value=True, key="metrics_sort_asc")

    base_col = reverse_rename.get(sort_target_display, sort_target_display)
    is_rank_sort = base_col in family_cols
    sort_col = f"{base_col}__rk" if is_rank_sort else base_col
    if sort_col not in view.columns and base_col in view.columns:
        sort_col = base_col
    # For ranks, "Ascending" means best-first (1 → …)
    actual_ascending = (True if asc_box else False) if is_rank_sort else asc_box
    view = view.sort_values(sort_col, ascending=actual_ascending, kind="mergesort")

    # --- Format for display ---
    def _fmt_metric(col_name: str) -> pd.Series:
        """'value (rank)' with 1 decimal; rates use 1 decimal %."""
        s = filt_df[col_name].reindex(view.index)
        rk = filt_df.get(f"{col_name}__rk", None)
        rk = rk.reindex(view.index) if rk is not None else None

        is_rate = ("Success Rate" in col_name) and ("Explosiveness" not in col_name)
        if is_rate:
            val_txt = s.map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
        else:
            val_txt = s.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        rk_txt = rk.map(lambda x: "" if (rk is None or pd.isna(x)) else f" ({int(x)})") if rk is not None else ""
        return (val_txt.fillna("") + rk_txt.fillna("")).str.strip()

    display_view = view.copy()
    for col in family_cols:
        if col in display_view.columns:
            display_view[col] = _fmt_metric(col)

    # Ratings as TEXT with exactly TWO decimals (for display)
    for raw_col, disp_col in [("Pwr Rtg", "Pwr"), ("Off Rtg", "Off"), ("Def Rtg", "Def")]:
        if raw_col in filt_df.columns:
            series_numeric = pd.to_numeric(filt_df[raw_col].reindex(view.index), errors="coerce")
            if disp_col in display_view.columns:
                display_view[disp_col] = series_numeric.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    # Drop helper rank cols and rename headers
    display_view = display_view[[c for c in display_view.columns if not c.endswith("__rk")]]
    display_view.rename(columns=rename_dict, inplace=True)

    # === Gradients ===
    BLUE_CMAP = "Blues"
    def _goodness_series(raw: pd.Series, hib: bool) -> pd.Series:
        vals = pd.to_numeric(raw, errors="coerce")
        if not hib:
            vals = vals.max() - vals
        return vals
    def _text_contrast_from_series(raw: pd.Series, hib: bool):
        vals = pd.to_numeric(raw, errors="coerce")
        if not hib:
            vals = vals.max() - vals
        rng = (vals.max() - vals.min()) or 1.0
        norm = (vals - vals.min()) / rng
        return ['color: white' if v >= 0.6 else 'color: black' for v in norm]

    styled = display_view.style.hide(axis="index")

    # Ratings shading: Pwr/Off higher=better; Def lower=better
    for raw_col, disp_col, hib in [("Pwr Rtg", "Pwr", True), ("Off Rtg", "Off", True), ("Def Rtg", "Def", False)]:
        if disp_col in display_view.columns and raw_col in filt_df.columns:
            gmap_vals = _goodness_series(filt_df[raw_col].reindex(view.index), hib)
            styled = styled.background_gradient(cmap=BLUE_CMAP, subset=[disp_col], gmap=gmap_vals)
            styled = styled.apply(lambda s, rc=raw_col, h=hib:
                                  _text_contrast_from_series(filt_df[rc].reindex(s.index), h),
                                  subset=[disp_col])

    # Metric shading: offense metrics higher=better; defense metrics lower=better
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

    # === Mobile-friendly render (no side scroll; tighter text; nowrap numbers) ===
    TABLE_CSS = """
    <style>
    .metrics-table-wrapper { overflow-x: hidden; }
    .metrics-table { table-layout: fixed; width: 100%; border-collapse: collapse; }
    .metrics-table th, .metrics-table td { padding: 2px 4px; line-height: 1.1; font-size: 12px; }
    .metrics-table th { text-align: center; }
    .metrics-table img { display: inline-block; vertical-align: middle; }
    @media (max-width: 480px) {
      .metrics-table th, .metrics-table td { font-size: 11px; padding: 2px 3px; }
    }
    </style>
    """
    st.markdown(TABLE_CSS, unsafe_allow_html=True)

    # Keep numeric columns on one line (reduce wrapping)
    nowrap_display_cols = []
    for disp_col in ("Pwr", "Off", "Def"):
        if disp_col in display_view.columns: nowrap_display_cols.append(disp_col)
    for raw_col in family_cols:
        disp_col = rename_dict.get(raw_col, raw_col)
        if disp_col in display_view.columns: nowrap_display_cols.append(disp_col)
    if nowrap_display_cols:
        styled = styled.set_properties(subset=nowrap_display_cols, **{"white-space": "nowrap"})

    html_table = styled.set_table_attributes('class="metrics-table"').to_html()
    st.markdown(f'<div class="metrics-table-wrapper">{html_table}</div>', unsafe_allow_html=True)

    st.caption("Ascending = best rank first (1 → …)" if is_rank_sort else "Ascending = low → high")

#---------------------------------------------------------Team Dashboards--------------------------------------------------------
if tab_choice == "📊 Team Dashboards":
    st.markdown("## 📊 Team Dashboards")
    all_teams = df.index.tolist()
    if st.session_state['selected_team'] in all_teams:
        default_index = all_teams.index(st.session_state['selected_team'])
    else:
        default_index = 0

    selected_team = st.selectbox("Select a Team", options=all_teams, index=default_index)
    st.session_state['selected_team'] = selected_team

    team_data = df.loc[[selected_team]]
    st.markdown(f"### Dashboard for {selected_team}")
    st.dataframe(team_data.T, use_container_width=True)

# ----------------------------------------------------- COMPARISON TAB ------------------------------------------------
if tab_choice == "🤝 Comparison":
    st.markdown("## 🤝 Comparison")

    # Build unified frame once
    team_frame = build_team_frame(df, metrics_df, logos_df)

    all_teams = sorted(team_frame['Team'].tolist())

    csel1, csel2, csel3 = st.columns([2,2,1])
    with csel1:
        home_team = st.selectbox(
            "Home Team",
            all_teams,
            index=all_teams.index(st.session_state.get('selected_team', all_teams[0]))
            if st.session_state.get('selected_team') in all_teams else 0
        )
    with csel2:
        away_team = st.selectbox("Away Team", all_teams, index=0 if all_teams[0] != home_team else 1)
    with csel3:
        neutral = st.checkbox("Neutral site?", value=False)

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

