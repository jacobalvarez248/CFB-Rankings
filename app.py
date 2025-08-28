import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from urllib.parse import unquote
import streamlit.components.v1 as components

st.set_page_config(page_title="CFB Rankings", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    metrics_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Metrics Tab', header=1)  # NEW
    return df, logos_df, metrics_df

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

# Tab selector with auto-switch logic
query_params = st.query_params
selected_team = query_params.get("selected_team", "")
default_tab = "📊 Team Dashboards" if selected_team else "🏆 Rankings"

tab_choice = st.radio(
    " ", 
    ["🏆 Rankings", "📈 Metrics", "📊 Team Dashboards"],
    horizontal=True, 
    label_visibility="collapsed", 
    index=0 if default_tab == "🏆 Rankings" else (2 if default_tab == "📊 Team Dashboards" else 1))

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

#-----------------------------------------------------------------METRICS COLUMN MAP--------------------------------------------------------------------------------------------------
METRIC_GROUPS = {
    ("Offense", "Yds/Game"):       [("Off. Yds/Game", "Yds/G"), ("Off. Pass Yds/Game", "Pass Yds/G"), ("Off. Rush Yds/Game", "Rush Yds/G"), ("Off. Points/Game", "Pts/G")],
    ("Defense", "Yds/Game"):       [("Def. Yds/Game", "Yds/G"), ("Def. Pass Yds/Game", "Pass Yds/G"), ("Def. Rush Yds/Game", "Rush Yds/G"), ("Def. Points/Game", "Pts/G")],

    ("Offense", "Yards/Play"):     [("Off. Yds/Play", "YPP"), ("Off. Pass Yds/Play", "Pass YPP"), ("Off. Rush Yds/Play", "Rush YPP"), ("Off. Points/Play", "Pts/Play")],
    ("Defense", "Yards/Play"):     [("Def. Yds/Play", "YPP"), ("Def. Pass Yds/Play", "Pass YPP"), ("Def. Rush Yds/Play", "Rush YPP"), ("Def. Points/Play", "Pts/Play")],

    ("Offense", "EPA/Play"):       [("Off. Points/Scoring Opp.", "Pts/ScOpp"), ("Off. EPA/Play", "EPA/P"), ("Off. Pass EPA/Play", "Pass EPA/P"), ("Off. Rush EPA/Play", "Rush EPA/P")],
    ("Defense", "EPA/Play"):       [("Def. Points/Scoring Opp.", "Pts/ScOpp"), ("Def. EPA/Play", "EPA/P"), ("Def. Pass EPA/Play", "Pass EPA/P"), ("Def. Rush EPA/Play", "Rush EPA/P")],

    ("Offense", "Success Rate"):   [("Off. Success Rate", "SR"), ("Off. Pass Success Rate", "Pass SR"), ("Off. Rush Success Rate", "Rush SR")],
    ("Defense", "Success Rate"):   [("Def. Success Rate", "SR"), ("Def. Pass Success Rate", "Pass SR"), ("Def. Rush Success Rate", "Rush SR")],

    ("Offense", "Explosiveness"):  [("Off. Explosiveness", "Expl"), ("Off. Pass Explosiveness", "Pass Expl"), ("Off. Rush Explosiveness", "Rush Expl")],
    ("Defense", "Explosiveness"):  [("Def. Explosiveness", "Expl"), ("Def. Pass Explosiveness", "Pass Expl"), ("Def. Rush Explosiveness", "Rush Expl")],
}

# The rating column to include right after the three fixed columns
UNIT_RATING = {
    "Offense": ("Offensive Rating", "Off Rtg"),
    "Defense": ("Defensive Rating", "Def Rtg"),
}

#-----------------------------------------------------METRICS TAB------------------------------------------------
st.markdown("## 📈 Metrics")

# Controls (you already set these up—keeping them here for clarity)
c1, c2 = st.columns(2)
with c1:
    unit_choice = st.selectbox("Unit", ["Offense", "Defense"], key="metrics_unit")
with c2:
    metric_choice = st.selectbox("Metric", ["Yds/Game", "Yards/Play", "EPA/Play", "Success Rate", "Explosiveness"], key="metrics_metric")

# --- Build base table from the Metrics Tab ---
# Start with the three fixed columns from your main df:
# Rk, Team (logo-only), Pwr Rtg (sorted descending by default)
# Re-use your logos_df to render the tiny team logo like Rankings. :contentReference[oaicite:2]{index=2}
logos_map = logos_df.set_index("Team")["Image URL"].to_dict()
base = df.copy()

# Ensure required columns exist
for col in ["Rk", "Pwr Rtg"]:
    if col not in base.columns:
        base[col] = pd.NA

# Logo-only Team column (small to fit phone)
base["Team"] = base.index.to_series().map(lambda name: f'<img src="{logos_map.get(name, "")}" width="18">' if logos_map.get(name) else "")

# Default sort by Pwr Rtg descending
base = base.sort_values(by="Pwr Rtg", ascending=False, kind="mergesort")

# --- Attach the rating column for the selected unit ---
rating_src, rating_short = UNIT_RATING[unit_choice]
if rating_src in metrics_df.columns:
    rating_series = metrics_df.set_index("Team")[rating_src]
else:
    rating_series = pd.NA
base[rating_short] = base.index.to_series().map(lambda t: rating_series.loc[t] if t in rating_series.index else pd.NA)

# --- Determine the metric columns to show next ---
cols_spec = METRIC_GROUPS[(unit_choice, metric_choice)]  # list of (source_name_in_xlsx, short_header)

# Helper: compute rank text to append in parens
def add_rank(series: pd.Series, offense: bool) -> pd.Series:
    # offense: rank descending (bigger→1); defense: ascending (smaller→1)
    asc = not offense
    # ranks among all valid teams (ignore NaN)
    ranks = series.rank(ascending=asc, method="min")
    return series, ranks

# Helper: format a value with rank, picking decimals/percent rules
def fmt_value(val, rank, is_rate: bool):
    if pd.isna(val):
        return ""
    # metrics formatting: one decimal for most; percent for rates
    if is_rate:
        # if values are 0–1, show as %; else assume already in 0–100
        out = f"{val*100:.1f}%" if 0 <= val <= 1 else f"{val:.1f}%"
    else:
        out = f"{val:.1f}"
    # tiny rank in parens
    return f'{out} <span style="font-size:10px;opacity:.7">({int(rank)})</span>'

# Which metrics are "rates" (show as %)
def is_rate_header(h: str) -> bool:
    return "SR" in h or "Success" in h

# Build the dynamic metric columns
offense = (unit_choice == "Offense")
for src_col, short in cols_spec:
    series = metrics_df.set_index("Team")[src_col] if src_col in metrics_df.columns else pd.Series(dtype=float)
    # map onto base index (teams)
    aligned = base.index.to_series().map(lambda t: series.loc[t] if t in series.index else pd.NA).astype("float64")
    vals, ranks = add_rank(aligned, offense=offense)
    base[short] = [
        fmt_value(v, r, is_rate=is_rate_header(short))
        for v, r in zip(vals, ranks)
    ]

# Select final visible columns:
# Rk | Team(logo) | Pwr Rtg | Off/Def Rtg | (chosen metrics…)
final_cols = ["Rk", "Team", "Pwr Rtg", rating_short] + [short for _, short in cols_spec]
view = base[final_cols].copy()

# Styling: compact, no side scrolling
st.markdown("""
<style>
.block-container { padding-left: .5rem !important; padding-right: .5rem !important; }
table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 11px; border-collapse: collapse; }
td, th { padding: 4px !important; text-align: center !important; vertical-align: middle !important; font-size: 11px; }
thead th { background-color: #002060 !important; color: #ffffff !important; font-weight: 500 !important; font-size: 10px !important; padding: 1px 3px !important; }
td img { display:block; margin:0 auto; }
</style>
""", unsafe_allow_html=True)

# Render
st.write(
    view.style.hide(axis="index").to_html(escape=False),
    unsafe_allow_html=True
)

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
