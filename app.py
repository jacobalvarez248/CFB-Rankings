import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from urllib.parse import unquote
import streamlit.components.v1 as components

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

#-----------------------------------------------------METRICS TAB------------------------------------------------
if tab_choice == "📈 Metrics":
    st.markdown("## 📈 Metrics")

    c1, c2 = st.columns(2)
    with c1:
        unit_choice = st.selectbox("Unit", ["Offense", "Defense"], key="metrics_unit")
    with c2:
        metric_choice = st.selectbox("Metric", ["Yards/Game", "Yards/Play", "EPA/Play", "Success Rate", "Explosiveness"], key="metrics_metric")

    # Clean column names
    df.columns = df.columns.str.strip()
    metrics_df.columns = metrics_df.columns.str.strip()

    # Merge Expected Wins data with Metrics data
    core_cols = ['Team', 'Rk', 'Pwr Rtg', 'Off Rtg', 'Def Rtg']
    df_core = df[[col for col in core_cols if col in df.columns]].copy()

    # Now try to merge (after checking columns)
    merged_df = pd.merge(df_core, metrics_df, on='Team', how='inner')

    # Sort BEFORE formatting to string
    if 'Pwr Rtg' in merged_df.columns:
        merged_df.sort_values("Pwr Rtg", ascending=False, inplace=True)

    # Dropdown logic
    metric_map = {
        "Yards/Game": {
            "Offense": ["Off. Yds/Game", "Off. Pass Yds/Game", "Off. Rush Yds/Game", "Off. Points/Game"],
            "Defense": ["Def. Yds/Game", "Def. Pass Yds/Game", "Def. Rush Yds/Game", "Def. Points/Game"],
        },
        "Yards/Play": {
            "Offense": ["Off. Yds/Play", "Off. Pass Yds/Play", "Off. Rush Yds/Play", "Off. Points/Play"],
            "Defense": ["Def. Yds/Play", "Def. Pass Yds/Play", "Def. Rush Yds/Play", "Def. Points/Play"],
        },
        "EPA/Play": {
            "Offense": ["Off. Points/Scoring Opp.", "Off. EPA/Play", "Off. Pass EPA/Play", "Off. Rush EPA/Play"],
            "Defense": ["Def. Points/Scoring Opp.", "Def. EPA/Play", "Def. Pass EPA/Play", "Def. Rush EPA/Play"],
        },
        "Success Rate": {
            "Offense": ["Off. Success Rate", "Off. Pass Success Rate", "Off. Rush Success Rate"],
            "Defense": ["Def. Success Rate", "Def. Pass Success Rate", "Def. Rush Success Rate"],
        },
        "Explosiveness": {
            "Offense": ["Off. Explosiveness", "Off. Pass Explosivenes", "Off. Rush Explosiveness"],
            "Defense": ["Def. Explosiveness", "Def. Pass Explosivenes", "Def. Rush Explosiveness"],
        },
    }

    base_cols = ["Rk", "Pwr Rtg"]
    extra = ["Off Rtg"] if unit_choice == "Offense" else ["Def Rtg"]
    metric_cols = metric_map[metric_choice][unit_choice]
    columns_to_show = base_cols + extra + metric_cols

    # Filter columns safely
    available_cols = [col for col in columns_to_show if col in merged_df.columns]
    missing_cols = [col for col in columns_to_show if col not in merged_df.columns]
    if missing_cols:
        st.warning(f"Missing columns in data: {', '.join(missing_cols)}")

    view = merged_df[available_cols].copy()

    # Ranking logic
    ranks = {}
    for col in metric_cols:
        if col not in view.columns:
            continue
        if "Def" in col:
            ranks[col] = {v: i + 1 for i, v in enumerate(sorted(view[col].dropna()))}
        else:
            ranks[col] = {v: i + 1 for i, v in enumerate(sorted(view[col].dropna(), reverse=True))}

    def format_cell(col, val):
        if pd.isna(val): 
            return ""
        # "Rate" columns are percentages, but Explosiveness is NOT
        is_rate = ("Rate" in col) and ("Explosiveness" not in col)
    
        # one decimal for numbers, one decimal % for rates
        val_fmt = f"{val:.1%}" if is_rate else f"{val:.1f}"
        rk = ranks.get(col, {}).get(val, "")
        return f"{val_fmt} ({rk})" if rk else val_fmt

    for col in metric_cols:
        if col in view.columns:
            view[col] = view[col].apply(lambda v: format_cell(col, v))

    # Add logo only for Team
    view.insert(1, 'Team', merged_df['Team'].map(
        lambda team: f'<img src="{logos_df.set_index("Team").at[team, "Image URL"]}" width="20">' 
        if team in logos_df.set_index("Team").index else team
    ))

    # Rename for mobile display
    rename_dict = {
        "Rk": "Rk", "Pwr Rtg": "Pwr", "Off Rtg": "Off", "Def Rtg": "Def",
        "Off. Yds/Game": "Y/G", "Off. Pass Yds/Game": "P Y/G", "Off. Rush Yds/Game": "R Y/G", "Off. Points/Game": "Pts/G",
        "Off. Yds/Play": "Y/Pl", "Off. Pass Yds/Play": "P Y/Pl", "Off. Rush Yds/Play": "R Y/Pl", "Off. Points/Play": "Pts/Pl",
        "Off. EPA/Play": "EPA", "Off. Pass EPA/Play": "P EPA", "Off. Rush EPA/Play": "R EPA", "Off. Points/Scoring Opp.": "Pts/ScOpp",
        "Off. Success Rate": "Succ%", "Off. Pass Success Rate": "P Succ%", "Off. Rush Success Rate": "R Succ%",
        "Off. Explosiveness": "Expl", "Off. Pass Explosivenes": "P Expl", "Off. Rush Explosiveness": "R Expl",
        "Def. Yds/Game": "Y/G", "Def. Pass Yds/Game": "P Y/G", "Def. Rush Yds/Game": "R Y/G", "Def. Points/Game": "Pts/G",
        "Def. Yds/Play": "Y/Pl", "Def. Pass Yds/Play": "P Y/Pl", "Def. Rush Yds/Play": "R Y/Pl", "Def. Points/Play": "Pts/Pl",
        "Def. EPA/Play": "EPA", "Def. Pass EPA/Play": "P EPA", "Def. Rush EPA/Play": "R EPA", "Def. Points/Scoring Opp.": "Pts/ScOpp",
        "Def. Success Rate": "Succ%", "Def. Pass Success Rate": "P Succ%", "Def. Rush Success Rate": "R Succ%",
        "Def. Explosiveness": "Expl%", "Def. Pass Explosivenes": "P Expl%", "Def. Rush Explosiveness": "R Expl%",
    }
    view.rename(columns=rename_dict, inplace=True)

    # Format numeric columns
    numeric_cols = ["Pwr", "Off", "Def"]
    for col in numeric_cols:
        if col in view.columns:
            view[col] = view[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    # Render table
    st.markdown("""
    <style>
    table {
      width: 100%;
      table-layout: fixed;
      font-size: 9px;
    }
    td, th {
      padding: 3px;
      text-align: center !important;
      vertical-align: middle;
      word-wrap: break-word;
      font-size: 9px;
    }
    thead th {
      background-color: #002060;
      color: white;
      font-weight: 600;
      font-size: 9px;
      text-align: center !important;
      vertical-align: middle;
    }
    td img {
      display: block;
      margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.write(view.to_html(escape=False, index=False), unsafe_allow_html=True)

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
