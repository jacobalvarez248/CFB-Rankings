import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from urllib.parse import unquote
import streamlit.components.v1 as components

st.set_page_config(page_title="CFB Rankings", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Metrics', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)

    # Clean up column names
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace(r'\s+', ' ', regex=True)
    logos_df.columns = logos_df.columns.str.strip()

    return df, logos_df

df, logos_df = load_data()

# Deduplicate and sanitize columns

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

df.columns = pd.Index([str(c) for c in deduplicate_columns(df.columns)])
df = df.loc[:, ~df.columns.str.contains(r'\.(1|2|3|4)$')]

# Ensure 'Team' column exists for merge
if 'Team' not in df.columns and 'Team Name' in df.columns:
    df.rename(columns={'Team Name': 'Team'}, inplace=True)

# Ensure 'Team' exists before merge
if 'Team' not in df.columns:
    df['Team'] = df.index

df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')


# Standardize columns
if 'Current Rank' in df.columns:
    df['Current Rank'] = df['Current Rank'].astype('Int64')

# Set index and alias
df['Team Name'] = df['Team']
df.set_index('Team Name', inplace=True)

# Conference logo map
conf_logo_map = logos_df.set_index('Team')['Image URL'].to_dict()
df['Conference Logo'] = df.get('Conference', pd.NA).apply(
    lambda conf: f'<img src="{conf_logo_map.get(conf, '')}" width="15">' if conf_logo_map.get(conf) else (conf if pd.notna(conf) else '')
)
df['Conf Name'] = df.get('Conference', pd.NA)

# Drop unnecessary columns
df.drop(columns=[
    "Conference", "Image URL", "Vegas Win Total",
    "Projected Overall Wins", "Projected Overall Losses",
    "Projected Conference Wins", "Projected Conference Losses",
    "Schedule Difficulty Rank", "Column1", "Column3", "Column5"
], errors='ignore', inplace=True)

# Rename key columns
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

# Reorder key columns
first_cols = ["Pre Rk", "Rk", "Team", "Conf"]
existing = [c for c in df.columns if c not in first_cols]
df = df[[c for c in first_cols if c in df.columns] + existing]

# Show cleaned columns for debugging
st.write("\u2705 Cleaned df columns:", df.columns.tolist())

# Parse query string for selected team
query_params = st.query_params
preselect_team = unquote(query_params.get("selected_team", ""))
if 'selected_team' not in st.session_state:
    st.session_state['selected_team'] = preselect_team if preselect_team else df.index[0]

# Tab selection logic
selected_team = query_params.get("selected_team", "")
default_tab = "\ud83c\udfcb\ufe0f Team Dashboards" if selected_team else "\ud83c\udfc6 Rankings"

tab_choice = st.radio(
    " ",
    ["\ud83c\udfc6 Rankings", "\ud83d\udcca Metrics", "\ud83c\udfcb\ufe0f Team Dashboards"],
    horizontal=True,
    label_visibility="collapsed",
    index=0 if default_tab == "\ud83c\udfc6 Rankings" else (2 if default_tab == "\ud83c\udfcb\ufe0f Team Dashboards" else 1)
)


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
    
#--------------- METRICS TAB: Advanced Metrics Table --------------------------------------------------------#
if tab_choice == "📈 Metrics":
    st.markdown("## 📈 Metrics")

    c1, c2 = st.columns(2)
    with c1:
        unit_choice = st.selectbox("Unit", ["Offense", "Defense"], key="metrics_unit")
    with c2:
        metric_choice = st.selectbox(
            "Metric",
            ["Yards/Game", "Yards/Play", "EPA/Play", "Success Rate", "Explosiveness"],
            key="metrics_metric"
        )

    # Always visible columns
    base_cols = ['Rk', 'Team', 'Pwr Rtg']

    # Rating column based on unit
    rating_col = 'Off Rtg' if unit_choice == 'Offense' else 'Def Rtg'
    rating_short = 'Off Rtg' if unit_choice == 'Offense' else 'Def Rtg'

    # Metrics mapping
    metrics_map = {
        "Yards/Game": {
            "Offense": ['Yds/G', 'Pass Y/G', 'Rush Y/G', 'Pts/G'],
            "Defense": ['Yds/G', 'Pass Y/G', 'Rush Y/G', 'Pts/G']
        },
        "Yards/Play": {
            "Offense": ['Yds/P', 'Pass Y/P', 'Rush Y/P', 'Pts/P'],
            "Defense": ['Yds/P', 'Pass Y/P', 'Rush Y/P', 'Pts/P']
        },
        "EPA/Play": {
            "Offense": ['ScOpp', 'EPA/P', 'Pass EPA', 'Rush EPA'],
            "Defense": ['ScOpp', 'EPA/P', 'Pass EPA', 'Rush EPA']
        },
        "Success Rate": {
            "Offense": ['Suc %', 'Pass %', 'Rush %'],
            "Defense": ['Suc %', 'Pass %', 'Rush %']
        },
        "Explosiveness": {
            "Offense": ['Expl', 'Pass Ex', 'Rush Ex'],
            "Defense": ['Expl', 'Pass Ex', 'Rush Ex']
        }
    }

    # Map back to real column names in df
    col_lookup = {
        'Yds/G': 'Off. Yds/Game',
        'Pass Y/G': 'Off. Pass Yds/Game',
        'Rush Y/G': 'Off. Rush Yds/Game',
        'Pts/G': 'Off. Points/Game',
        'Yds/P': 'Off. Yds/Play',
        'Pass Y/P': 'Off. Pass Yds/Play',
        'Rush Y/P': 'Off. Rush Yds/Play',
        'Pts/P': 'Off. Points/Play',
        'ScOpp': 'Off. Points/Scoring Opp.',
        'EPA/P': 'Off. EPA/Play',
        'Pass EPA': 'Off. Pass EPA/Play',
        'Rush EPA': 'Off. Rush EPA/Play',
        'Suc %': 'Off. Success Rate',
        'Pass %': 'Off. Pass Success Rate',
        'Rush %': 'Off. Rush Success Rate',
        'Expl': 'Off. Explosiveness',
        'Pass Ex': 'Off. Pass Explosivenes',
        'Rush Ex': 'Off. Rush Explosiveness'
    }

    # Adjust for defense
    if unit_choice == 'Defense':
        for k in list(col_lookup.keys()):
            col_lookup[k] = col_lookup[k].replace("Off.", "Def.")
    
    selected_short = metrics_map[metric_choice][unit_choice]
    selected_cols = [col_lookup[c] for c in selected_short if col_lookup[c] in df.columns]

    # Debug: Show which metric columns are selected
    st.write("🔎 selected_short (metric labels):", selected_short)
    st.write("🔎 selected_cols (from df):", selected_cols)
    st.write("🔎 All columns in df:", df.columns.tolist())
    
    # Optional: Show missing ones for clarity
    missing_cols = [col_lookup[c] for c in selected_short if col_lookup[c] not in df.columns]
    st.warning(f"⚠️ Missing from DataFrame: {missing_cols}")

    display_df = df[['Rk', 'Team', 'Pwr Rtg', rating_col] + selected_cols].copy()
    display_df.rename(columns={rating_col: rating_short}, inplace=True)

    # Add ranks in parentheses
    def add_rank(series, inverse=False, is_percent=False):
        ranked = series.rank(ascending=not inverse, method='min').astype("Int64")
        if is_percent:
            value_fmt = series.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '')
        else:
            value_fmt = series.map(lambda x: f"{x:.1f}" if pd.notna(x) else '')
        return value_fmt + ranked.map(lambda r: f" ({r})" if pd.notna(r) else '')

    for col in selected_cols:
        if 'Rate' in col or 'Success' in col or '%' in col:
            display_df[col] = add_rank(df[col], inverse=(unit_choice == 'Defense'), is_percent=True)
        else:
            display_df[col] = add_rank(df[col], inverse=(unit_choice == 'Defense'))
    
    display_df[rating_short] = df[rating_col].map(lambda x: f"{x:.1f}" if pd.notna(x) else '')

    # Replace team name with logo
    def team_logo_html(team):
        url = logos_df.set_index("Team").at[team, "Image URL"]
        return f'<img src="{url}" width="22">' if url else team
    display_df['Team'] = display_df.index.map(team_logo_html)

    # Format columns
    styled = display_df.style.hide(axis="index")

    # Reduce font size and set tight layout
    st.markdown("""
    <style>
    .block-container { padding-left: .5rem !important; padding-right: .5rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 11px; border-collapse: collapse; }
    td, th { padding: 4px !important; text-align: center !important; vertical-align: middle !important; font-size: 10px; }
    thead th {
      background-color: #004080 !important;
      color: #ffffff !important;
      font-weight: 500 !important;
      font-size: 9.5px !important;
    }
    td img { display: block; margin: 0 auto; }
    </style>
    """, unsafe_allow_html=True)

    st.write(styled.to_html(escape=False), unsafe_allow_html=True)

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
