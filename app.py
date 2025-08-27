import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(page_title="CFB Rankings", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------
# Load Excel data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# ---------------------------------
# Utilities
# ---------------------------------
def deduplicate_columns(columns):
    """Make column names unique: Name, Name.1, Name.2, ..."""
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

# ---------------------------------
# Normalize columns & merge logos
# ---------------------------------
df.columns = deduplicate_columns(df.columns)
# Drop any duplicate-suffixed cols that might still exist
df = df.loc[:, ~df.columns.str.contains(r'\.(1|2|3|4)$')]

# Merge in team logo URL
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# Convert rank to integer (nullable safe)
if 'Current Rank' in df.columns:
    df['Current Rank'] = df['Current Rank'].astype('Int64')

# Save team name for index and set it NOW (before any subsetting later)
df['Team Name'] = df['Team']
df.set_index('Team Name', inplace=True)

# Build team logo HTML
df['Team Logo'] = df['Image URL'].apply(lambda u: f'<img src="{u}" width="15">' if pd.notna(u) else '')

# Build conference logos from the same Logos sheet (expects rows like "SEC", "Big Ten", etc.)
conf_logo_map = logos_df.set_index('Team')['Image URL'].to_dict()
if 'Conference' in df.columns:
    df['Conference Logo'] = df['Conference'].apply(
        lambda conf: f'<img src="{conf_logo_map.get(conf, "")}" width="15">' if conf_logo_map.get(conf) else (conf if pd.notna(conf) else '')
    )
else:
    df['Conference Logo'] = ''

# Keep a plain-text conference name for filtering before we drop the text column
if 'Conference' in df.columns:
    df['Conf Name'] = df['Conference']
else:
    df['Conf Name'] = pd.NA

# --- Drop unwanted columns ---
df.drop(columns=[
    "Conference",                # drop text version, keep only logo
    "Team",                      # drop team text, keep only logo
    "Image URL",                 # raw logo URL not needed
    "Vegas Win Total",
    "Projected Overall Wins",
    "Projected Overall Losses",
    "Projected Conference Wins",
    "Projected Conference Losses",
    "Schedule Difficulty Rank",  # spelling-safe, drops if present
    "Column1", "Column3", "Column5"
], errors='ignore', inplace=True)

# --- Rename columns ---
df.rename(columns={
    "Preseason Rank": "Pre Rk",
    "Current Rank": "Rk",
    "Team Logo": "Team",
    "Conference Logo": "Conf",
    "Power Rating": "Pwr Rtg",
    "Offensive Rating": "Off Rtg",
    "Defensive Rating": "Def Rtg",
    "Current Wins": "W",
    "Current Losses": "L",
    "Schedule Difficulty": "Sched Diff"
}, inplace=True)

# --- Reorder cleanly ---
first_cols = ["Pre Rk", "Rk", "Team", "Conf"]
existing = [c for c in df.columns if c not in first_cols]
ordered = [c for c in first_cols if c in df.columns] + existing
df = df[ordered]

# ---------------------------------
# Sidebar controls (filters + sorting)
# ---------------------------------
with st.sidebar:
    st.header("Filters & Sort")

    # Team substring search (uses the index, which you set to 'Team Name')
    team_query = st.text_input("Team contains", value="")

    # Conference multiselect (based on the preserved text column)
    conf_options = sorted([c for c in df['Conf Name'].dropna().unique()])
    conf_selected = st.multiselect("Conference", conf_options)

    # Choose sort column (include text helpers for better UX)
    sortable_cols = [c for c in df.columns if c not in ['Team', 'Conf']] + ['Team Name', 'Conf Name']
    primary_sort = st.selectbox("Sort by", options=sortable_cols, index=sortable_cols.index('Rk') if 'Rk' in sortable_cols else 0)
    sort_ascending = st.checkbox("Ascending", value=True)

# Start from the working view
view = df.copy()

# Apply Team filter
if team_query:
    view = view[view.index.str.contains(team_query, case=False, na=False)]

# Apply Conference filter
if conf_selected:
    view = view[view['Conf Name'].isin(conf_selected)]

# Sorting: if user chose helper columns that aren't displayed (Team Name / Conf Name), they still work
if primary_sort in view.columns:
    view = view.sort_values(by=primary_sort, ascending=sort_ascending, kind="mergesort")
elif primary_sort == 'Team Name':
    view = view.sort_values(by=view.index.name, ascending=sort_ascending, kind="mergesort")
elif primary_sort == 'Conf Name':
    view = view.sort_values(by='Conf Name', ascending=sort_ascending, kind="mergesort")

# We don't want to *display* helper columns; keep your original visible ordering
visible_cols = [c for c in view.columns if c != 'Conf Name']
view = view[visible_cols]

# ---------------------------------
# Styling (with gradients) + number formats
# ---------------------------------
numeric_cols = [c for c in view.columns if pd.api.types.is_numeric_dtype(view[c])]

# Base formatting: 1 decimal for most numerics
fmt = {c: '{:.1f}' for c in numeric_cols}
# Whole numbers for these
for col in ['Pre Rk', 'Rk', 'W', 'L']:
    if col in view.columns:
        fmt[col] = '{:.0f}'

# Build base styler
styled = view.style.format(fmt).hide(axis='index')

# Colormaps
dark_navy = '#002060'
dark_green = '#006400'
dark_gold = '#b8860b'

from matplotlib.colors import LinearSegmentedColormap
cmap_blue = LinearSegmentedColormap.from_list('white_to_darknavy', ['#ffffff', dark_navy])
cmap_blue_r = cmap_blue.reversed()
cmap_green = LinearSegmentedColormap.from_list('white_to_darkgreen', ['#ffffff', dark_green])
cmap_gold = LinearSegmentedColormap.from_list('darkgold_to_white', [dark_gold, '#ffffff'])

# Helper to keep text readable on dark backgrounds
def text_contrast(series, invert=False):
    vmin = float(series.min(skipna=True))
    vmax = float(series.max(skipna=True))
    rng = (vmax - vmin) if vmax != vmin else 1.0
    norm = (series - vmin) / rng
    if invert:
        norm = 1 - norm
    return ['color: white' if (x >= 0.6) else 'color: black' for x in norm.fillna(0)]

# Apply gradients (each column on its own scale, using view's min/max)
for col in ['Pwr Rtg', 'Off Rtg']:
    if col in view.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_blue, subset=[col],
                                 vmin=view[col].min(), vmax=view[col].max())
            .apply(text_contrast, subset=[col])
        )

for col in ['Proj W', 'Proj Conf W']:
    if col in view.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_green, subset=[col],
                                 vmin=view[col].min(), vmax=view[col].max())
            .apply(text_contrast, subset=[col])
        )

for col in ['Def Rtg']:
    if col in view.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_blue_r, subset=[col],
                                 vmin=view[col].min(), vmax=view[col].max())
            .apply(lambda s: text_contrast(s, invert=True), subset=[col])
        )

for col in ['Sched Diff']:
    if col in view.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_gold, subset=[col],
                                 vmin=view[col].min(), vmax=view[col].max())
            .apply(lambda s: text_contrast(s, invert=True), subset=[col])
        )

# ---------------------------------
# CSS: header bar color, centered headers, tight mobile layout
# ---------------------------------
st.markdown("""
<style>
.block-container { padding-left: .5rem !important; padding-right: .5rem !important; }
table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 11px; border-collapse: collapse; }
td, th { padding: 4px !important; text-align: center !important; vertical-align: middle !important; font-size: 11px; }

thead th {
  background-color: #002060 !important;
  color: #ffffff !important;
  font-weight: 500 !important;
  font-size: 10px !important;   /* smaller header font */
  padding: 1px 3px !important;  /* tighter header padding */
}

/* Team & Conference columns */
thead th:nth-child(3),
thead th:nth-child(4),
tbody td:nth-child(3),
tbody td:nth-child(4) {
  text-align: center !important;
  vertical-align: middle !important;
  width: 50px;
  min-width: 55px;
  max-width: 75px;
  overflow: hidden;
}

/* center images */
td img { display: block; margin: 0 auto; }

/* tighter spacing for phones */
tbody td { padding-left: 2px !important; padding-right: 2px !important; }
</style>
""", unsafe_allow_html=True)

# Tabs for Rankings and Team Dashboards
tab1, tab2 = st.tabs(["🏆 Rankings", "📊 Team Dashboards"])

# ---------------------------------
# 🏆 Tab 1: Rankings (existing logic goes here)
# ---------------------------------
with tab1:
    st.markdown("## 🏈 College Football Rankings")

    # Modify logo column to make each one clickable (adds deep link via query param)
    df['Team'] = df.apply(
        lambda row: f'<a href="#📊%20Team%20Dashboards" onclick="window.location.search=\'?selected_team={row.name}\'"><img src="{row["Image URL"]}" width="15"></a>'
        if pd.notna(row["Image URL"]) else '',
        axis=1
    )

    # Rebuild and show styled table
    styled = view.style.format(fmt).hide(axis='index')
    # (Add your styling and background gradient logic again here if needed)

    st.write(styled.to_html(escape=False), unsafe_allow_html=True)

# ---------------------------------
# 📊 Tab 2: Team Dashboards (NEW)
# ---------------------------------
with tab2:
    st.markdown("## 📊 Team Dashboards")

    all_teams = df.index.tolist()

    # Handle query param
    from urllib.parse import unquote
    query_params = st.query_params
    preselect_team = unquote(query_params.get("selected_team", ""))

    # Set up default selection from query param or first team
    if 'selected_team' not in st.session_state:
        st.session_state['selected_team'] = preselect_team if preselect_team in all_teams else all_teams[0]

    selected_team = st.selectbox("Select a Team", options=all_teams, index=all_teams.index(st.session_state['selected_team']))
    st.session_state['selected_team'] = selected_team

    team_data = df.loc[[selected_team]]
    st.markdown(f"### Dashboard for {selected_team}")
    st.dataframe(team_data.T, use_container_width=True)

