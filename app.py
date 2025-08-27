import streamlit as st
import pandas as pd
import urllib.parse
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

# Preserve a plain team name and set as index
df['Team Name'] = df['Team']
df.set_index('Team Name', inplace=True)

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
    # (If your workbook contains these, you can keep them by removing from this list)
    "Projected Overall Wins",
    "Projected Overall Losses",
    "Projected Conference Wins",
    "Projected Conference Losses",
    "Schedule Difficulty Rank",
    "Column1", "Column3", "Column5"
], errors='ignore', inplace=True)

# --- Rename columns ---
df.rename(columns={
    "Preseason Rank": "Pre Rk",
    "Current Rank": "Rk",
    "Conference Logo": "Conf",
    "Power Rating": "Pwr Rtg",
    "Offensive Rating": "Off Rtg",
    "Defensive Rating": "Def Rtg",
    "Current Wins": "W",
    "Current Losses": "L",
    # If you kept the projections above, these will apply:
    "Projected Overall Wins": "Proj W",
    "Projected Conference Wins": "Proj Conf W",
    "Schedule Difficulty": "Sched Diff"
}, inplace=True)

# --- Make team logos clickable (link passes ?team=<name> and scrolls to #teamdash) ---
def mk_team_logo(url, team_name):
    if pd.notna(url):
        return f'<a href="?tab=team&team={urllib.parse.quote(str(team_name))}"><img src="{url}" width="15"></a>'
    return ''
# Use the index (Team Name) for display name in href
df['Team'] = [mk_team_logo(u, name) for u, name in zip(
    logos_df.set_index('Team').reindex(df.index)['Image URL'].fillna(''),
    df.index
)]

# --- Reorder cleanly ---
first_cols = ["Pre Rk", "Rk", "Team", "Conf"]
existing = [c for c in df.columns if c not in first_cols]
ordered = [c for c in first_cols if c in df.columns] + existing
df = df[ordered]

# ---------------------------------
# Query params
# ---------------------------------
selected_team = None
target_tab = None
try:
    qp = st.query_params
    # team (normalize to str or None)
    selected_team = qp.get("team", [None])
    if isinstance(selected_team, list):
        selected_team = selected_team[0] if selected_team else None
    # tab hint (e.g., 'team')
    target_tab = qp.get("tab", [None])
    if isinstance(target_tab, list):
        target_tab = target_tab[0] if target_tab else None
except Exception:
    qp = st.experimental_get_query_params()
    selected_team = qp.get("team", [None])[0] if qp.get("team") else None
    target_tab = qp.get("tab", [None])[0] if qp.get("tab") else None

# ---------------------------------
# Sidebar controls (filters + sorting + team dashboard picker)
# ---------------------------------
with st.sidebar:
    st.header("Filters & Sort")

    # Team substring search (uses the index 'Team Name')
    team_query = st.text_input("Team contains", value="")

    # Conference multiselect
    conf_options = sorted([c for c in df['Conf Name'].dropna().unique()])
    conf_selected = st.multiselect("Conference", conf_options)

    # Choose sort column (include text helpers for better UX)
    sortable_cols = [c for c in df.columns if c not in ['Team', 'Conf']] + ['Team Name', 'Conf Name']
    default_sort_idx = sortable_cols.index('Rk') if 'Rk' in sortable_cols else 0
    primary_sort = st.selectbox("Sort by", options=sortable_cols, index=default_sort_idx)
    sort_ascending = st.checkbox("Ascending", value=True)

    st.divider()
    st.subheader("Team Dashboards")
    team_list = sorted(df.index.unique().tolist())
    selected_team = st.selectbox(
        "Select team",
        options=team_list,
        index=(team_list.index(selected_team) if selected_team in team_list else 0)
    )

# Start from the working view
view = df.copy()

# Apply Team filter
if team_query:
    view = view[view.index.str.contains(team_query, case=False, na=False)]

# Apply Conference filter
if conf_selected:
    view = view[view['Conf Name'].isin(conf_selected)]

# Sorting: helper columns that aren't displayed (Team Name / Conf Name) still work
if primary_sort in view.columns:
    view = view.sort_values(by=primary_sort, ascending=sort_ascending, kind="mergesort")
elif primary_sort == 'Team Name':
    view = view.sort_values(by=view.index.name, ascending=sort_ascending, kind="mergesort")
elif primary_sort == 'Conf Name':
    view = view.sort_values(by='Conf Name', ascending=sort_ascending, kind="mergesort")

# We don't want to display helper columns
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
dark_gold  = '#b8860b'

cmap_blue   = LinearSegmentedColormap.from_list('white_to_darknavy', ['#ffffff', dark_navy])
cmap_blue_r = cmap_blue.reversed()
cmap_green  = LinearSegmentedColormap.from_list('white_to_darkgreen', ['#ffffff', dark_green])
cmap_gold   = LinearSegmentedColormap.from_list('darkgold_to_white', [dark_gold, '#ffffff'])

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

# ---------------------------------
# Tabs: Rankings | Team Dashboards
# ---------------------------------
tab_rankings, tab_team = st.tabs(["🏈 Rankings", "🧭 Team Dashboards"])

with tab_rankings:
    # ====== RANKINGS VIEW ======
    st.markdown("## 🏈 College Football Rankings")
    # ---- build 'view' exactly as you already do (filters/sort applied) ----
    # (your existing code already produced 'view' and 'styled')
    st.write(styled.to_html(escape=False), unsafe_allow_html=True)

with tab_team:
    # ====== TEAM DASHBOARD VIEW ======
    # Top-of-page team selector (no sidebar here)
    team_list = sorted(df.index.unique().tolist())
    # if query param had a team, default to that; otherwise first in list
    default_team = selected_team if selected_team in team_list else team_list[0]
    picked_team = st.selectbox("Select team", options=team_list, index=team_list.index(default_team))
    
    # Single-team slice (use full df so columns match main table)
    team_df = df.loc[[picked_team]]
    team_view = team_df[[c for c in df.columns if c != 'Conf Name']].copy()

    # Formatting: same rules
    num_cols_team = [c for c in team_view.columns if pd.api.types.is_numeric_dtype(team_view[c])]
    fmt_team = {c: '{:.1f}' for c in num_cols_team}
    for col in ['Pre Rk', 'Rk', 'W', 'L']:
        if col in team_view.columns:
            fmt_team[col] = '{:.0f}'

    team_styled = team_view.style.format(fmt_team).hide(axis='index')

    # Reuse colormaps and text_contrast you defined earlier
    for col in ['Pwr Rtg', 'Off Rtg']:
        if col in team_view.columns:
            team_styled = (team_styled
                .background_gradient(cmap=cmap_blue, subset=[col],
                                     vmin=team_view[col].min(), vmax=team_view[col].max())
                .apply(text_contrast, subset=[col])
            )
    for col in ['Proj W', 'Proj Conf W']:
        if col in team_view.columns:
            team_styled = (team_styled
                .background_gradient(cmap=cmap_green, subset=[col],
                                     vmin=team_view[col].min(), vmax=team_view[col].max())
                .apply(text_contrast, subset=[col])
            )
    for col in ['Def Rtg']:
        if col in team_view.columns:
            team_styled = (team_styled
                .background_gradient(cmap=cmap_blue_r, subset=[col],
                                     vmin=team_view[col].min(), vmax=team_view[col].max())
                .apply(lambda s: text_contrast(s, invert=True), subset=[col])
            )
    for col in ['Sched Diff']:
        if col in team_view.columns:
            team_styled = (team_styled
                .background_gradient(cmap=cmap_gold, subset=[col],
                                     vmin=team_view[col].min(), vmax=team_view[col].max())
                .apply(lambda s: text_contrast(s, invert=True), subset=[col])
            )

    st.write(team_styled.to_html(escape=False), unsafe_allow_html=True)
