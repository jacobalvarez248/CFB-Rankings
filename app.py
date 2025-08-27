import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from urllib.parse import quote_plus


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
# Query params (for cross-tab navigation)
# ---------------------------------
try:
    qp = st.query_params  # Streamlit >=1.30
    requested_tab = qp.get("tab", "Rankings")
    requested_team = qp.get("team", None)
except Exception:
    requested_tab = "Rankings"
    requested_team = None

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

# Build team logo HTML (clickable to jump to Team Dashboards with team preselected)
def team_logo_html(url, team_name):
    if pd.isna(url):
        return ''
    q_team = quote_plus(str(team_name))
    return f'<a href="?tab=Team%20Dashboards&team={q_team}" title="Open Team Dashboard"><img src="{url}" width="15"></a>'

df['Team Logo'] = [
    team_logo_html(u, idx)
    for u, idx in zip(df['Image URL'], df['Team Name'])
]

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

# ---------------------------------
# Tabs UI
# ---------------------------------
tabs = st.tabs(["Rankings", "Team Dashboards"])

# Figure out which tab to show first via lightweight JS click (optional nicety)
# (Streamlit doesn't natively preselect a tab; this will auto-click the header if query param requests the dashboard)
if requested_tab == "Team Dashboards":
    st.markdown("""
    <script>
    const want = "Team Dashboards";
    const tryClick = () => {
      const els = window.parent.document.querySelectorAll('button[role="tab"]');
      for (const el of els) {
        if (el.innerText.trim() === want) { el.click(); break; }
      }
    };
    window.addEventListener('load', () => setTimeout(tryClick, 50));
    </script>
    """, unsafe_allow_html=True)

# --- Rankings tab ---
with tabs[0]:
    st.markdown("## 🏈 College Football Rankings")
    st.write(styled.to_html(escape=False), unsafe_allow_html=True)

# --- Team Dashboards tab ---
with tabs[1]:
    st.markdown("## 📊 Team Dashboards")

    teams = list(df.index.unique())
    # Default to the requested team (from query param) if valid; else first team alphabetically
    default_team = requested_team if (requested_team in teams) else (requested_team if requested_team in (t.lower() for t in teams) else None)
    if default_team is None:
        default_team = sorted(teams)[0]

    # Build the selector (kept simple for now)
    selected_team = st.selectbox("Team", sorted(teams), index=sorted(teams).index(default_team) if default_team in teams else 0, key="team_dashboard_select")

    # --- Example content: quick KPIs + the row slice ---
    row = df.loc[selected_team]

    # Show a few top-line metrics if present
    kpi_cols = [c for c in ["Rk", "Pre Rk", "Pwr Rtg", "Off Rtg", "Def Rtg", "W", "L", "Proj W", "Proj Conf W", "Sched Diff"] if c in df.columns]
    cols = st.columns(min(4, len(kpi_cols)) or 1)
    for i, c in enumerate(kpi_cols[:4]):
        with cols[i]:
            val = row[c] if pd.notna(row[c]) else "-"
            st.metric(c, f"{val:.1f}" if isinstance(val, (int, float)) and c not in ["Rk","Pre Rk","W","L"] else f"{val}")

    # Full team details table
    st.markdown("#### Team Detail")
    # Rebuild a neat 1-row frame for display (hide the logo HTML)
    show_cols = [c for c in df.columns if c not in ["Team"]]  # keep numeric/text, omit logo cell
    detail = pd.DataFrame([row[show_cols]]).T
    detail.columns = [selected_team]
    st.dataframe(detail, use_container_width=True)

