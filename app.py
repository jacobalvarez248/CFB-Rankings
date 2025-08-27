import streamlit as st
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(page_title="CFB Rankings", layout="wide")

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
# Styling (with gradients) + number formats
# ---------------------------------
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# Base formatting: 1 decimal for most numerics
fmt = {c: '{:.1f}' for c in numeric_cols}
# Whole numbers for these
for col in ['Pre Rk', 'Rk', 'W', 'L']:
    if col in df.columns:
        fmt[col] = '{:.0f}'

# Build base styler
styled = df.style.format(fmt).hide(axis='index')

# Colormaps
dark_navy = '#002060'
dark_green = '#006400'
dark_red = '#8B0000'

cmap_blue = LinearSegmentedColormap.from_list('white_to_darknavy', ['#ffffff', dark_navy])
cmap_blue_r = cmap_blue.reversed()

cmap_green = LinearSegmentedColormap.from_list('white_to_darkgreen', ['#ffffff', dark_green])
dark_gold = '#b8860b'
cmap_gold = LinearSegmentedColormap.from_list('darkgold_to_white', [dark_gold, '#ffffff'])

# Apply “lower = darker gold”
for col in ['Sched Diff']:
    if col in df.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_gold, subset=[col],
                                 vmin=df[col].min(), vmax=df[col].max())
            .apply(lambda s: text_contrast(s, invert=True), subset=[col])
        )


# Helper to set readable text color on dark backgrounds
def text_contrast(series, invert=False):
    vmin = float(series.min(skipna=True))
    vmax = float(series.max(skipna=True))
    rng = (vmax - vmin) if vmax != vmin else 1.0
    norm = (series - vmin) / rng
    if invert:
        norm = 1 - norm
    return ['color: white' if (x >= 0.6) else 'color: black' for x in norm.fillna(0)]

# Apply “higher = darker navy”
for col in ['Pwr Rtg', 'Off Rtg']:
    if col in df.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_blue, subset=[col],
                                 vmin=df[col].min(), vmax=df[col].max())
            .apply(text_contrast, subset=[col])
        )

# Apply “higher = darker green”
for col in ['Proj W', 'Proj Conf W']:
    if col in df.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_green, subset=[col],
                                 vmin=df[col].min(), vmax=df[col].max())
            .apply(text_contrast, subset=[col])
        )

# Apply “lower = darker navy” (inverse)
for col in ['Def Rtg']:
    if col in df.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_blue_r, subset=[col],
                                 vmin=df[col].min(), vmax=df[col].max())
            .apply(lambda s: text_contrast(s, invert=True), subset=[col])
        )

# Apply “lower = darker red”
for col in ['Sched Diff']:
    if col in df.columns:
        styled = (
            styled
            .background_gradient(cmap=cmap_gold, subset=[col],
                                 vmin=df[col].min(), vmax=df[col].max())
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

st.markdown("## 🏈 College Football Rankings")
st.write(styled.to_html(escape=False), unsafe_allow_html=True)
