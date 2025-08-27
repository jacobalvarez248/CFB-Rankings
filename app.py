import streamlit as st
import pandas as pd

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
    "Projected Overall Losses",
    "Projected Conference Losses",
    "Scheudle Difficulty Rank",  # spelling-safe, drops if present
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
    "Projected Overall Wins": "Proj W",
    "Projected Conference Wins": "Proj Conf W",
    "Schedule Difficulty": "Sched Diff"
}, inplace=True)

# --- Reorder cleanly ---
first_cols = ["Pre Rk", "Rk", "Team", "Conf"]
existing = [c for c in df.columns if c not in first_cols]
ordered = [c for c in first_cols if c in df.columns] + existing
df = df[ordered]

# ---------------------------------
# Styling (no gradient) + number formats
# ---------------------------------
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# Base formatting: 1 decimal for most numerics
fmt = {c: '{:.1f}' for c in numeric_cols}

# Override special cases to whole numbers
for col in ['Pre Rk', 'Rk', 'W', 'L']:
    if col in df.columns:
        fmt[col] = '{:.0f}'

styled = df.style.format(fmt).hide(axis='index')


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
  width: 30px;
  min-width: 35px;
  max-width: 45px;
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
