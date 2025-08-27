import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# --- De-duplicate columns coming from Excel merges ---
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
# drop any duplicate-suffixed cols if they slipped through
df = df.loc[:, ~df.columns.str.contains(r'\.(1|2|3)$')]

# --- Drop columns for mobile cleanliness ---
df.drop(columns=[
    "Vegas Win Total",
    "Projected Overall Losses",
    "Schedule Difficulty Rank"
], errors='ignore', inplace=True)

# --- Merge team logos ---
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# types/formatting
if 'Current Rank' in df.columns:
    df['Current Rank'] = df['Current Rank'].astype('Int64')

# save Team Name first, then set as index BEFORE subsetting (critical fix)
df['Team Name'] = df['Team']
df.set_index('Team Name', inplace=True)

# build team logo html
df['Team Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')

# --- Conference logos (from same Logos sheet) ---
conf_logo_map = logos_df.set_index('Team')['Image URL'].to_dict()
if 'Conference' in df.columns:
    df['Conference Logo'] = df['Conference'].apply(
        lambda conf: f'<img src="{conf_logo_map.get(conf, "")}" width="40">' if conf_logo_map.get(conf) else (conf if pd.notna(conf) else '')
    )
else:
    # If no Conference column is present, create an empty logo column to keep order stable
    df['Conference Logo'] = ''

# --- Reorder columns: Preseason Rank | Current Rank | Team Logo | Conference Logo | (rest) ---
cols = list(df.columns)  # these are columns AFTER index is set; index is not in this list
for drop in ['Image URL', 'Team', 'Team Logo', 'Conference Logo']:
    if drop in cols:
        cols.remove(drop)
for must_first in ['Preseason Rank', 'Current Rank']:
    if must_first in cols:
        cols.remove(must_first)

ordered = ['Preseason Rank', 'Current Rank', 'Team Logo', 'Conference Logo'] + cols
# Keep only columns that actually exist (in case some are missing in your sheet)
ordered = [c for c in ordered if c in df.columns]
df = df[ordered]

# number formatting (no gradients)
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
fmt = {c: '{:.1f}' for c in numeric_cols}
for rank_col in ['Preseason Rank', 'Current Rank']:
    if rank_col in df.columns:
        fmt[rank_col] = '{:.0f}'

styled = df.style.format(fmt).hide(axis='index')

# --- Mobile CSS: tighter padding + fixed table layout to avoid side-scroll ---
st.markdown("""
<style>
.block-container { padding-left: .5rem !important; padding-right: .5rem !important; }
table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 13px; }
td, th { padding: 4px !important; text-align: center !important; vertical-align: middle !important; }
img { display: block; margin-left: auto; margin-right: auto; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🏈 College Football Rankings (Mobile-Optimized)")
st.write(styled.to_html(escape=False), unsafe_allow_html=True)
