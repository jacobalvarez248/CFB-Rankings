import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# --- De-duplicate columns ---
def deduplicate_columns(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}.{seen[col]}")
    return new_columns

df.columns = deduplicate_columns(df.columns)
df = df.loc[:, ~df.columns.str.contains(r'\.1$|\.2$|\.3$')]

# --- Drop columns you want removed ---
columns_to_remove = [
    "Vegas Win Total",
    "Projected Overall Losses",
    "Schedule Difficulty Rank"
]
df.drop(columns=columns_to_remove, errors='ignore', inplace=True)

df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')
df['Current Rank'] = df['Current Rank'].astype('Int64')

# ✅ Save Team Name BEFORE anything else touches 'Team'
df['Team Name'] = df['Team']

# --- Add logo columns ---
df['Team Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')

# Get conference logos using same "Logos" tab
conf_logos = logos_df.set_index('Team')['Image URL'].to_dict()
df['Conference Logo'] = df['Conference'].apply(lambda conf: f'<img src="{conf_logos.get(conf, "")}" width="40">' if conf in conf_logos else conf)

# Reorder columns
cols = df.columns.tolist()
for col in ['Team', 'Image URL', 'Team Logo', 'Conference Logo', 'Team Name']:
    if col in cols:
        cols.remove(col)

# Ensure no duplicates in manual list
for col in ['Preseason Rank', 'Current Rank', 'Conference']:
    if col in cols:
        cols.remove(col)

ordered = ['Preseason Rank', 'Current Rank', 'Team Logo', 'Conference Logo'] + cols
df = df[ordered]
df.set_index('Team Name', inplace=True)

# Format numeric columns
numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
format_dict = {col: '{:.1f}' for col in numeric_cols}
for col in ['Preseason Rank', 'Current Rank']:
    if col in df.columns:
        format_dict[col] = '{:.0f}'

# Style DataFrame
styled_df = df.style.format(format_dict).hide(axis='index')

# --- CSS to prevent scroll & optimize mobile ---
st.markdown("""
    <style>
    .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 13px; }
    td, th { padding: 4px !important; text-align: center !important; }
    img { display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# --- Render page ---
st.markdown("## 🏈 College Football Rankings (Mobile-Optimized)")
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
