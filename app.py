import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# Deduplicate columns
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

# Drop bad duplicates like 'Current Rank.1'
df = df.loc[:, ~df.columns.str.contains(r'\.1$|\.2$|\.3$')]

# Merge logos
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')
df.drop(columns=['Column1', 'Column3', 'Column5'], errors='ignore', inplace=True)
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create logo image column
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')
df['Team Name'] = df['Team']

# Reorder columns
cols = df.columns.tolist()
for col in ['Team', 'Image URL', 'Logo']:
    if col in cols:
        cols.remove(col)
ordered = ['Preseason Rank', 'Current Rank', 'Logo'] + cols
df = df[ordered]
df.set_index('Team Name', inplace=True)

# Format numeric columns
numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

format_dict = {col: '{:.1f}' for col in numeric_cols}
for col in ['Preseason Rank', 'Current Rank']:
    if col in df.columns:
        format_dict[col] = '{:.0f}'

# Apply styling (no gradient)
styled_df = df.style.format(format_dict).hide(axis='index')

# CSS for mobile layout
st.markdown("""
    <style>
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 14px; }
    td, th { padding: 6px !important; text-align: center !important; }
    img { display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# Page content
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
