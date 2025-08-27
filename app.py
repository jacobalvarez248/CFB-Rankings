import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# 🔥 DROP DUPLICATE COLUMNS
df = df.loc[:, ~df.columns.duplicated(keep='first')]

# Merge logos
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')
df.drop(columns=['Column1', 'Column3', 'Column5'], inplace=True)
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create Logo column
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')
df['Team Name'] = df['Team']

# Reorder columns
columns = df.columns.tolist()
for col in ['Team', 'Image URL', 'Logo']:
    if col in columns:
        columns.remove(col)
ordered = ['Preseason Rank', 'Current Rank', 'Logo'] + columns
df = df[ordered]
df.set_index('Team Name', inplace=True)

# Ensure numeric columns exist and are valid
valid_cols = df.columns.tolist()
numeric_cols = [col for col in valid_cols if pd.api.types.is_numeric_dtype(df[col])]
numeric_cols = [col for col in numeric_cols if col in df.columns]

# Format dictionary
format_dict = {col: '{:.1f}' for col in numeric_cols if col not in ['Preseason Rank', 'Current Rank']}
format_dict.update({'Preseason Rank': '{:.0f}', 'Current Rank': '{:.0f}'})

# Safe styling
styled_df = df.style \
    .format(format_dict) \
    .background_gradient(subset=numeric_cols, cmap='Blues') \
    .hide(axis='index')

# CSS to eliminate scroll
st.markdown("""
    <style>
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 14px; }
    td, th { padding: 6px !important; text-align: center !important; }
    img { display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# Render
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
