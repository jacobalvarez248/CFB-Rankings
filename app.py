import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

# Load data
df, logos_df = load_data()

# 🔁 Force all column names to be unique
df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

# Merge logos
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')
df.drop(columns=['Column1', 'Column3', 'Column5'], errors='ignore', inplace=True)

# Format rank
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Logo image
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

# 💡 Recalculate valid numeric columns
valid_cols = df.columns.tolist()
numeric_cols = [col for col in valid_cols if pd.api.types.is_numeric_dtype(df[col])]

# 🧠 Debug info for you
st.write("✅ Final DataFrame Columns:", valid_cols)
st.write("✅ Columns used for gradient styling:", numeric_cols)

# Format dictionary
format_dict = {col: '{:.1f}' for col in numeric_cols if 'Rank' not in col}
for col in ['Preseason Rank', 'Current Rank']:
    if col in df.columns:
        format_dict[col] = '{:.0f}'

# 🔐 Final safe styling
styled_df = df.style \
    .format(format_dict) \
    .background_gradient(subset=numeric_cols, cmap='Blues') \
    .hide(axis='index')

# CSS
st.markdown("""
    <style>
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 14px; }
    td, th { padding: 6px !important; text-align: center !important; }
    img { display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# Display
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")

# 🔐 Write styled table as HTML
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
