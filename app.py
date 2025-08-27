import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

# ---------- Load Excel Data ----------
@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# ---------- Merge and Clean ----------
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# Drop columns not needed
df.drop(columns=['Column1', 'Column3', 'Column5'], inplace=True)

# Format 'Current Rank' as integer
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create 'Logo' column from Image URL
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')

# Reorder columns: Logo goes after 'Preseason Rank' and 'Current Rank'
cols = df.columns.tolist()
cols.remove('Team')
cols.remove('Image URL')
cols.remove('Logo')
ordered_cols = ['Preseason Rank', 'Current Rank', 'Logo'] + cols
df = df[ordered_cols]

# Use 'Team' as index for filtering (not shown in output)
df['Team Name'] = df['Team']
df.set_index('Team Name', inplace=True)

# ---------- Streamlit Styling ----------
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    table {
        width: 100% !important;
        table-layout: fixed;
        word-wrap: break-word;
        font-size: 14px;
    }
    td, th {
        padding: 6px !important;
        text-align: center !important;
    }
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Page Content ----------
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")

# ---------- Styling ----------
# Apply background gradient and number formatting
float_cols = df.select_dtypes(include='number').columns.tolist()

styled_df = df.style \
    .format({col: '{:.1f}' for col in float_cols if col not in ['Preseason Rank', 'Current Rank']}) \
    .format({'Preseason Rank': '{:.0f}', 'Current Rank': '{:.0f}'}) \
    .background_gradient(subset=float_cols, cmap='Blues') \
    .hide(axis='index')

# ---------- Display as HTML with images ----------
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
