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
# Merge and Clean Data
# ---------------------------------
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# Drop unnecessary columns
df.drop(columns=['Column1', 'Column3', 'Column5'], inplace=True)

# Format 'Current Rank' as integer
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create Logo column from image URL
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')

# Save team name before dropping
df['Team Name'] = df['Team']

# Reorder columns (put Logo after rank columns)
columns = df.columns.tolist()
for col in ['Team', 'Image URL', 'Logo']:
    if col in columns:
        columns.remove(col)
ordered = ['Preseason Rank', 'Current Rank', 'Logo'] + columns
df = df[ordered]

# Set index to Team Name
df.set_index('Team Name', inplace=True)

# ---------------------------------
# Ensure numeric columns are valid
# ---------------------------------
valid_columns = df.columns.tolist()
numeric_cols = [
    col for col in valid_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

# Remove any non-existent columns just in case
numeric_cols = [col for col in numeric_cols if col in df.columns]

# Format dictionary
format_dict = {
    col: '{:.1f}' for col in numeric_cols if col not in ['Preseason Rank', 'Current Rank']
}
format_dict.update({'Preseason Rank': '{:.0f}', 'Current Rank': '{:.0f}'})

# ---------------------------------
# Safe Styled DataFrame
# ---------------------------------
styled_df = df.style \
    .format(format_dict) \
    .background_gradient(subset=numeric_cols, cmap='Blues') \
    .hide(axis='index')

# ---------------------------------
# CSS to prevent side scrolling
# ---------------------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# ---------------------------------
# Display Page
# ---------------------------------
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")

st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
