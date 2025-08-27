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
# Merge logo URL from 'Logos' sheet
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# Drop unnecessary columns
df.drop(columns=['Column1', 'Column3', 'Column5'], inplace=True)

# Format 'Current Rank' as integer
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create 'Logo' column from image URL
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')

# Save 'Team' name before dropping the column
df['Team Name'] = df['Team']

# Reorder columns: Logo appears after ranks
cols = df.columns.tolist()
cols.remove('Team')         # Already saved as 'Team Name'
cols.remove('Image URL')    # No need to display
cols.remove('Logo')         # Will reinsert
ordered_cols = ['Preseason Rank', 'Current Rank', 'Logo'] + cols
df = df[ordered_cols]

# Set team name as index (not shown)
df.set_index('Team Name', inplace=True)

# ---------------------------------
# Numeric columns for gradient styling
# ---------------------------------
# Safe selection of numeric columns that actually exist
float_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# Optional: show in dev
# st.write("Numeric columns detected:", float_cols)

# ---------------------------------
# Styled DataFrame
# ---------------------------------
styled_df = df.style \
    .format({col: '{:.1f}' for col in float_cols if col not in ['Preseason Rank', 'Current Rank']}) \
    .format({'Preseason Rank': '{:.0f}', 'Current Rank': '{:.0f}'}) \
    .background_gradient(subset=float_cols, cmap='Blues') \
    .hide(axis='index')

# ---------------------------------
# CSS: Prevent side scrolling
# ---------------------------------
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

# ---------------------------------
# Page Content
# ---------------------------------
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")

# Render the styled table with HTML (preserves image tags)
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
