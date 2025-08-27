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
# Merge logo URL
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# Drop unwanted columns
df.drop(columns=['Column1', 'Column3', 'Column5'], inplace=True)

# Ensure 'Current Rank' is integer
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create logo HTML
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')

# Save team name before removing
df['Team Name'] = df['Team']

# Reorder columns
cols = df.columns.tolist()
for col_to_remove in ['Team', 'Image URL', 'Logo']:
    if col_to_remove in cols:
        cols.remove(col_to_remove)
ordered_cols = ['Preseason Rank', 'Current Rank', 'Logo'] + cols
df = df[ordered_cols]

# Set index to team name
df.set_index('Team Name', inplace=True)

# ---------------------------------
# Validate columns for styling
# ---------------------------------
# Only style columns that exist
existing_cols = df.columns.tolist()
numeric_cols = [col for col in existing_cols if pd.api.types.is_numeric_dtype(df[col])]

# Defensive filtering: only format columns present
format_dict = {col: '{:.1f}' for col in numeric_cols if col not in ['Preseason Rank', 'Current Rank']}
format_dict.update({'Preseason Rank': '{:.0f}', 'Current Rank': '{:.0f}'})

# ---------------------------------
# Apply styling safely
# ---------------------------------
styled_df = df.style \
    .format(format_dict) \
    .background_gradient(subset=numeric_cols, cmap='Blues') \
    .hide(axis='index')

# ---------------------------------
# CSS for mobile: no side scroll
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
# Display the Table
# ---------------------------------
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")

# Write styled table
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
