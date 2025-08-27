import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFB Rankings", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
    logos_df = pd.read_excel('CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)
    return df, logos_df

df, logos_df = load_data()

# ✅ Deduplicate column names
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

# ✅ Drop any known bad duplicates manually (keep clean set only)
columns_to_drop = [col for col in df.columns if col.endswith('.1') or col.endswith('.2')]
df.drop(columns=columns_to_drop, inplace=True)

# Merge logo image
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left', how='left')
df.drop(columns=['Column1', 'Column3', 'Column5'], errors='ignore', inplace=True)
df['Current Rank'] = df['Current Rank'].astype('Int64')

# Create logo column
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')
df['Team Name'] = df['Team']

# Reorder columns (Logo after ranks)
columns = df.columns.tolist()
for col in ['Team', 'Image URL', 'Logo']:
    if col in columns:
        columns.remove(col)
ordered = ['Preseason Rank', 'Current Rank', 'Logo'] + columns
df = df[ordered]
df.set_index('Team Name', inplace=True)

# ✅ Only use valid numeric columns
valid_cols = df.columns.tolist()
numeric_cols = [col for col in valid_cols if pd.api.types.is_numeric_dtype(df[col])]

# Format dictionary
format_dict = {col: '{:.1f}' for col in numeric_cols}
for col in ['Preseason Rank', 'Current Rank']:
    if col in df.columns:
        format_dict[col] = '{:.0f}'

# ✅ Final safe styling block
styled_df = df.style.format(format_dict)

if numeric_cols:
    styled_df = styled_df.background_gradient(subset=numeric_cols, cmap='Blues')

styled_df = styled_df.hide(axis='index')

# Mobile layout
st.markdown("""
    <style>
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    table { width: 100% !important; table-layout: fixed; word-wrap: break-word; font-size: 14px; }
    td, th { padding: 6px !important; text-align: center !important; }
    img { display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# ✅ Final display
st.markdown("## 🏈 College Football Rankings")
st.markdown("Click a logo to view that team’s dashboard (coming soon).")
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
