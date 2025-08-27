import streamlit as st
import pandas as pd

# Load data
df = pd.read_excel('data/CFB Rankings Upload.xlsm', sheet_name='Expected Wins', header=1)
logos_df = pd.read_excel('data/CFB Rankings Upload.xlsm', sheet_name='Logos', header=1)

# Merge logos into main DataFrame
df = df.merge(logos_df[['Team', 'Image URL']], on='Team', how='left')

# Drop unwanted columns
df.drop(columns=['Column1', 'Column3', 'Column5'], inplace=True)

# Add team name as index (for filtering), but display logo instead
df['Logo'] = df['Image URL'].apply(lambda url: f'<img src="{url}" width="40">' if pd.notna(url) else '')
df.set_index('Team', inplace=True)

# Reorder: Logo first
columns = ['Logo'] + [col for col in df.columns if col not in ['Logo', 'Image URL']]
df = df[columns]

# Tabs
tab1, tab2 = st.tabs(["🏆 Rankings", "📊 Team Dashboards"])

# Session state to control navigation between tabs
if "selected_team" not in st.session_state:
    st.session_state.selected_team = None

# --- Rankings Tab ---
with tab1:
    st.markdown("### 📈 Rankings Overview")

    # Make logos clickable
    def make_clickable_logo(row):
        team = row.name
        return f'<a href="#team={team}">{row["Logo"]}</a>'

    df['Logo'] = df.apply(make_clickable_logo, axis=1)

    # Prepare styled dataframe (numeric columns with gradient)
    styled_df = df.drop(columns=['Image URL']).style \
        .format(precision=1) \
        .hide(axis="index") \
        .applymap(lambda v: 'background-color: #c6e2ff' if isinstance(v, (int, float)) else '', subset=df.select_dtypes(include='number').columns)

    st.markdown("Click a logo to view team dashboard 👇", unsafe_allow_html=True)
    st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

# --- Team Dashboards Tab ---
with tab2:
    st.markdown("### 📊 Team Dashboards")

    all_teams = sorted(df.index.unique())
    default_team = st.session_state.selected_team or all_teams[0]

    team_select = st.selectbox("Select a Team:", all_teams, index=all_teams.index(default_team))
    st.session_state.selected_team = team_select

    st.markdown(f"Showing data for **{team_select}**...")
    # Placeholder for actual team-level content
    st.info("Team dashboard content will be added here soon.")
