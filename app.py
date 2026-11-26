import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Custom Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Toronto Blue Jays | 2025 Postseason Pitching Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #F8F9FB;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(19, 74, 142, 0.08);
        border-top: 4px solid #134A8E;
        text-align: center;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #556987;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #134A8E;
        margin: 6px 0;
    }
    .metric-subtext {
        font-size: 0.82rem;
        color: #8896AB;
    }
    .card-accent-red { border-top: 4px solid #E8291C; }
    .card-accent-cyan { border-top: 4px solid #53ACDC; }
    .card-accent-navy { border-top: 4px solid #1D2D5C; }

    .header-banner {
        background: linear-gradient(135deg, #134A8E 0%, #1D2D5C 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(19, 74, 142, 0.25);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
    }
    .header-subtitle {
        font-size: 1.0rem;
        color: #E0E7FF;
        margin-top: 6px;
    }
    .header-badge {
        display: inline-block;
        background-color: #E8291C;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .insight-box {
        background-color: #EFF6FF;
        border-left: 5px solid #134A8E;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #1E293B;
    }
    .recommendation-box {
        background-color: #FEF2F2;
        border-left: 5px solid #E8291C;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Preprocessing (Matches Team Notebooks)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    csv_file = 'Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv'
    if not os.path.exists(csv_file):
        st.error(f"Dataset file '{csv_file}' was not found in the root directory.")
        st.stop()

    df = pd.read_csv(csv_file)

    # 1. Determine pitching team (Home pitches Top, Away pitches Bot)
    df['pitching_team'] = df['home_team'].where(df['inning_topbot'].eq('Top'), df['away_team'])

    # 2. Filter for Toronto Blue Jays (TOR) 2025 Postseason & drop all-null columns
    tor_df = df[(df['pitching_team'] == 'TOR') & (df['game_year'] == 2025)].copy()
    empty_cols = tor_df.columns[tor_df.isna().all()]
    tor_df = tor_df.drop(columns=empty_cols)
    tor_df = tor_df.dropna(subset=['pitcher', 'player_name', 'pitch_type']).copy()

    # 3. Count & Game State Definitions (Matches Q1)
    tor_df['count'] = tor_df['balls'].astype(str) + '-' + tor_df['strikes'].astype(str)
    tor_df['is_runner_on'] = (
        tor_df['on_1b'].notna() | tor_df['on_2b'].notna() | tor_df['on_3b'].notna()
    )
    tor_df['runner_state'] = np.where(tor_df['is_runner_on'], 'Runners On', 'Bases Empty')

    def categorize_count(c):
        if c in ['0-1', '0-2', '1-2']:
            return 'Ahead (0-1, 0-2, 1-2)'
        elif c in ['2-0', '2-1', '3-0', '3-1']:
            return 'Behind (2-0, 2-1, 3-0, 3-1)'
        else:
            return 'Even / Full (0-0, 1-0, 1-1, 2-2, 3-2)'

    tor_df['count_state'] = tor_df['count'].apply(categorize_count)

    # 4. Swing and Whiff Definitions (Matches Q3)
    whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
    swing_descriptions = whiff_descriptions + [
        'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip', 'hit_into_play'
    ]
    tor_df['is_swing'] = tor_df['description'].isin(swing_descriptions)
    tor_df['is_whiff'] = tor_df['description'].isin(whiff_descriptions)

    # 5. Previous Pitch within Same At-Bat (Matches Q2 Transition Matrix)
    tor_df = tor_df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)
    tor_df['prev_pitch_type'] = tor_df.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)

    return tor_df

tor_raw = load_and_preprocess_data()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/b/ba/Toronto_Blue_Jays_logo.svg/300px-Toronto_Blue_Jays_logo.svg.png",
        width=110
    )
    st.markdown("### **Postseason Filter Console**")
    st.caption("Filter the dataset by pitcher, batter handedness, count state, and base runners.")

    all_pitchers = sorted(list(tor_raw['player_name'].dropna().unique()))
    selected_pitchers = st.multiselect("👤 Pitcher(s)", all_pitchers, default=[])

    stand_filter = st.radio(
        "🏏 Batter Handedness",
        ['All (L & R)', 'Right-Handed (R)', 'Left-Handed (L)'],
        horizontal=True
    )

    count_filter = st.selectbox(
        "⚾ Count State",
        [
            'All Counts',
            'Ahead (0-1, 0-2, 1-2)',
            'Behind (2-0, 2-1, 3-0, 3-1)',
            'Even / Full (0-0, 1-0, 1-1, 2-2, 3-2)'
        ],
        index=0
    )

    runners_filter = st.radio(
        "🏃 Base Runners",
        ['All States', 'Bases Empty', 'Runners On'],
        horizontal=True
    )

    st.markdown("---")
    st.markdown("### **Project Info**")
    st.markdown("""
    **Course:** MATH 1130  
    **Team:** Data Science Group A  
    **Dataset:** MLB 2025 Postseason Statcast (TOR: 2,788 pitches)  
    """)

# ---------------------------------------------------------
# Apply Global Filters
# ---------------------------------------------------------
tor = tor_raw.copy()

if selected_pitchers:
    tor = tor[tor['player_name'].isin(selected_pitchers)]

if stand_filter == 'Right-Handed (R)':
    tor = tor[tor['stand'] == 'R']
elif stand_filter == 'Left-Handed (L)':
    tor = tor[tor['stand'] == 'L']

if count_filter != 'All Counts':
    tor = tor[tor['count_state'] == count_filter]

if runners_filter == 'Bases Empty':
    tor = tor[~tor['is_runner_on']]
elif runners_filter == 'Runners On':
    tor = tor[tor['is_runner_on']]

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <span class="header-badge">MATH 1130 | Group A</span>
    <h1 class="header-title">Toronto Blue Jays 2025 Postseason Pitching Analytics</h1>
    <p class="header-subtitle">Analyzing pitch predictability across counts and sequences, whether predictability hurts outcomes, and which pitch traits and locations drive whiffs.</p>
</div>
""", unsafe_allow_html=True)

if len(tor) == 0:
    st.warning("No pitches match the current sidebar filter combination. Please adjust your filters.")
    st.stop()

# ---------------------------------------------------------
# Executive KPI Cards (4 Core Metrics from Team Analysis)
# ---------------------------------------------------------
total_pitches = len(tor)
total_swings = int(tor['is_swing'].sum())
total_whiffs = int(tor['is_whiff'].sum())
whiff_rate = (total_whiffs / total_swings * 100) if total_swings > 0 else 0.0

balls_in_play = tor[tor['description'].eq('hit_into_play')]
measured_bip = int(balls_in_play['launch_speed'].count())
avg_ev = balls_in_play['launch_speed'].mean() if measured_bip > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pitches Analyzed</div>
        <div class="metric-value">{total_pitches:,}</div>
        <div class="metric-subtext">{tor['player_name'].nunique()} Blue Jays Pitchers</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card card-accent-red">
        <div class="metric-label">Whiff Rate</div>
        <div class="metric-value">{whiff_rate:.1f}%</div>
        <div class="metric-subtext">{total_whiffs:,} Misses / {total_swings:,} Swings</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card card-accent-cyan">
        <div class="metric-label">Avg Exit Velocity</div>
        <div class="metric-value">{avg_ev:.1f} <span style="font-size:1.1rem;font-weight:600;">mph</span></div>
        <div class="metric-subtext">{measured_bip:,} Measured Balls in Play</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card card-accent-navy">
        <div class="metric-label">Avg Release Speed</div>
        <div class="metric-value">{tor['release_speed'].mean():.1f} <span style="font-size:1.1rem;font-weight:600;">mph</span></div>
        <div class="metric-subtext">Max: {tor['release_speed'].max():.1f} mph</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4 Focused Tabs Matching the Team's Actual Analysis
# ---------------------------------------------------------
tab_overview, tab_pred, tab_outcomes, tab_traits = st.tabs([
    "📊 Overview & Arsenal",
    "🎯 Predictability & Sequencing (Q1 & Q2)",
    "📈 Predictability vs. Outcomes (Q3)",
    "⚡ Pitch Traits, Location & Scouting"
])

# =========================================================
# TAB 1: OVERVIEW & ARSENAL
# =========================================================
with tab_overview:
    st.subheader("Research Question & Key Findings")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown("""
        <div class="insight-box">
            <b>Core Research Question:</b><br>
            How predictable were Toronto Blue Jays pitchers in the 2025 postseason, does overall predictability 
            hurt whiff rates or contact quality, and what pitch-level traits and locations actually drive swings and misses?
            <br><br>
            <b>Data Summary:</b> 2,788 pitches thrown by 15 Toronto Blue Jays pitchers in the 2025 MLB Postseason 
            (0 duplicate pitch IDs; 3 balls in play missing exit velocity).
        </div>
        """, unsafe_allow_html=True)

    with col_o2:
        st.markdown("""
        <div class="recommendation-box">
            <b>Main Conclusions:</b><br>
            • <b>High Situational & Sequential Predictability:</b> Pitchers heavily favor fastballs when behind in the count, 
            splitters/sliders when ahead, and repeat the exact same pitch type over 40–52% of the time.<br>
            • <b>Aggregate Predictability ≠ Worse Outcomes:</b> Once 4 low-volume relievers (≤100 pitches) are removed, 
            overall predictability shows near-zero correlation with whiff rate (<b>r = -0.15</b>) and exit velocity (<b>r = -0.19</b>).<br>
            • <b>Execution Drives Whiffs:</b> Elevating high-velocity fastballs (3.0–4.0 ft) and burying offspeed pitches 
            below the zone (&lt;1.5 ft) generate the most swings and misses.
        </div>
        """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        pitch_usage = tor['pitch_name'].value_counts().reset_index()
        pitch_usage.columns = ['Pitch Name', 'Count']
        fig_donut = px.pie(
            pitch_usage,
            names='Pitch Name',
            values='Count',
            hole=0.45,
            title='<b>Blue Jays Pitch Usage Breakdown</b>',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_donut.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_a2:
        # Team Whiff% by Pitch Type (Matches Q3 A6 Bar Chart)
        whiff_by_type = tor.groupby('pitch_type').agg(
            pitches=('pitch_type', 'count'),
            swings=('is_swing', 'sum'),
            whiffs=('is_whiff', 'sum')
        ).reset_index()
        whiff_by_type = whiff_by_type[whiff_by_type['swings'] > 0].copy()
        whiff_by_type['whiff_pct'] = (whiff_by_type['whiffs'] / whiff_by_type['swings'] * 100).round(1)
        whiff_by_type = whiff_by_type.sort_values('whiff_pct', ascending=False)
        whiff_by_type['Role'] = np.where(
            whiff_by_type['pitch_type'].isin(['FS', 'SL']),
            'Primary Put-Away (FS / SL)',
            'Other Pitch Types'
        )

        fig_team_whiff = px.bar(
            whiff_by_type,
            x='pitch_type',
            y='whiff_pct',
            text='whiff_pct',
            color='Role',
            color_discrete_map={
                'Primary Put-Away (FS / SL)': '#E8291C',
                'Other Pitch Types': '#134A8E'
            },
            hover_data=['pitches', 'swings', 'whiffs'],
            title='<b>Team Whiff Rate (%) by Pitch Type</b>',
            labels={'pitch_type': 'Pitch Type', 'whiff_pct': 'Whiff Rate (%)'}
        )
        fig_team_whiff.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_team_whiff.update_layout(yaxis_range=[0, max(60, whiff_by_type['whiff_pct'].max() + 10)], plot_bgcolor='white')
        st.plotly_chart(fig_team_whiff, use_container_width=True)

# =========================================================
# TAB 2: PREDICTABILITY & SEQUENCING (Q1 & Q2)
# =========================================================
with tab_pred:
    st.subheader("Q1: Pitch Selection by Count, Batter Handedness & Base Runners")

    exclude_30 = st.checkbox(
        "Exclude small-sample 3-0 count (matches Q1 adjusted heatmaps: n=18 LHB, n=14 RHB)",
        value=True
    )

    clean_tor = tor.dropna(subset=['count', 'stand', 'pitch_type']).copy()
    if exclude_30:
        clean_tor = clean_tor[clean_tor['count'] != '3-0']

    prob_crosstab = pd.crosstab(
        [clean_tor['count'], clean_tor['stand']],
        clean_tor['pitch_type'],
        normalize='index'
    ) * 100

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("##### vs. Left-Handed Batters (LHB)")
        if 'L' in prob_crosstab.index.get_level_values('stand'):
            df_left = prob_crosstab.xs('L', level='stand')
            fig_hm_left = px.imshow(
                df_left.round(1),
                labels=dict(x="Pitch Type", y="Count", color="Prob (%)"),
                x=df_left.columns,
                y=df_left.index,
                color_continuous_scale="YlGnBu",
                zmin=0, zmax=60,
                text_auto='.1f',
                aspect="auto",
                title="<b>P(Pitch | Count, LHB) %</b>"
            )
            fig_hm_left.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_hm_left, use_container_width=True)
        else:
            st.info("No Left-Handed batter data for current filters.")

    with col_p2:
        st.markdown("##### vs. Right-Handed Batters (RHB)")
        if 'R' in prob_crosstab.index.get_level_values('stand'):
            df_right = prob_crosstab.xs('R', level='stand')
            fig_hm_right = px.imshow(
                df_right.round(1),
                labels=dict(x="Pitch Type", y="Count", color="Prob (%)"),
                x=df_right.columns,
                y=df_right.index,
                color_continuous_scale="YlGnBu",
                zmin=0, zmax=60,
                text_auto='.1f',
                aspect="auto",
                title="<b>P(Pitch | Count, RHB) %</b>"
            )
            st.plotly_chart(fig_hm_right, use_container_width=True)
        else:
            st.info("No Right-Handed batter data for current filters.")

    st.markdown("""
    <div class="insight-box">
        <b>Q1 Takeaway:</b> When <b>behind in the count</b> (2-0, 2-1, 3-1), Blue Jays pitchers heavily spike their 
        Four-Seam Fastball (FF) usage to throw a strike. When <b>ahead</b> (0-1, 0-2, 1-2), they pivot sharply to 
        offspeed chase pitches—<b>Splitters (FS)</b> against LHB and <b>Sliders (SL)</b> against RHB.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_q2_1, col_q2_2 = st.columns([3, 2])

    with col_q2_1:
        st.subheader("Q2: Pitch Sequencing — Transition Matrix P(Next | Previous)")
        seq_df = tor.dropna(subset=['prev_pitch_type', 'pitch_type'])
        if len(seq_df) > 0:
            trans_matrix = pd.crosstab(
                seq_df['prev_pitch_type'],
                seq_df['pitch_type'],
                normalize='index'
            ) * 100
            fig_trans = px.imshow(
                trans_matrix.round(1),
                labels=dict(x="Current Pitch Type", y="Previous Pitch Type", color="Prob (%)"),
                x=trans_matrix.columns,
                y=trans_matrix.index,
                color_continuous_scale="Blues",
                text_auto='.1f',
                aspect="auto",
                title="<b>Pitch-to-Pitch Transition Probabilities (%) — Diagonal = Repeat Pitch</b>"
            )
            st.plotly_chart(fig_trans, use_container_width=True)
            st.caption("If pitch selection were random across 9 pitch types, each cell would be ~11.1%. High diagonal values (40–52%) show strong back-to-back repeat tendencies.")
        else:
            st.info("Not enough sequential pitches for the current filter.")

    with col_q2_2:
        st.subheader("Runners on Base Impact")
        runner_crosstab = pd.crosstab(
            tor['runner_state'],
            tor['pitch_type'],
            normalize='index'
        ) * 100
        fig_runner = px.bar(
            runner_crosstab.reset_index().melt(id_vars='runner_state', var_name='Pitch Type', value_name='Percentage'),
            x='runner_state',
            y='Percentage',
            color='Pitch Type',
            barmode='stack',
            title='<b>Pitch Mix: Bases Empty vs. Runners On</b>',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_runner.update_layout(yaxis_title="Percentage (%)", xaxis_title="", plot_bgcolor='white')
        st.plotly_chart(fig_runner, use_container_width=True)

# =========================================================
# TAB 3: PREDICTABILITY VS OUTCOMES (Q3 - JONATHAN'S WORK)
# =========================================================
with tab_outcomes:
    st.subheader("Q3: Does Overall Predictability Hurt Whiff Rate or Contact Quality?")

    sample_mode = st.radio(
        "Pitcher Sample Scope (Sensitivity Check):",
        [
            "All 15 Pitchers (Unfiltered)",
            "Qualified 11 Pitchers (>100 Pitches — Removes Little, Rodríguez, Nance, Bruihl)"
        ],
        horizontal=True
    )

    # Pitcher-level metrics (computed on full TOR 2025 dataset to match Q3 tables exactly)
    group_cols = ['pitcher', 'player_name', 'p_throws']
    pitch_counts = pd.crosstab([tor_raw[c] for c in group_cols], tor_raw['pitch_type'])
    most_common_count = pitch_counts.max(axis=1)

    pred_summary = pd.DataFrame({
        'total_pitches': pitch_counts.sum(axis=1),
        'most_common_pitch_type': pitch_counts.apply(lambda r: ', '.join(r.index[r.eq(r.max())]), axis=1),
        'predictability_pct': most_common_count / pitch_counts.sum(axis=1) * 100
    }).reset_index()

    swings_agg = tor_raw.groupby(group_cols).agg(
        total_swings=('is_swing', 'sum'),
        missed_swings=('is_whiff', 'sum')
    ).reset_index()
    swings_agg['whiff_rate_pct'] = (
        swings_agg['missed_swings']
        / swings_agg['total_swings'].where(swings_agg['total_swings'].gt(0)) * 100
    )

    bip_raw = tor_raw[tor_raw['description'].eq('hit_into_play')]
    contact_agg = bip_raw.groupby(group_cols).agg(
        balls_in_play=('description', 'size'),
        measured_bip=('launch_speed', 'count'),
        avg_exit_velo=('launch_speed', 'mean')
    ).reset_index()

    full_q3 = pred_summary.merge(swings_agg, on=group_cols, how='left').merge(contact_agg, on=group_cols, how='left')
    full_q3['balls_in_play'] = full_q3['balls_in_play'].fillna(0).astype(int)
    full_q3['measured_bip'] = full_q3['measured_bip'].fillna(0).astype(int)

    if sample_mode.startswith("Qualified"):
        q3_df = full_q3[full_q3['total_pitches'] > 100].copy()
    else:
        q3_df = full_q3.copy()

    # Graph 1: Predictability by Pitcher
    bar_df = q3_df.sort_values('predictability_pct', ascending=True).copy()
    bar_df['label'] = (
        bar_df['player_name'] + ' (' + bar_df['most_common_pitch_type']
        + '; n=' + bar_df['total_pitches'].astype(str) + ')'
    )
    fig_pred_bar = px.bar(
        bar_df,
        x='predictability_pct',
        y='label',
        orientation='h',
        text='predictability_pct',
        title='<b>Overall Pitch Predictability by Pitcher (% Most Common Pitch Type)</b>',
        labels={'predictability_pct': 'Predictability (%)', 'label': ''},
        color_discrete_sequence=['#134A8E']
    )
    fig_pred_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_pred_bar.update_layout(xaxis_range=[0, 70], plot_bgcolor='white', height=430)
    st.plotly_chart(fig_pred_bar, use_container_width=True)

    # Graph 2 & Graph 3: Scatterplots with Pearson r
    valid_w = q3_df.dropna(subset=['predictability_pct', 'whiff_rate_pct'])
    r_whiff = valid_w['predictability_pct'].corr(valid_w['whiff_rate_pct'])
    slope_w, intercept_w = np.polyfit(valid_w['predictability_pct'], valid_w['whiff_rate_pct'], 1)

    valid_ev = q3_df.dropna(subset=['predictability_pct', 'avg_exit_velo'])
    r_ev = valid_ev['predictability_pct'].corr(valid_ev['avg_exit_velo'])
    slope_ev, intercept_ev = np.polyfit(valid_ev['predictability_pct'], valid_ev['avg_exit_velo'], 1)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_r1 = px.scatter(
            valid_w,
            x='predictability_pct',
            y='whiff_rate_pct',
            text=valid_w['player_name'].apply(lambda x: x.split(',')[0]),
            size='total_swings',
            color='p_throws',
            color_discrete_map={'R': '#134A8E', 'L': '#E8291C'},
            hover_data=['total_pitches', 'total_swings', 'missed_swings'],
            title=f"<b>Predictability vs. Whiff Rate (Pearson r = {r_whiff:.4f}, n = {len(valid_w)})</b>",
            labels={
                'predictability_pct': 'Predictability (%)',
                'whiff_rate_pct': 'Whiff Rate (%)',
                'p_throws': 'Throws'
            }
        )
        x_line = np.linspace(valid_w['predictability_pct'].min() - 2, valid_w['predictability_pct'].max() + 2, 50)
        fig_r1.add_trace(go.Scatter(
            x=x_line, y=slope_w * x_line + intercept_w,
            mode='lines', name='Linear Trend',
            line=dict(color='#B3261E', width=2, dash='dash')
        ))
        fig_r1.update_traces(textposition='top center', selector=dict(mode='markers+text'))
        fig_r1.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_r1, use_container_width=True)

    with col_r2:
        fig_r2 = px.scatter(
            valid_ev,
            x='predictability_pct',
            y='avg_exit_velo',
            text=valid_ev['player_name'].apply(lambda x: x.split(',')[0]),
            size='measured_bip',
            color='p_throws',
            color_discrete_map={'R': '#134A8E', 'L': '#E8291C'},
            hover_data=['total_pitches', 'measured_bip'],
            title=f"<b>Predictability vs. Avg Exit Velocity (Pearson r = {r_ev:.4f}, n = {len(valid_ev)})</b>",
            labels={
                'predictability_pct': 'Predictability (%)',
                'avg_exit_velo': 'Avg Exit Velocity (mph)',
                'p_throws': 'Throws'
            }
        )
        x_line_ev = np.linspace(valid_ev['predictability_pct'].min() - 2, valid_ev['predictability_pct'].max() + 2, 50)
        fig_r2.add_trace(go.Scatter(
            x=x_line_ev, y=slope_ev * x_line_ev + intercept_ev,
            mode='lines', name='Linear Trend',
            line=dict(color='#B3261E', width=2, dash='dash')
        ))
        fig_r2.update_traces(textposition='top center', selector=dict(mode='markers+text'))
        fig_r2.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Key Sensitivity Finding:</b> Across all 15 pitchers, overall predictability shows a moderate negative 
        association with whiff rate (<b>r = -0.4512</b>) and exit velocity (<b>r = -0.4651</b>). However, removing 4 low-volume 
        relievers (≤100 pitches: Little, Rodríguez, Nance, Bruihl) causes both correlations to collapse near zero 
        (<b>r = -0.1475</b> and <b>r = -0.1879</b>). Aggregate pitch % hides which specific pitch generates whiffs vs. hard contact.
    </div>
    """, unsafe_allow_html=True)

    # Expanders for Jonathan's Proof Table, Brendon Little Case Study, and Throwing Hand Breakdown
    with st.expander("📐 View Pearson Correlation Working Table (Current Sample)"):
        x = valid_w['predictability_pct']
        y = valid_w['whiff_rate_pct']
        calc_df = pd.DataFrame({
            'Pitcher': valid_w['player_name'],
            'Pitches': valid_w['total_pitches'],
            'Predictability (x)': x,
            'Whiff Rate (y)': y,
            'x - x̄': x - x.mean(),
            'y - ȳ': y - y.mean(),
            '(x - x̄)(y - ȳ)': (x - x.mean()) * (y - y.mean())
        })
        st.dataframe(calc_df.round(3), use_container_width=True)

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        with st.expander("🔍 Case Study: Brendon Little (Why Pitch-Level Breakdowns Matter)"):
            little_df = tor_raw[tor_raw['pitcher'] == 663893].copy()
            little_df['ev_bip'] = little_df['launch_speed'].where(little_df['description'].eq('hit_into_play'))
            little_df['is_hr'] = little_df['events'].eq('home_run')
            little_tbl = little_df.groupby('pitch_name').agg(
                Pitches=('pitch_type', 'size'),
                Swings=('is_swing', 'sum'),
                Whiffs=('is_whiff', 'sum'),
                Measured_BIP=('ev_bip', 'count'),
                Avg_EV=('ev_bip', 'mean'),
                HR=('is_hr', 'sum')
            ).reset_index()
            little_tbl['Whiff %'] = (little_tbl['Whiffs'] / little_tbl['Swings'].where(little_tbl['Swings'] > 0) * 100).round(1)
            st.dataframe(little_tbl.round(2), use_container_width=True)
            st.caption("Little had a balanced mix (34.4% predictability) and 33.3% whiff rate (driven by his Knuckle Curve at 55.6%), yet allowed the highest average exit velocity (95.3 mph) and 2 HRs on his Sinker.")

    with col_ex2:
        with st.expander("✋ Pitcher Throwing Hand Comparison (LHP vs. RHP)"):
            hand_tbl = tor_raw.groupby('p_throws').agg(
                Pitchers=('pitcher', 'nunique'),
                Pitches=('pitch_type', 'size'),
                Swings=('is_swing', 'sum'),
                Whiffs=('is_whiff', 'sum')
            ).reset_index()
            hand_ev = tor_raw[tor_raw['description'].eq('hit_into_play')].groupby('p_throws')['launch_speed'].mean().reset_index(name='Avg_EV_mph')
            hand_tbl = hand_tbl.merge(hand_ev, on='p_throws', how='left')
            hand_tbl['Whiff %'] = (hand_tbl['Whiffs'] / hand_tbl['Swings'] * 100).round(2)
            st.dataframe(hand_tbl.round(2), use_container_width=True)
            st.caption("Small LHP sample (n=4 pitchers) makes handedness trends descriptive only—throwing hand does not explain the predictability-outcome relationship.")

# =========================================================
# TAB 4: PITCH TRAITS, LOCATION & INDIVIDUAL SCOUTING
# =========================================================
with tab_traits:
    st.subheader("Pitch-Level Traits & Strike Zone Whiff Optimization")

    swings_subset = tor[tor['is_swing'] & tor['plate_x'].notna() & tor['plate_z'].notna()].copy()
    swings_subset['Result'] = np.where(swings_subset['is_whiff'], 'Whiff (Miss)', 'Contact (Foul/In-Play)')

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_sz = px.scatter(
            swings_subset,
            x='plate_x',
            y='plate_z',
            color='Result',
            color_discrete_map={'Whiff (Miss)': '#E8291C', 'Contact (Foul/In-Play)': '#134A8E'},
            hover_data=['player_name', 'pitch_name', 'release_speed', 'count'],
            opacity=0.65,
            title='<b>2D Strike Zone Map on Swings (plate_x vs. plate_z)</b>',
            labels={'plate_x': 'Horizontal Plate Location (ft)', 'plate_z': 'Vertical Plate Height (ft)'}
        )
        fig_sz.add_shape(
            type="rect", x0=-0.85, x1=0.85, y0=1.5, y1=3.5,
            line=dict(color="black", width=3),
            fillcolor="rgba(150, 255, 150, 0.15)"
        )
        fig_sz.update_layout(xaxis_range=[-2.5, 2.5], yaxis_range=[0.0, 5.0], plot_bgcolor='white', height=480)
        st.plotly_chart(fig_sz, use_container_width=True)

    with col_t2:
        fig_vh = px.scatter(
            swings_subset,
            x='release_speed',
            y='plate_z',
            color='Result',
            color_discrete_map={'Whiff (Miss)': '#E8291C', 'Contact (Foul/In-Play)': '#134A8E'},
            hover_data=['player_name', 'pitch_name', 'count'],
            opacity=0.7,
            title='<b>How Pitch Velocity & Height Create Whiffs</b>',
            labels={'release_speed': 'Pitch Speed (mph)', 'plate_z': 'Plate Height (ft)'}
        )
        fig_vh.add_hrect(y0=1.5, y1=3.5, fillcolor="green", opacity=0.12, line_width=0, annotation_text="Strike Zone (1.5–3.5 ft)")
        fig_vh.update_layout(plot_bgcolor='white', height=480)
        st.plotly_chart(fig_vh, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Pitcher Execution Takeaways:</b><br>
        • <b>Elevate the Fastball:</b> Whiffs cluster at the top of and above the strike zone (3.0–4.0 ft) when velocity reaches 93–98 mph.<br>
        • <b>Bury the Offspeed:</b> Offspeed pitches (80–88 mph) generate chase whiffs below the zone (&lt;1.5 ft), but turn into contact when left up in the zone.<br>
        • <b>Velocity vs. Movement:</b> Extra speed directly boosts whiffs on <b>4-Seam Fastballs &amp; Splitters</b>, whereas <b>Sliders &amp; Sweepers</b> overlap heavily in speed—meaning location and movement matter more than raw velocity.
    </div>
    """, unsafe_allow_html=True)

    # Boxplot of Velocity by Pitch Type (min 15 swings, matching notebook)
    swing_counts_by_pitch = swings_subset['pitch_name'].value_counts()
    popular_pitches = swing_counts_by_pitch[swing_counts_by_pitch >= 15].index
    swings_pop = swings_subset[swings_subset['pitch_name'].isin(popular_pitches)]

    if len(swings_pop) > 0:
        fig_box = px.box(
            swings_pop,
            x='pitch_name',
            y='release_speed',
            color='Result',
            color_discrete_map={'Whiff (Miss)': '#E8291C', 'Contact (Foul/In-Play)': '#134A8E'},
            title='<b>Does Extra Velocity Help Each Pitch Type? (Whiff vs. Contact Speed Distribution)</b>',
            labels={'pitch_name': 'Pitch Type', 'release_speed': 'Pitch Speed (mph)'}
        )
        fig_box.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Individual Pitcher Arsenal & Recommendations")

    selected_p = st.selectbox(
        "Inspect Individual Pitcher Arsenal (B6 Pitch-Level Breakdown):",
        all_pitchers,
        index=all_pitchers.index('Gausman, Kevin') if 'Gausman, Kevin' in all_pitchers else 0
    )
    p_data = tor_raw[tor_raw['player_name'] == selected_p]
    p_mix = p_data.groupby('pitch_name').agg(
        Pitches=('pitch_type', 'count'),
        Avg_Speed=('release_speed', 'mean'),
        Avg_Spin_Axis=('spin_axis', 'mean'),
        Swings=('is_swing', 'sum'),
        Whiffs=('is_whiff', 'sum')
    ).reset_index()
    p_mix['Usage %'] = (p_mix['Pitches'] / len(p_data) * 100).round(1)
    p_mix['Whiff %'] = (p_mix['Whiffs'] / p_mix['Swings'].where(p_mix['Swings'] > 0) * 100).round(1)
    st.dataframe(p_mix.sort_values('Pitches', ascending=False).round(2), use_container_width=True)

    st.markdown("""
    <div class="recommendation-box">
        <b>Actionable Recommendations for Blue Jays Management:</b><br>
        1. <b>Break Count &amp; Repeat Habits:</b> Mix ~15% more elevated four-seam fastballs in Ahead counts and avoid repeating the exact same pitch type back-to-back on 0-2 counts.<br>
        2. <b>Build Secondary Whiff Options:</b> One-dimensional arms (e.g., Gausman, Hoffman, Lauer, Fisher) should develop a second reliable put-away pitch rather than relying on a single pitch for whiffs.<br>
        3. <b>Leverage Balanced Arsenals:</b> Multi-pitch arms with 3+ whiff options (e.g., Scherzer, Bassitt) are the hardest to game-plan against and should be prioritized in high-leverage spots.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer & Filtered CSV Download
# ---------------------------------------------------------
st.markdown("---")
fcol1, fcol2 = st.columns([3, 1])
with fcol1:
    st.caption("⚾ Toronto Blue Jays 2025 Postseason Pitching Analytics | MATH 1130 Group A | Data: MLB 2025 Statcast Postseason")
with fcol2:
    st.download_button(
        label="📥 Download Filtered CSV",
        data=tor.to_csv(index=False).encode('utf-8'),
        file_name='tor_blue_jays_2025_postseason_filtered.csv',
        mime='text/csv'
    )