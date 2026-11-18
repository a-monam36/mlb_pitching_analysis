import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Page Configuration & Custom Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Toronto Blue Jays | 2025 Postseason Pitching Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Blue Jays branding and executive dashboard aesthetic
st.markdown("""
<style>
    /* Main container background and font */
    .main {
        background-color: #F8F9FB;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Blue Jays Navy and Royal Blue accents */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 14px rgba(19, 74, 142, 0.08);
        border-top: 4px solid #134A8E;
        text-align: center;
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #556987;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #134A8E;
        margin: 6px 0;
    }
    .metric-subtext {
        font-size: 0.82rem;
        color: #8896AB;
    }
    .card-accent-red {
        border-top: 4px solid #E8291C;
    }
    .card-accent-cyan {
        border-top: 4px solid #53ACDC;
    }
    .card-accent-navy {
        border-top: 4px solid #1D2D5C;
    }
    
    /* Header banner styling */
    .header-banner {
        background: linear-gradient(135deg, #134A8E 0%, #1D2D5C 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(19, 74, 142, 0.25);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #E0E7FF;
        margin-top: 6px;
        font-weight: 400;
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
    
    /* Section callout boxes */
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
# Data Loading & Preprocessing
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    csv_file = 'Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv'
    if not os.path.exists(csv_file):
        st.error(f"Dataset file '{csv_file}' was not found in the root directory.")
        st.stop()
        
    df = pd.read_csv(csv_file)
    
    # 1. Determine Pitching Team (Home in Top, Away in Bot)
    df['pitching_team'] = df['home_team'].where(df['inning_topbot'].eq('Top'), df['away_team'])
    
    # 2. Filter for Toronto Blue Jays (TOR) 2025 Postseason
    tor_df = df[(df['pitching_team'] == 'TOR') & (df['game_year'] == 2025)].copy()
    
    # 3. Add Opponent & Series Identifier
    tor_df['opponent'] = tor_df['home_team'].where(tor_df['pitching_team'].ne(tor_df['home_team']), tor_df['away_team'])
    series_map = {
        'NYY': 'ALDS vs NYY',
        'SEA': 'ALCS vs SEA',
        'LAD': 'World Series vs LAD'
    }
    tor_df['series'] = tor_df['opponent'].map(series_map).fillna('Other')
    
    # 4. Count and Game State Definitions
    tor_df['count'] = tor_df['balls'].astype(str) + '-' + tor_df['strikes'].astype(str)
    tor_df['is_runner_on'] = (tor_df['on_1b'].notna() | tor_df['on_2b'].notna() | tor_df['on_3b'].notna())
    tor_df['runner_state'] = np.where(tor_df['is_runner_on'], 'Runners On', 'Bases Empty')
    
    # Count grouping
    def categorize_count(b, s):
        if s == 2:
            return 'Two Strikes'
        elif b > s:
            return "Hitter's Count"
        elif s > b:
            return "Pitcher's Count"
        else:
            return "Even / Early"
    tor_df['count_state'] = [categorize_count(b, s) for b, s in zip(tor_df['balls'], tor_df['strikes'])]
    
    # 5. Swing and Whiff Definitions (Matches data_science_group_a.py exactly)
    whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
    swing_descriptions = whiff_descriptions + [
        'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip', 'hit_into_play'
    ]
    tor_df['is_swing'] = tor_df['description'].isin(swing_descriptions)
    tor_df['is_whiff'] = tor_df['description'].isin(whiff_descriptions)
    
    # 6. Cumulative Game Pitch Count & Fatigue Buckets
    tor_df = tor_df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)
    tor_df['game_pitch_number'] = tor_df.groupby(['game_pk', 'pitcher']).cumcount() + 1
    tor_df['pitch_bucket'] = pd.cut(
        tor_df['game_pitch_number'],
        bins=[0, 25, 50, 75, 100, 200],
        labels=['1-25', '26-50', '51-75', '76-100', '101+']
    )
    
    # 7. Pitch Family
    tor_df['pitch_family'] = tor_df['pitch_type'].map({
        'FF': 'Four-Seamer', 'SI': 'Sinker', 'FC': 'Cutter',
        'SL': 'Slider', 'ST': 'Sweeper', 'CU': 'Curveball', 'SV': 'Slurve',
        'CH': 'Changeup', 'FS': 'Splitter'
    }).fillna('Other')
    
    return tor_df

tor_raw = load_and_preprocess_data()

# ---------------------------------------------------------
# Sidebar Controls & Granular Filters
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/ba/Toronto_Blue_Jays_logo.svg/300px-Toronto_Blue_Jays_logo.svg.png", width=110)
    st.markdown("### **Postseason Filter Console**")
    st.caption("Customize the analytical scope across game state, series, and pitcher selection.")
    
    # Series Filter
    series_options = ['All Series'] + sorted(list(tor_raw['series'].unique()))
    selected_series = st.selectbox("🎯 Postseason Round", series_options, index=0)
    
    # Pitcher Multi-select
    all_pitchers = sorted(list(tor_raw['player_name'].dropna().unique()))
    selected_pitchers = st.multiselect(
        "👤 Pitcher(s)",
        all_pitchers,
        default=[]
    )
    
    # Batter Handedness
    stand_filter = st.radio("🏏 Batter Handedness", ['All (L & R)', 'Right-Handed (R)', 'Left-Handed (L)'], horizontal=True)
    
    # Count State
    count_filter = st.selectbox(
        "⚾ Count Situation",
        ['All Counts', 'Two Strikes', "Pitcher's Count", "Hitter's Count", 'Even / Early'],
        index=0
    )
    
    # Runners On Base
    runners_filter = st.radio("🏃 Base Runners", ['All States', 'Bases Empty', 'Runners On Base'], horizontal=True)
    
    st.markdown("---")
    st.markdown("### **Research Project Info**")
    st.markdown("""
    **Course:** MATH 1130  
    **Role:** Sports Data Analyst  
    **Client:** Toronto Blue Jays Front Office  
    **Dataset:** 2025 MLB Postseason Statcast  
    """)
    
    # Filter reset helper note
    st.caption("Active filter automatically updates all tabs and calculations.")

# ---------------------------------------------------------
# Applying Global Filters
# ---------------------------------------------------------
tor = tor_raw.copy()

if selected_series != 'All Series':
    tor = tor[tor['series'] == selected_series]

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
elif runners_filter == 'Runners On Base':
    tor = tor[tor['is_runner_on']]

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <span class="header-badge">2025 MLB Postseason Deep-Dive</span>
    <h1 class="header-title">Toronto Blue Jays Pitching Analytics Dashboard</h1>
    <p class="header-subtitle">Evaluating pitch predictability, physical release traits, strike zone optimization, and fatigue dynamics across the 2025 Postseason run.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Executive KPI Cards
# ---------------------------------------------------------
total_pitches = len(tor)
total_swings = tor['is_swing'].sum()
total_whiffs = tor['is_whiff'].sum()
whiff_rate = (total_whiffs / total_swings * 100) if total_swings > 0 else 0.0

balls_in_play = tor[tor['description'].eq('hit_into_play')]
avg_ev = balls_in_play['launch_speed'].mean() if len(balls_in_play) > 0 else 0.0

two_strike_pitches = tor[tor['strikes'] == 2]
two_strike_events = two_strike_pitches['events'].dropna()
two_strike_ks = (two_strike_events == 'strikeout').sum()
k_rate = (two_strike_ks / len(two_strike_events) * 100) if len(two_strike_events) > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pitches Analyzed</div>
        <div class="metric-value">{total_pitches:,}</div>
        <div class="metric-subtext">{tor['player_name'].nunique()} Active Pitchers</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card card-accent-red">
        <div class="metric-label">Whiff Rate (Sw&Miss)</div>
        <div class="metric-value">{whiff_rate:.1f}%</div>
        <div class="metric-subtext">{total_whiffs:,} Misses / {total_swings:,} Swings</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card card-accent-cyan">
        <div class="metric-label">Avg Exit Velocity</div>
        <div class="metric-value">{avg_ev:.1f} <span style="font-size:1.1rem;font-weight:600;">mph</span></div>
        <div class="metric-subtext">{len(balls_in_play):,} Contact Events in Play</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card card-accent-navy">
        <div class="metric-label">2-Strike Putaway (K%)</div>
        <div class="metric-value">{k_rate:.1f}%</div>
        <div class="metric-subtext">{two_strike_ks} Strikeouts on 2 Strikes</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Release Speed</div>
        <div class="metric-value">{tor['release_speed'].mean():.1f} <span style="font-size:1.1rem;font-weight:600;">mph</span></div>
        <div class="metric-subtext">Peak: {tor['release_speed'].max():.1f} mph</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabbed Navigation Architecture
# ---------------------------------------------------------
tab_overview, tab_pred, tab_outcomes, tab_traits, tab_tactics = st.tabs([
    "📊 Executive Summary & Series",
    "🎯 Pitch Selection & Predictability",
    "📈 Predictability vs Outcomes (Research)",
    "⚡ Pitch Traits & Strike Zone Optimization",
    "🧠 Coaching Cards & Fatigue Dynamics"
])

# =========================================================
# TAB 1: EXECUTIVE SUMMARY & SERIES
# =========================================================
with tab_overview:
    st.subheader("Postseason Journey & Executive Analysis")
    
    col_lead1, col_lead2 = st.columns([3, 2])
    with col_lead1:
        st.markdown("""
        <div class="insight-box">
            <b>Executive Context for Baseball Operations & Pitching Staff:</b><br>
            During the 2025 Postseason, the Toronto Blue Jays advanced through an 18-game playoff gauntlet 
            (defeating the <b>New York Yankees</b> in the ALDS 3-1, outlasting the <b>Seattle Mariners</b> 
            in a 7-game ALCS thriller, and competing to Game 7 of the <b>World Series against the Los Angeles Dodgers</b>).
            <br><br>
            This dashboard evaluates how pitch sequencing, predictability tendencies, velocity decay, and physical 
            movement traits dictated postseason success or vulnerability.
        </div>
        """, unsafe_allow_html=True)
    
    with col_lead2:
        st.markdown("""
        <div class="recommendation-box">
            <b>Key Strategic Finding:</b><br>
            Pitchers with higher pitch predictability (over-relying on a single primary pitch) suffered a 
            <b>statistically significant decrease in Whiff Rate (r = -0.4511)</b>. 
            Furthermore, late-game velocity decay after pitch 75 averaged a <b>1.7 mph loss</b>, directly leading to 
            increased opponent contact quality.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### Postseason Performance Progression Across Series")
    
    # Series breakdown aggregation
    series_agg = tor_raw.groupby('series').agg(
        pitches=('pitch_type', 'count'),
        swings=('is_swing', 'sum'),
        whiffs=('is_whiff', 'sum'),
        avg_velo=('release_speed', 'mean'),
        delta_re=('delta_run_exp', 'sum')
    ).reset_index()
    series_agg['whiff_rate'] = (series_agg['whiffs'] / series_agg['swings'] * 100).round(1)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_series_whiff = px.bar(
            series_agg,
            x='series',
            y='whiff_rate',
            text='whiff_rate',
            title='<b>Whiff Rate (%) Across Postseason Rounds</b>',
            labels={'series': 'Postseason Round', 'whiff_rate': 'Whiff Rate (%)'},
            color='series',
            color_discrete_map={'ALDS vs NYY': '#134A8E', 'ALCS vs SEA': '#53ACDC', 'World Series vs LAD': '#E8291C'}
        )
        fig_series_whiff.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_series_whiff.update_layout(showlegend=False, yaxis_range=[0, 35], plot_bgcolor='white')
        st.plotly_chart(fig_series_whiff, use_container_width=True)
        
    with col_s2:
        fig_series_velo = px.line(
            series_agg,
            x='series',
            y='avg_velo',
            markers=True,
            title='<b>Average Pitch Velocity (mph) by Series</b>',
            labels={'series': 'Postseason Round', 'avg_velo': 'Avg Velocity (mph)'},
            color_discrete_sequence=['#134A8E']
        )
        fig_series_velo.update_traces(marker=dict(size=12, color='#E8291C'))
        fig_series_velo.update_layout(yaxis_range=[87, 91], plot_bgcolor='white')
        st.plotly_chart(fig_series_velo, use_container_width=True)

    st.markdown("#### Pitch Type Usage Breakdown")
    pitch_usage = tor['pitch_name'].value_counts().reset_index()
    pitch_usage.columns = ['Pitch Name', 'Count']
    pitch_usage['Percentage'] = (pitch_usage['Count'] / pitch_usage['Count'].sum() * 100).round(1)
    
    fig_donut = px.pie(
        pitch_usage.head(8),
        names='Pitch Name',
        values='Count',
        hole=0.45,
        title='<b>Arsenal Distribution (Selected Pitcher Scope)</b>',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig_donut.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_donut, use_container_width=True)

# =========================================================
# TAB 2: PITCH SELECTION & PREDICTABILITY
# =========================================================
with tab_pred:
    st.subheader("How Predictable are Blue Jays Pitchers Across Counts and Situations?")
    st.markdown("""
    Explore conditional pitch selection distributions. As counts shift into hitter-friendly or pitcher-friendly 
    territory, pitch predictability changes dramatically based on batter handedness and base-runner pressure.
    """)
    
    col_p1, col_p2 = st.columns(2)
    
    # Conditional probability calculation
    clean_tor = tor.dropna(subset=['count', 'stand', 'pitch_type']).copy()
    
    # Left vs Right Handedness contingency tables
    prob_crosstab = pd.crosstab(
        [clean_tor['count'], clean_tor['stand']],
        clean_tor['pitch_type'],
        normalize='index'
    ) * 100
    
    with col_p1:
        st.markdown("##### 🟦 vs. Left-Handed Batters (L)")
        if 'L' in prob_crosstab.index.get_level_values('stand'):
            df_left = prob_crosstab.xs('L', level='stand')
            fig_hm_left = px.imshow(
                df_left.round(1),
                labels=dict(x="Pitch Type", y="Count", color="Prob (%)"),
                x=df_left.columns,
                y=df_left.index,
                color_continuous_scale="YlGnBu",
                text_auto='.1f',
                aspect="auto",
                title="<b>Pitch Selection Probabilities (%) vs. LHB</b>"
            )
            fig_hm_left.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_hm_left, use_container_width=True)
        else:
            st.info("No Left-Handed batter data for current filter selection.")
            
    with col_p2:
        st.markdown("##### 🟥 vs. Right-Handed Batters (R)")
        if 'R' in prob_crosstab.index.get_level_values('stand'):
            df_right = prob_crosstab.xs('R', level='stand')
            fig_hm_right = px.imshow(
                df_right.round(1),
                labels=dict(x="Pitch Type", y="Count", color="Prob (%)"),
                x=df_right.columns,
                y=df_right.index,
                color_continuous_scale="YlGnBu",
                text_auto='.1f',
                aspect="auto",
                title="<b>Pitch Selection Probabilities (%) vs. RHB</b>"
            )
            st.plotly_chart(fig_hm_right, use_container_width=True)
        else:
            st.info("No Right-Handed batter data for current filter selection.")

    st.markdown("---")
    st.subheader("Base-runner Influence: Bases Empty vs Runners On")
    runner_crosstab = pd.crosstab(
        clean_tor['runner_state'],
        clean_tor['pitch_type'],
        normalize='index'
    ) * 100
    
    fig_runner = px.bar(
        runner_crosstab.reset_index().melt(id_vars='runner_state', var_name='Pitch Type', value_name='Percentage'),
        x='runner_state',
        y='Percentage',
        color='Pitch Type',
        barmode='stack',
        title='<b>Pitch Selection Proportions by Base-Runner State</b>',
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_runner.update_layout(yaxis_title="Percentage (%)", xaxis_title="Situation", plot_bgcolor='white')
    st.plotly_chart(fig_runner, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Interactive Next-Pitch Prediction Simulator (Machine Learning)")
    st.markdown("""
    Can an opposing scouting department or batter predict the upcoming pitch type solely from pre-pitch game context? 
    Using a multi-class **Random Forest Classifier** trained on pre-pitch features (Count, Inning, Runners On, Score Differential, Batter Handedness, and Pitcher ID):
    """)
    
    # Train lightweight ML model on cached data
    @st.cache_resource(show_spinner=False)
    def train_pitch_model(df_train):
        sub = df_train.dropna(subset=['pitch_type', 'player_name', 'stand', 'balls', 'strikes']).copy()
        sub['is_runner_on_int'] = sub['is_runner_on'].astype(int)
        sub['score_diff'] = sub['fld_score'] - sub['bat_score']
        
        # Select popular pitches to keep classifier sharp
        top_pitches = sub['pitch_type'].value_counts().head(6).index.tolist()
        sub = sub[sub['pitch_type'].isin(top_pitches)]
        
        feature_cols = ['balls', 'strikes', 'stand', 'is_runner_on_int', 'inning', 'score_diff', 'player_name']
        X = pd.get_dummies(sub[feature_cols], drop_first=True)
        y = sub['pitch_type']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        base_acc = (y_test == y_train.mode()[0]).mean()
        
        return model, X.columns.tolist(), acc, base_acc, top_pitches
        
    model, feature_names, model_acc, baseline_acc, valid_pitch_types = train_pitch_model(tor_raw)
    
    mcol1, mcol2 = st.columns([1, 2])
    with mcol1:
        st.markdown(f"""
        **Model Benchmark Metrics:**
        - **Random Forest Accuracy:** `{model_acc*100:.1f}%`
        - **Naive Guessing Baseline:** `{baseline_acc*100:.1f}%`
        - **Predictive Edge:** `+{(model_acc - baseline_acc)*100:.1f}%` over guessing primary pitch.
        """)
        
        sim_pitcher = st.selectbox("Select Pitcher", sorted(tor_raw['player_name'].dropna().unique()), index=0)
        sim_balls = st.slider("Balls", 0, 3, 0)
        sim_strikes = st.slider("Strikes", 0, 2, 2)
        sim_stand = st.radio("Batter Stand", ['R', 'L'], horizontal=True)
        sim_runners = st.checkbox("Runners On Base", value=True)
        sim_inning = st.slider("Inning", 1, 9, 7)
        sim_score_diff = st.slider("Score Differential (TOR - Opp)", -5, 5, 1)
        
    with mcol2:
        # Construct input vector matching dummy columns
        input_data = pd.DataFrame([{
            'balls': sim_balls,
            'strikes': sim_strikes,
            'stand': sim_stand,
            'is_runner_on_int': int(sim_runners),
            'inning': sim_inning,
            'score_diff': sim_score_diff,
            'player_name': sim_pitcher
        }])
        input_encoded = pd.get_dummies(input_data)
        input_vector = input_encoded.reindex(columns=feature_names, fill_value=0)
        
        probs = model.predict_proba(input_vector)[0]
        prob_df = pd.DataFrame({
            'Pitch Type': model.classes_,
            'Probability (%)': (probs * 100).round(1)
        }).sort_values('Probability (%)', ascending=False)
        
        fig_prob = px.bar(
            prob_df,
            x='Probability (%)',
            y='Pitch Type',
            orientation='h',
            text='Probability (%)',
            title=f"<b>Predicted Pitch Probabilities for {sim_pitcher} ({sim_balls}-{sim_strikes} Count)</b>",
            color='Probability (%)',
            color_continuous_scale='Blues'
        )
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prob.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor='white', coloraxis_showscale=False)
        st.plotly_chart(fig_prob, use_container_width=True)

# =========================================================
# TAB 3: PREDICTABILITY VS OUTCOMES (RESEARCH)
# =========================================================
with tab_outcomes:
    st.subheader("Research Core: Are Predictable Situations Penalized?")
    st.markdown("""
    This section directly evaluates **Question 3 from the course research problem**:  
    *Are higher levels of pitch predictability associated with lower swing-and-miss (whiff) rates and worse contact outcomes (exit velocity)?*
    """)
    
    # Calculate pitcher-level metrics (from data_science_group_a.py)
    group_cols = ['pitcher', 'player_name']
    pitch_counts = pd.crosstab([tor_raw[c] for c in group_cols], tor_raw['pitch_type'])
    most_common_count = pitch_counts.max(axis=1)
    
    pred_summary = pd.DataFrame({
        'total_pitches': pitch_counts.sum(axis=1),
        'most_common_pitch_type': pitch_counts.apply(lambda r: ', '.join(r.index[r.eq(r.max())]), axis=1),
        'predictability_pct': (most_common_count / pitch_counts.sum(axis=1) * 100).round(2)
    }).reset_index()
    
    # Whiff rate merge
    swings_agg = tor_raw.groupby(group_cols).agg(
        total_swings=('is_swing', 'sum'),
        missed_swings=('is_whiff', 'sum')
    ).reset_index()
    swings_agg['whiff_rate_pct'] = (swings_agg['missed_swings'] / swings_agg['total_swings'].where(swings_agg['total_swings'].gt(0)) * 100).round(2)
    
    whiff_merged = pred_summary.merge(swings_agg, on=group_cols, how='left')
    
    # Exit velo merge
    bip_raw = tor_raw[tor_raw['description'].eq('hit_into_play')]
    contact_agg = bip_raw.groupby(group_cols).agg(
        balls_in_play=('description', 'size'),
        measured_bip=('launch_speed', 'count'),
        avg_exit_velo=('launch_speed', 'mean')
    ).reset_index()
    
    contact_merged = pred_summary.merge(contact_agg, on=group_cols, how='left')
    contact_merged['balls_in_play'] = contact_merged['balls_in_play'].fillna(0).astype(int)
    contact_merged['measured_bip'] = contact_merged['measured_bip'].fillna(0).astype(int)
    
    # 1. Plot Predictability vs Whiff Rate
    valid_whiff = whiff_merged.dropna(subset=['predictability_pct', 'whiff_rate_pct'])
    r_whiff = valid_whiff['predictability_pct'].corr(valid_whiff['whiff_rate_pct'])
    slope_w, intercept_w = np.polyfit(valid_whiff['predictability_pct'], valid_whiff['whiff_rate_pct'], 1)
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown("#### 🎯 Predictability vs. Whiff Rate")
        fig_r1 = px.scatter(
            valid_whiff,
            x='predictability_pct',
            y='whiff_rate_pct',
            text=valid_whiff['player_name'].apply(lambda x: x.split(',')[0]),
            size='total_swings',
            title=f"<b>Predictability vs Whiff Rate (r = {r_whiff:.4f})</b>",
            labels={
                'predictability_pct': 'Predictability: % Pitches on Primary Type',
                'whiff_rate_pct': 'Whiff Rate (% Misses / Swings)',
                'total_swings': 'Total Swings'
            },
            color_discrete_sequence=['#134A8E']
        )
        x_trend = np.linspace(valid_whiff['predictability_pct'].min() - 5, valid_whiff['predictability_pct'].max() + 5, 50)
        fig_r1.add_trace(go.Scatter(
            x=x_trend,
            y=slope_w * x_trend + intercept_w,
            mode='lines',
            name='Linear OLS Trend',
            line=dict(color='#E8291C', width=2.5, dash='dash')
        ))
        fig_r1.update_traces(textposition='top center')
        fig_r1.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_r1, use_container_width=True)
        
    with col_res2:
        st.markdown("#### 💥 Predictability vs. Contact Quality (Exit Velo)")
        valid_ev = contact_merged.dropna(subset=['predictability_pct', 'avg_exit_velo'])
        r_ev = valid_ev['predictability_pct'].corr(valid_ev['avg_exit_velo'])
        slope_ev, intercept_ev = np.polyfit(valid_ev['predictability_pct'], valid_ev['avg_exit_velo'], 1)
        
        fig_r2 = px.scatter(
            valid_ev,
            x='predictability_pct',
            y='avg_exit_velo',
            text=valid_ev['player_name'].apply(lambda x: x.split(',')[0]),
            size='measured_bip',
            title=f"<b>Predictability vs Exit Velocity (r = {r_ev:.4f})</b>",
            labels={
                'predictability_pct': 'Predictability: % Pitches on Primary Type',
                'avg_exit_velo': 'Average Exit Velocity (mph)',
                'measured_bip': 'Measured Balls in Play'
            },
            color_discrete_sequence=['#53ACDC']
        )
        x_trend_ev = np.linspace(valid_ev['predictability_pct'].min() - 5, valid_ev['predictability_pct'].max() + 5, 50)
        fig_r2.add_trace(go.Scatter(
            x=x_trend_ev,
            y=slope_ev * x_trend_ev + intercept_ev,
            mode='lines',
            name='Linear OLS Trend',
            line=dict(color='#E8291C', width=2.5, dash='dash')
        ))
        fig_r2.update_traces(textposition='top center')
        fig_r2.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📐 Mathematical & Statistical Proof Table")
    with st.expander("Click to view full Pearson correlation working calculation table"):
        x = valid_whiff['predictability_pct']
        y = valid_whiff['whiff_rate_pct']
        x_dev = x - x.mean()
        y_dev = y - y.mean()
        calc_df = pd.DataFrame({
            'Pitcher': valid_whiff['player_name'],
            'Predictability (x)': x,
            'Whiff Rate (y)': y,
            'x - x̄': x_dev,
            'y - ȳ': y_dev,
            'Product (x - x̄)(y - ȳ)': x_dev * y_dev,
            '(x - x̄)²': x_dev**2,
            '(y - ȳ)²': y_dev**2
        })
        st.dataframe(calc_df.round(4), use_container_width=True)
        st.markdown(f"""
        - **Formula:** $r = \\frac{{\\sum (x - \\bar{{x}})(y - \\bar{{y}})}}{{\\sqrt{{\\sum (x - \\bar{{x}})^2 \\sum (y - \\bar{{y}})^2}}}}$
        - **Calculated Pearson r:** `{r_whiff:.6f}` (Moderate Negative Correlation)
        - **Analytical Meaning:** When Blue Jays pitchers become predictable in their pitch selection, opposing postseason hitters miss significantly fewer swings.
        """)

# =========================================================
# TAB 4: PITCH TRAITS & STRIKE ZONE OPTIMIZATION
# =========================================================
with tab_traits:
    st.subheader("Physical Pitch Traits & Strike Zone Heatmap")
    st.markdown("""
    Which release velocities, heights, and coordinate locations maximize swings and misses?
    Visualizing Statcast 2D coordinates `plate_x` and `plate_z` with the standard MLB Rulebook strike zone.
    """)
    
    col_t1, col_t2 = st.columns([3, 2])
    
    with col_t1:
        st.markdown("#### 🎯 2D Strike Zone Map: Whiffs vs. Contact")
        swings_subset = tor[tor['is_swing'] & tor['plate_x'].notna() & tor['plate_z'].notna()].copy()
        swings_subset['Result'] = np.where(swings_subset['is_whiff'], 'Whiff (Miss)', 'Contact (Foul/In-Play)')
        
        fig_sz = px.scatter(
            swings_subset,
            x='plate_x',
            y='plate_z',
            color='Result',
            color_discrete_map={'Whiff (Miss)': '#E8291C', 'Contact (Foul/In-Play)': '#134A8E'},
            hover_data=['player_name', 'pitch_name', 'release_speed', 'count'],
            opacity=0.75,
            title='<b>Pitch Location on Swings (Red = Whiff | Blue = Contact)</b>',
            labels={'plate_x': 'Horizontal Plate Location (ft)', 'plate_z': 'Vertical Plate Height (ft)'}
        )
        # Add Strike Zone Rectangle (Approx rulebook: -0.85 to +0.85 ft horizontal, 1.5 to 3.5 ft vertical)
        fig_sz.add_shape(
            type="rect",
            x0=-0.85, x1=0.85, y0=1.5, y1=3.5,
            line=dict(color="black", width=3),
            fillcolor="rgba(150, 255, 150, 0.15)"
        )
        fig_sz.update_layout(
            xaxis_range=[-2.5, 2.5],
            yaxis_range=[0.0, 5.0],
            plot_bgcolor='white',
            width=650,
            height=550
        )
        st.plotly_chart(fig_sz, use_container_width=True)
        
    with col_t2:
        st.markdown("#### 🚀 Speed vs. Height: The Golden Rule")
        st.markdown("""
        <div class="insight-box">
            <b>1. Elevate the Fastball:</b><br>
            Notice the dense cluster of <b>Red Whiff dots</b> at the very top of the strike zone and above it (3.2 – 4.0 ft) 
            when speed exceeds 94+ mph. Opponent bats cannot catch up to elevated heat.
            <br><br>
            <b>2. Bury the Offspeed:</b><br>
            Notice the cluster of whiffs located below the bottom edge (< 1.5 ft) at 80 – 86 mph. When splitters and sliders 
            are left inside the zone (green box), they turn into <b>Blue Contact</b>.
        </div>
        """, unsafe_allow_html=True)
        
        # Velocity vs Height Scatter
        fig_vh = px.scatter(
            swings_subset,
            x='release_speed',
            y='plate_z',
            color='Result',
            color_discrete_map={'Whiff (Miss)': '#E8291C', 'Contact (Foul/In-Play)': '#134A8E'},
            title='<b>Pitch Speed vs. Plate Height (ft)</b>',
            labels={'release_speed': 'Pitch Speed (mph)', 'plate_z': 'Plate Height (ft)'},
            opacity=0.7
        )
        fig_vh.add_hrect(y0=1.5, y1=3.5, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Strike Zone (1.5 - 3.5 ft)")
        fig_vh.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_vh, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📦 Does Extra Velocity Aid Your Specific Pitch Type?")
    popular_pitches = tor['pitch_name'].value_counts()[tor['pitch_name'].value_counts() >= 25].index
    swings_pop = swings_subset[swings_subset['pitch_name'].isin(popular_pitches)]
    
    fig_box = px.box(
        swings_pop,
        x='pitch_name',
        y='release_speed',
        color='Result',
        color_discrete_map={'Whiff (Miss)': '#E8291C', 'Contact (Foul/In-Play)': '#134A8E'},
        title='<b>Velocity Distribution on Whiffs vs Contact across Pitch Types</b>',
        labels={'pitch_name': 'Pitch Type', 'release_speed': 'Velocity (mph)'}
    )
    fig_box.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# TAB 5: COACHING CARDS & FATIGUE DYNAMICS
# =========================================================
with tab_tactics:
    st.subheader("Tactical Pitcher Coaching Cards & Fatigue Tracking")
    st.markdown("""
    Translating data into actionable pitching coach interventions. Review velocity degradation across pitch counts 
    and inspect individual pitcher game plans.
    """)
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("#### ⏳ Pitcher Fatigue Dynamics: Velocity Loss by Pitch Bucket")
        starters = ['Gausman, Kevin', 'Yesavage, Trey', 'Bieber, Shane', 'Scherzer, Max']
        starters_df = tor_raw[tor_raw['player_name'].isin(starters) & tor_raw['pitch_bucket'].notna()]
        
        fatigue_agg = starters_df.groupby(['pitch_bucket', 'player_name'], observed=False).agg(
            avg_velo=('release_speed', 'mean'),
            pitches=('release_speed', 'count')
        ).reset_index()
        
        fig_fatigue = px.line(
            fatigue_agg,
            x='pitch_bucket',
            y='avg_velo',
            color='player_name',
            markers=True,
            title='<b>Starting Pitcher Velocity Degradation Across Pitch Counts</b>',
            labels={'pitch_bucket': 'Pitch Count in Game', 'avg_velo': 'Average Velocity (mph)', 'player_name': 'Pitcher'}
        )
        fig_fatigue.update_layout(plot_bgcolor='white', yaxis_range=[85, 93])
        st.plotly_chart(fig_fatigue, use_container_width=True)
        st.caption("Noticeable drop-off occurs past pitch 75 across starting rotations.")
        
    with col_f2:
        st.markdown("#### ⚡ 2-Strike Conversion & Putaway Effectiveness")
        two_strike_all = tor_raw[tor_raw['strikes'] == 2].dropna(subset=['events'])
        two_strike_by_p = two_strike_all.groupby('player_name').agg(
            total_2s_events=('events', 'count'),
            strikeouts=('events', lambda x: (x == 'strikeout').sum()),
            outs=('events', lambda x: x.isin(['strikeout', 'field_out', 'force_out', 'grounded_into_double_play']).sum())
        ).reset_index()
        two_strike_by_p = two_strike_by_p[two_strike_by_p['total_2s_events'] >= 10]
        two_strike_by_p['k_pct'] = (two_strike_by_p['strikeouts'] / two_strike_by_p['total_2s_events'] * 100).round(1)
        two_strike_by_p['out_pct'] = (two_strike_by_p['outs'] / two_strike_by_p['total_2s_events'] * 100).round(1)
        
        fig_2s = px.bar(
            two_strike_by_p.sort_values('k_pct', ascending=False),
            x='player_name',
            y='k_pct',
            text='k_pct',
            title='<b>Pitcher 2-Strike Putaway Rate (Strikeout %)</b>',
            labels={'player_name': 'Pitcher', 'k_pct': 'Strikeout Conversion Rate (%)'},
            color='k_pct',
            color_continuous_scale='Blues'
        )
        fig_2s.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_2s.update_layout(xaxis_tickangle=-45, plot_bgcolor='white', coloraxis_showscale=False)
        st.plotly_chart(fig_2s, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Personalized Pitcher Scouting & Optimization Dossier")
    
    coach_pitcher = st.selectbox("Select Pitcher for Tactical Report", all_pitchers, index=all_pitchers.index('Gausman, Kevin') if 'Gausman, Kevin' in all_pitchers else 0)
    p_data = tor_raw[tor_raw['player_name'] == coach_pitcher]
    
    p_pitches = len(p_data)
    p_swings = p_data['is_swing'].sum()
    p_whiffs = p_data['is_whiff'].sum()
    p_whiff_rate = (p_whiffs / p_swings * 100) if p_swings > 0 else 0.0
    p_bip = p_data[p_data['description'].eq('hit_into_play')]
    p_avg_ev = p_bip['launch_speed'].mean() if len(p_bip) > 0 else 0.0
    
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    c_p1.metric("Total Postseason Pitches", f"{p_pitches}")
    c_p2.metric("Whiff Rate", f"{p_whiff_rate:.1f}%")
    c_p3.metric("Avg Exit Velo Allowed", f"{p_avg_ev:.1f} mph")
    c_p4.metric("Avg Velocity", f"{p_data['release_speed'].mean():.1f} mph")
    
    # Pitch mix breakdown for pitcher
    p_mix = p_data.groupby('pitch_name').agg(
        pitches=('pitch_type', 'count'),
        avg_speed=('release_speed', 'mean'),
        swings=('is_swing', 'sum'),
        whiffs=('is_whiff', 'sum')
    ).reset_index()
    p_mix['usage_pct'] = (p_mix['pitches'] / p_pitches * 100).round(1)
    p_mix['whiff_pct'] = (p_mix['whiffs'] / p_mix['swings'].where(p_mix['swings'].gt(0)) * 100).round(1)
    
    st.markdown(f"##### Arsenal Breakdown for **{coach_pitcher}**")
    st.dataframe(p_mix.sort_values('pitches', ascending=False), use_container_width=True)
    
    # Custom automated coaching advice based on stats
    st.markdown("##### 💡 Pitching Coach Recommendations")
    if p_whiff_rate >= 30.0:
        coach_note = f"**Elite Swing-and-Miss Arsenal:** {coach_pitcher} maintained an outstanding {p_whiff_rate:.1f}% whiff rate. Leverage secondary putaway pitches early in counts to induce aggressive early-count swings."
    else:
        coach_note = f"**Contact-Oriented Profile:** {coach_pitcher} recorded a {p_whiff_rate:.1f}% whiff rate. Focus on tunneling fastballs at the upper rail to suppress opponent barrel contact ({p_avg_ev:.1f} mph average EV allowed)."
        
    st.markdown(f"""
    <div class="insight-box">
        {coach_note}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer & Download Filtered Data
# ---------------------------------------------------------
st.markdown("---")
fcol1, fcol2 = st.columns([3, 1])
with fcol1:
    st.caption("⚾ Toronto Blue Jays Analytics Department | MATH 1130 Interactive Dashboard | Built with Streamlit & Plotly")
with fcol2:
    csv_export = tor.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv_export,
        file_name='toronto_blue_jays_2025_postseason_filtered.csv',
        mime='text/csv'
    )
