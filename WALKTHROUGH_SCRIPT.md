# 5-Minute Video Walkthrough Script & Presentation Guide
### MATH 1130 - Interactive Visualization Dashboard
**Project:** Toronto Blue Jays 2025 Postseason Pitching Analytics Dashboard  
**Role:** Sports Data Analyst (Group A)  
**Target Duration:** Exactly 4:45 – 5:00 Minutes  

---

## ⏱️ Video Walkthrough Storyboard & Script

### Part 1: Introduction & Research Question (0:00 – 0:45)
**Screen Action:**
- Show the Streamlit Dashboard landing page on browser fullscreen.
- Point cursor to the top header banner: *"Toronto Blue Jays Pitching Analytics Dashboard"* and the 5 top metric cards.

**Spoken Script:**
> *"Hello everyone, and welcome to our MATH 1130 interactive dashboard presentation. Our team acted as sports data analysts for the Toronto Blue Jays front office, evaluating pitching performance during the team's historic 2025 postseason run to Game 7 of the World Series.*  
> 
> *Our central research question investigated: **How predictable were Blue Jays pitchers in their pitch selection across counts and game states, and did higher predictability penalize their swing-and-miss rates and contact quality? Furthermore, which physical pitch traits and strike zone locations truly optimized postseason success?***  
> 
> *To answer this, our dashboard analyzes 2,788 pitch-by-pitch Statcast records spanning 18 postseason games across three rounds: the ALDS against the Yankees, the ALCS against Seattle, and the World Series against the Dodgers. As you can see on our executive header, the Blue Jays staff posted an overall 28.4% whiff rate and held opponents to an 86.9 mph average exit velocity."*

---

### Part 2: Key Findings – Predictability & Empirical Outcomes (0:45 – 2:00)
**Screen Action:**
- Navigate to **Tab 2 ("Pitch Selection & Predictability")**.
- Hover over the Left-Handed vs. Right-Handed Count Heatmaps.
- Then click over to **Tab 3 ("Predictability vs Outcomes")**.
- Hover over the Scatter Plots showing Trey Yesavage, Kevin Gausman, Chris Bassitt, and the regression trendline.

**Spoken Script:**
> *"Looking at Tab 2, we examined pitch predictability across counts. In early and even counts, Blue Jays pitchers maintained a diverse repertoire, but as counts advanced to two strikes or hitter-friendly counts, reliance on primary offerings jumped significantly. Notice the difference in pitch selection between left-handed and right-handed batters—particularly the heavy reliance on splitters and sweepers against opposite-handed hitters.*  
> 
> *Turning to Tab 3, this is the empirical core of our analysis. We tested whether higher pitch predictability harmed pitcher effectiveness. Our findings revealed a **statistically significant moderate negative correlation of r = -0.4511 between pitch predictability and whiff rate**.  
> 
> *In plain terms: pitchers who over-relied on a single pitch type suffered lower swing-and-miss rates because opposing postseason batters anticipated the offering. Pitchers like Trey Yesavage, who maintained a balanced three-pitch mix, achieved an elite 37% whiff rate. Meanwhile, predictability also correlated negatively with exit velocity (r = -0.4650), showing that predictability directly influences how hitters track and impact the ball."*

---

### Part 3: Interactive Features & Trait Optimization Demo (2:00 – 3:15)
**Screen Action:**
- Go back to Tab 2 and demonstrate the **"Interactive Next-Pitch Prediction Simulator"**:
  - Select "Gausman, Kevin", change Count to "0-2", check "Runners On Base", and watch the probability bars dynamically update.
- Navigate to **Tab 4 ("Pitch Traits & Strike Zone Optimization")**.
- Hover over the 2D Strike Zone scatter plot to show hover tooltips with speed and count.
- Point cursor to the green strike zone box and the speed vs. height graph.

**Spoken Script:**
> *"Our dashboard was built for deep interactivity. Here in Tab 2, we embedded a Machine Learning Next-Pitch Classifier trained on pre-pitch context. If pitching coach Pete Walker wants to know how predictable Kevin Gausman is in an 0-2 count with runners on base, the model predicts his splitter probability jumps to over 55%, demonstrating a measurable 14% predictive edge over a baseline guess.  
> 
> *Next, in Tab 4, we examine physical traits in the strike zone. Our interactive 2D pitch map reveals what we term the **'Two Golden Rules of Postseason Whiffs'**:  
> First, **Elevate the Fastball**—notice the dense cluster of red whiff dots above the zone (between 3.2 and 4.0 feet) when velocity exceeds 94 mph. Opponent swings consistently lag behind elevated four-seamers.  
> Second, **Bury the Offspeed**—the orange whiff dots for splitters and sweepers cluster below the bottom zone border under 1.5 feet. When these offspeed pitches were left over the heart of the plate, they turned into blue contact dots."*

---

### Part 4: Fatigue, Coaching Dossier & Actionable Recommendations (3:15 – 4:15)
**Screen Action:**
- Click on **Tab 5 ("Coaching Cards & Fatigue Dynamics")**.
- Point to the Starting Pitcher Velocity Degradation line chart.
- Under the "Personalized Pitcher Dossier", select "Yesavage, Trey" or "Gausman, Kevin" from the dropdown to display their tactical coaching card.

**Spoken Script:**
> *"In Tab 5, we added actionable features directly addressing front-office strategy. Our fatigue curve analysis tracks velocity decay across pitch count buckets. For our starting rotation—Gausman, Yesavage, Bieber, and Scherzer—average velocity dropped from 89.2 mph in the first 25 pitches down to 87.6 mph past pitch 75—a nearly 2 mph decline that coincided with higher opponent hard contact.  
> 
> *Based on these findings, we offer three clear, actionable recommendations for Blue Jays Baseball Operations:  
> 1. **Implement a Strict Pitch-75 Threshold in October:** Relieve starters before their third time through the order when velocity and whiff rates sharply decay.  
> 2. **Diversify Two-Strike Sequences:** For pitchers with predictability scores above 50%, integrate secondary breaking pitches earlier in the sequence to keep elite hitters off the primary fastball.  
> 3. **Vertical Tunneling Protocol:** Train catchers to target high-and-tight quadrants with four-seamers and force chase contact by burying splitters below the bottom rail."*

---

### Part 5: Limitations & Conclusion (4:15 – 5:00)
**Screen Action:**
- Open the left sidebar and demonstrate the global series filter (switch to "World Series vs LAD") and click "Download Filtered CSV".
- Return to the main overview tab for closing remarks.

**Spoken Script:**
> *"Finally, we must note three important analytical limitations:  
> First, postseason sample sizes are inherently small—some relievers threw fewer than 40 pitches across 18 games, meaning individual correlations should be interpreted with caution.  
> Second, survivor bias exists: pitcher usage was conditional on series game state and opponent matchups.  
> Third, our next-pitch model uses general game state and does not yet account for individual opposing hitter tendencies or pitch sequencing pitch-by-pitch history.  
> 
> *Our dashboard is deployed and publicly accessible with no login required, and full filtered data can be exported with one click. Thank you for your time, and we look forward to your questions!"*

---

## 📋 Rubric Self-Check Matrix

| Rubric Component | Weight | How It Is Satisfied |
| :--- | :---: | :--- |
| **1. Alignment with Project Analysis** | **25%** | Incorporates all crosstabs, predictability definitions, Pearson $r$ calculations ($-0.4511$ and $-0.4650$), strike zone rules, and groupings directly from `data_science_group_a.py`. |
| **2. Visualization Quality** | **20%** | Professional Plotly interactive graphics with official Toronto Blue Jays colors (`#134A8E`, `#E8291C`, `#53ACDC`), clean labels, tooltips, and rulebook strike zone bounding boxes. |
| **3. Interactivity & Usability** | **20%** | Comprehensive sidebar filters (Series, Pitchers, Handedness, Count, Runners), Machine Learning live pitch simulator, interactive tooltips, and data export. |
| **4. Communication of Findings** | **20%** | Executive summary cards, non-technical translations, tactical coaching dossiers, clear strategic recommendations, and a word-for-word 5-minute video script. |
| **5. Design & Technical Execution** | **15%** | Production-ready Streamlit app with `@st.cache_data`, responsive CSS card layout, complete `requirements.txt`, and 1-click Streamlit Cloud public deployment instructions. |
