# Toronto Blue Jays | 2025 Postseason Pitching Analytics Dashboard
### MATH 1130 - Interactive Visualization Dashboard Project

An interactive, executive-grade sports analytics dashboard built for the **Toronto Blue Jays Baseball Operations & Coaching Staff**, analyzing pitching performance across the **2025 MLB Postseason**.

---

## 🚀 Live Public Deployment & Local Run

### Option A: Run Locally in 1 Minute
Ensure you have Python 3.10+ installed:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit dashboard
streamlit run app.py
```
Or with the Windows Python launcher:
```bash
py -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

### Option B: Deploy to Streamlit Community Cloud (Free Public URL - No Sign-in Required)
To meet **Deliverable 1** of the MATH 1130 rubric (publicly accessible URL without permissions):

1. **Push this folder to a GitHub Repository:**
   ```bash
   git init
   git add app.py requirements.txt Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv README.md
   git commit -m "Toronto Blue Jays Postseason Analytics Dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **"Create app"** -> select your repository (`<your-repo-name>`), branch (`main`), and main file path (`app.py`).
   - Click **"Deploy!"**
   - In 1–2 minutes, your dashboard will be live at a public URL: `https://<your-app-name>.streamlit.app/`.
   - **Verification:** Test the link in an incognito window to verify it loads without login or permission requests.

---

## ⚾ Business Problem & Analytical Scope

As sports data analysts working for the **Toronto Blue Jays**, our objective is to evaluate team-specific pitching performance during the **2025 Postseason** across 18 high-stakes games:
- **ALDS vs. New York Yankees** (3-1 series victory)
- **ALCS vs. Seattle Mariners** (4-3 series victory)
- **World Series vs. Los Angeles Dodgers** (7-game battle)

### Core Research Questions Addressed:
1. **Pitch Predictability:** How predictable were Blue Jays pitchers in their pitch selection across counts, batter handedness, and base-runner situations?
2. **Predictive Modeling:** Can pre-pitch game context (count, inning, score diff, runners on base) predict the next pitch type using Machine Learning?
3. **Outcomes & Empirical Research:** Are predictable pitching situations associated with lower whiff rates or worse contact quality (exit velocity)?
4. **Physical Movement & Strike Zone Optimization:** Which release velocities, heights, and plate coordinates maximized swing-and-miss outcomes?
5. **Fatigue & Situational Execution:** Did starting pitchers suffer velocity or command decay as pitch count mounted, and how effectively were two-strike counts converted into outs?

---

## 📊 Dashboard Architecture & Tabs

1. **Executive Summary & Series Command Center:**
   - High-level KPIs: 2,788 pitches, 28.4% Whiff Rate, 86.9 mph Avg Exit Velocity, 73.1% Two-Strike Out Conversion.
   - Postseason Series Progression comparing ALDS, ALCS, and World Series trends.
   - Arsenal distribution donut chart.

2. **Pitch Selection & Predictability (Questions 1 & 2):**
   - Conditional probability contingency heatmaps for Left-Handed vs. Right-Handed batters across all counts.
   - Base-runner impact: Pitch mix shifts between Bases Empty vs. Runners On Base.
   - **Interactive Next-Pitch Machine Learning Classifier (Random Forest):** 45.8% multi-class accuracy vs. 31.4% naive baseline. Adjust game parameters to simulate next-pitch probabilities.

3. **Predictability vs. Outcomes (Question 3 - Empirical Core):**
   - Interactive scatter plots with OLS linear trendlines.
   - **Whiff Rate Correlation ($r = -0.4511$):** Demonstrates that increased reliance on a primary pitch reduces swing-and-miss rates.
   - **Exit Velocity Correlation ($r = -0.4650$):** Demonstrates the contact quality relationship.
   - Mathematical proof table showing manual Pearson correlation calculations (deviation products and sums of squares).

4. **Pitch Traits & Strike Zone Optimization:**
   - 2D coordinate strike zone map (`plate_x` vs `plate_z`) with MLB rulebook boundary.
   - *"Elevate the Fastball, Bury the Offspeed"* speed vs. height analysis.
   - Velocity boxplots comparing Whiffs vs. Contact across all pitch types.

5. **Coaching Cards & Fatigue Dynamics:**
   - Starting pitcher velocity degradation tracking across pitch buckets (`1-25`, `26-50`, `51-75`, `76-100`, `101+`).
   - Two-strike strikeout putaway leaderboard.
   - Personalized pitcher scouting dossier with automated tactical recommendations.

---

## 🛠️ Technology Stack
- **Python 3.11**
- **Streamlit** (Web application framework)
- **Plotly Express & Graph Objects** (Interactive web visualizations)
- **Pandas & NumPy** (Statistical modeling & matrix computations)
- **Scikit-Learn** (Random Forest pitch prediction classifier)

---

## 👥 Course & Author Information
- **Course:** MATH 1130 - Data Science & Visualization
- **Topic:** Interactive Visualization Dashboard
- **Team:** Data Science Group A (Toronto Blue Jays Analytics)
