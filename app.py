import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
 
st.title("Trader Sentiment Analysis")
 
# Load feature names from pkl
with open('trading_analysis_model.pkl', 'rb') as f:
    feature_names = list(pickle.load(f))
 
# Load and prepare data
@st.cache_data
def prepare_data():
    fg = pd.read_csv('data/fear_greed_index.csv')
    file_id = "1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs"
    url = f"https://drive.google.com/uc?export=download&id={1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs}"
    td = pd.read_csv(url)
 
    td = td.drop_duplicates(subset=['Account', 'Timestamp IST', 'Trade ID'])
    td['datetime'] = pd.to_datetime(td['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    td['date'] = pd.to_datetime(td['datetime'].dt.date)
    td = td.dropna(subset=['datetime'])
    td['is_win'] = td['Closed PnL'] > 0
    td['is_long'] = td['Side'] == 'BUY'
 
    daily = td.groupby(['date', 'Account']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        num_trades=('Trade ID', 'count'),
        wins=('is_win', 'sum'),
        avg_trade_size=('Size USD', 'mean'),
        long_count=('is_long', 'sum')
    ).reset_index()
 
    daily['win_rate'] = daily['wins'] / daily['num_trades']
    daily['long_ratio'] = daily['long_count'] / daily['num_trades']
 
    fg['date'] = pd.to_datetime(fg['date'])
    df = pd.merge(daily, fg[['date', 'value', 'classification']], on='date', how='inner')
    df = df.rename(columns={'value': 'fg_score', 'classification': 'sentiment'})
    df['sentiment_group'] = df['sentiment'].apply(
        lambda x: 'Fear' if 'Fear' in x else ('Greed' if 'Greed' in x else 'Neutral')
    )
 
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment_group'])
 
    df = df.sort_values(['Account', 'date'])
    df['prev_pnl'] = df.groupby('Account')['daily_pnl'].shift(1)
    df['prev_trades'] = df.groupby('Account')['num_trades'].shift(1)
    df['prev_win_rate'] = df.groupby('Account')['win_rate'].shift(1)
    df['prev_long_ratio'] = df.groupby('Account')['long_ratio'].shift(1)
    df['rolling_3d_pnl'] = df.groupby('Account')['daily_pnl'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_3d_winrate'] = df.groupby('Account')['win_rate'].transform(lambda x: x.shift(1).rolling(3).mean())
 
    df['next_day_pnl'] = df.groupby('Account')['daily_pnl'].shift(-1)
 
    def label(p):
        if p > 10:
            return 'Profit'
        elif p < -10:
            return 'Loss'
        else:
            return 'Neutral'
 
    df['target'] = df['next_day_pnl'].apply(label)
 
    return df
 
df = prepare_data()
 
# Train model
@st.cache_resource
def train_model(df):
    clean = df.dropna(subset=feature_names + ['target'])
    X = clean[feature_names]
    y = clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test
 
model, X_test, y_test = train_model(df)
 
# Section 1 - Data
st.subheader("1. Merged Data (first 30 rows)")
st.dataframe(df[['date', 'Account', 'sentiment', 'fg_score', 'daily_pnl', 'num_trades', 'win_rate', 'long_ratio']].head(30))
 
st.markdown("---")
 
# Section 2 - Summary
st.subheader("2. Average Metrics by Sentiment")
 
summary = df.groupby('sentiment_group').agg(
    avg_pnl=('daily_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trades=('num_trades', 'mean'),
    avg_long_ratio=('long_ratio', 'mean')
).reindex(['Fear', 'Neutral', 'Greed']).reset_index()
 
st.dataframe(summary)
 
st.markdown("---")
 
# Section 3 - Charts
st.subheader("3. Charts")
 
colors = ['red', 'grey', 'green']
 
st.write("Avg Daily PnL by Sentiment")
fig, ax = plt.subplots()
ax.bar(summary['sentiment_group'], summary['avg_pnl'], color=colors)
ax.set_ylabel("PnL (USD)")
st.pyplot(fig)
plt.close()
 
st.write("Avg Win Rate by Sentiment")
fig, ax = plt.subplots()
ax.bar(summary['sentiment_group'], summary['avg_win_rate'], color=colors)
ax.set_ylabel("Win Rate")
st.pyplot(fig)
plt.close()
 
st.write("Avg Trades per Day by Sentiment")
fig, ax = plt.subplots()
ax.bar(summary['sentiment_group'], summary['avg_trades'], color=colors)
ax.set_ylabel("Trades")
st.pyplot(fig)
plt.close()
 
st.markdown("---")
 
# Section 4 - Model Results
st.subheader("4. Model Performance")
 
preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
 
st.write(f"Accuracy: **{accuracy:.2%}**")
st.text(classification_report(y_test, preds))
 
st.write("Feature Importance")
feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
fig, ax = plt.subplots()
ax.barh(feat_imp.index, feat_imp.values, color='steelblue')
ax.set_xlabel("Importance")
st.pyplot(fig)
plt.close()
 
st.markdown("---")
 
# Section 5 - Predict
st.subheader("5. Predict Next Day")
 
fg_score = st.number_input("Fear/Greed Score (0-100)", 0, 100, 50)
sentiment = st.selectbox("Sentiment", ['Fear', 'Neutral', 'Greed'])
num_trades = st.number_input("Trades Today", 1, 500, 10)
win_rate = st.slider("Win Rate Today", 0.0, 1.0, 0.5)
long_ratio = st.slider("Long Ratio Today", 0.0, 1.0, 0.5)
avg_trade_size = st.number_input("Avg Trade Size (USD)", 1, 100000, 1000)
prev_pnl = st.number_input("Yesterday PnL", value=0.0)
prev_trades = st.number_input("Yesterday Trades", 1, 500, 10)
prev_win_rate = st.slider("Yesterday Win Rate", 0.0, 1.0, 0.5)
prev_long_ratio = st.slider("Yesterday Long Ratio", 0.0, 1.0, 0.5)
rolling_3d_pnl = st.number_input("3-Day Avg PnL", value=0.0)
rolling_3d_winrate = st.slider("3-Day Avg Win Rate", 0.0, 1.0, 0.5)
 
sent_map = {'Fear': 0, 'Neutral': 1, 'Greed': 2}
 
if st.button("Predict"):
    row = pd.DataFrame([[
        fg_score, sent_map[sentiment], num_trades, win_rate,
        long_ratio, avg_trade_size, prev_pnl, prev_trades,
        prev_win_rate, prev_long_ratio, rolling_3d_pnl, rolling_3d_winrate
    ]], columns=feature_names)
 
    result = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
 
    st.write(f"**Predicted: {result}**")
    for cls, prob in zip(model.classes_, proba):
        st.write(f"{cls}: {prob:.2%}")
 

