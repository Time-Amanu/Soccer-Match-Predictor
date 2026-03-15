## Erik Cupsa 
## PL Predictor using scikit-learn to predict from the matches.csv stat sheet containing data from all matches from 2022-2020
## improved by Time-Amanu, improved things> controlled data leakage, increased efficiency and speed by removing redundent code

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# 1. Load and Clean Data
matches = pd.read_csv("matches.csv", index_col=0)

# Convert date and time info
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# 2. Setup Initial Model
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# 3. Define Rolling Averages Function
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    # closed='left' prevents the model from "seeing" the current game's stats
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    return group.dropna(subset=new_cols)

# Features to calculate rolling averages for
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages per team
matches_rolling = matches.groupby("team", group_keys=False).apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling.index = range(matches_rolling.shape[0])

# 4. Prediction Logic
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

# Define predictors (Base features + Rolling stats)
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols

combined, precision = make_predictions(matches_rolling, predictors)

# 5. Team Name Mapping for Merging
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined["team"] = matches_rolling["team"]
combined["date"] = matches_rolling["date"]
combined["opponent"] = matches_rolling["opponent"]
combined["new_team"] = combined["team"].map(mapping)

# 6. Merge Home and Away Predictions
# This allows us to see what the AI predicted for both Team A and Team B in the same match
merged = combined.merge(
    combined, 
    left_on=["date", "new_team"], 
    right_on=["date", "opponent"], 
    suffixes=("_team", "_opp")
)

# 7. Final Output Analysis
print(f"Precision Score: {precision:.2f}")

# Show cases where the model was confident (Predicted a win for one and a loss for the other)
logic_check = merged[(merged["prediction_team"] == 1) & (merged["prediction_opp"] == 0)]
actual_win_rate = (logic_check["actual_team"] == 1).mean()

print(f"Accuracy when predicting a specific winner: {actual_win_rate:.2f}")
