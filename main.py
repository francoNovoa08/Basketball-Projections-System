import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz


def main():
    team_region = pd.read_csv("Team Region Groups.csv")
    games_2022 = pd.read_csv("games_2022.csv")
    east_games = pd.read_csv("East Regional Games to predict.csv")


    numeric_columns = [
        "FGA_2", "FGM_2", "FGA_3", "FGM_3", "FTA", "FTM", "AST", "BLK", "STL", "TOV", "rest_days",
        "TOV_team", "DREB", "OREB", "F_personal", "F_tech", "team_score", "opponent_team_score",
        "largest_lead", "tz_dif_H_E", "prev_game_dist", "travel_dist"
    ]
    
    games_2022["game_date"] = pd.to_datetime(games_2022["game_date"], format="%Y-%m-%d")
    games_2022[numeric_columns] = games_2022[numeric_columns].apply(pd.to_numeric, errors="coerce")

    games_merged = pd.merge(games_2022, team_region, on="team", how="left")

    home = games_merged[games_merged["home_away_NS"] == 1].add_prefix("home_")
    away = games_merged[games_merged["home_away_NS"] == -1].add_prefix("away_")

    games_combined = pd.merge(
        home,
        away,
        left_on="home_game_id",
        right_on="away_game_id",
        suffixes=("", "_away"),
    )
    games_combined["home_win"] = (games_combined["home_team_score"] > games_combined["away_team_score"]).astype(int)

    # Model training
    features = [
        "home_FGA_2", "home_FGM_2", "home_AST",
        "home_FGA_3", "home_FGM_3", "home_FTA",
        "away_FGA_3", "away_FGM_3", "away_FTA",
        "away_FGA_2", "away_FGM_2", "away_AST",
        "home_rest_days", "away_rest_days",
        "home_BLK", "away_BLK",
        "home_FTM", "away_FTM",
        "home_STL", "away_STL",
        "home_TOV_team", "away_TOV_team",
        "home_DREB", "away_DREB",
        "home_OREB", "away_OREB",
        "home_F_personal", "away_F_personal",
        "home_F_tech", "away_F_tech",
        "home_team_score", "away_team_score",
        "home_opponent_team_score", "away_opponent_team_score",
        "home_largest_lead", "away_largest_lead",
        "home_tz_dif_H_E", "away_tz_dif_H_E",
        "home_prev_game_dist", "away_prev_game_dist",
        "home_travel_dist", "away_travel_dist"
    ]
    X = games_combined[features]
    y = games_combined["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    team_stats = games_merged.groupby("team")[numeric_columns].mean().reset_index()

    east_processed = pd.merge(
        east_games,
        team_stats,
        left_on="team_home",
        right_on="team",
        how="left"
    )
    stats_cols = team_stats.columns.difference(['team'])
    east_processed = east_processed.rename(columns={col: f"home_{col}" for col in stats_cols})

    east_processed = pd.merge(
        east_processed,
        team_stats,
        left_on="team_away",
        right_on="team",
        how="left"
    )
    east_processed = east_processed.rename(columns={col: f"away_{col}" for col in stats_cols})

    print("Available columns in east_processed:", east_processed.columns.tolist())
    missing_columns = [col for col in features if col not in east_processed.columns]
    print("Missing columns:", missing_columns)


    X_east = east_processed[features]
    east_games["WINNING %"] = model.predict_proba(X_east)[:, 1] * 100
    east_games.to_csv("East_Predictions.csv", index=False)

    # Bar plot of predicted winning percentages
    east_games_sorted = east_games.sort_values("WINNING %", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="team_home", y="WINNING %", data=east_games_sorted)
    plt.xticks(rotation=45)
    plt.title("Predicted Winning Percentages for Home Teams")
    plt.xlabel("Home Team")
    plt.ylabel("Winning Percentage")
    plt.show()

    # Export one tree from the forest
    tree = model.estimators_[0]
    dot_data = export_graphviz(tree, out_file=None,
                            feature_names=features,
                            class_names=["Loss", "Win"],
                            filled=True, rounded=True,
                            special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree") 


if __name__ == "__main__":
    main()