import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz


def load_data():
    try:
        team_regions = pd.read_csv("src/Data/Team Region Groups.csv")
        games_data = pd.read_csv("src/Data/games_2022.csv")
        east_games = pd.read_csv("src/Data/East Regional Games to predict.csv")
    except FileNotFoundError:
        print("File not found, check file paths")
        return
    except pd.errors.ParserError:
        print("CSV parsing error")
        return
    except Exception as e:
        print(f"Error occurred: {e}")
        return 

    return team_regions, games_data, east_games


def process_data(games_df, team_regions):
    num_type_cols = [
        "FGA_2", "FGM_2", "FGA_3", "FGM_3", "FTA", "FTM", "AST", "BLK", "STL", "TOV", "rest_days",
        "TOV_team", "DREB", "OREB", "F_personal", "F_tech", "team_score", "opponent_team_score",
        "largest_lead", "prev_game_dist", "travel_dist"
    ]

    try:
        games_df["game_date"] = pd.to_datetime(games_df["game_date"], format="%Y-%m-%d")
        games_df[num_type_cols] = games_df[num_type_cols].apply(pd.to_numeric, errors="coerce")
        merged_df = pd.merge(games_df, team_regions, on="team", how="left")
    except Exception as e:
        print(f"Error occurred: {e}")
        return
    return merged_df, num_type_cols


def merge_home_and_away_games(merged_games):
    home_games = merged_games[merged_games["home_away_NS"] == 1].add_prefix("home_")
    away_games = merged_games[merged_games["home_away_NS"] == -1].add_prefix("away_")

    combined_games = pd.merge(
        home_games,
        away_games,
        left_on="home_game_id",
        right_on="away_game_id",
        suffixes=("", "_away")
    )

    combined_games["home_win"] = (combined_games["home_team_score"] > combined_games["away_team_score"]).astype(int)
    return combined_games


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importance:\n", feature_importance)

    return model


def prep_east(east_df, team_stats):
    try:
        stats_cols = [col for col in team_stats.columns if col != "team"]
        
        east_df = pd.merge(
            east_df,
            team_stats,
            left_on="team_home",
            right_on="team",
            how="left"
        ).drop(columns=["team"]).rename(columns={col: f"home_{col}" for col in stats_cols})
        
        east_df = pd.merge(
            east_df,
            team_stats,
            left_on="team_away",
            right_on="team",
            how="left"
        ).drop(columns=["team"]).rename(columns={col: f"away_{col}" for col in stats_cols})
    except KeyError as e:
        print(f"Merge error: {e}")
        return None

    return east_df


def make_win_percent_barchart(games_df):
    games_sorted = games_df.sort_values("WINNING %", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="team_home", y="WINNING %", data=games_sorted, order=games_sorted["team_home"])
    plt.xticks(rotation=45)
    plt.title("Predicted Winning Percentages for Home Teams")
    plt.xlabel("Home Team")
    plt.ylabel("Winning Percentage")
    plt.tight_layout()
    plt.show()


def export_decision_tree(model, features):
    tree = model.estimators_[0]
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=features,
        class_names=["Loss", "Win"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("src/decision_tree")


def main():
    team_regions, games_data, east_games = load_data()
    if team_regions is None or games_data is None or east_games is None:
        return

    merged_games, num_type_columns = process_data(games_data, team_regions)
    if merged_games is None:
        return

    combined_games = merge_home_and_away_games(merged_games)

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
        "home_prev_game_dist", "away_prev_game_dist",
        "home_travel_dist", "away_travel_dist"
    ]
    X = combined_games[features]
    y = combined_games["home_win"]

    model = train(X, y)

    team_stats = merged_games.groupby("team")[num_type_columns].mean().reset_index()

    east_df = prep_east(east_games, team_stats)
    if east_df is None:
        return

    X_east = east_df[features]
    east_games["WINNING %"] = model.predict_proba(X_east)[:, 1] * 100
    east_games.to_csv("src/Data/East_Predictions.csv", index=False)

    make_win_percent_barchart(east_games)

    export_decision_tree(model, features)


if __name__ == "__main__":
    main()