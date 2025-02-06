# 2025 Wharton Data Science Competition
## NCAA Basketball Projections Machine Learning System

This project was built for the Wharton High School Data Science Competition. It utilises a Random FOrest Classifier to predict the winning percentages of teams in the NCAA basketball East region. 
A model is trained on the game data provided and generates predictions along with two visualisations. 

### Features
- Predicts home team winning probabilities for the specific matches, outputting them in a CSV
- Generates a bar chart to visualise the predictions
- Exports a pdf of a sample decision tree used in the Random Forest model. 
*Note: The Random Forest model uses many decision trees; the pdf generated is of one of them*

### Prerequisites
- Python 3.7+
- Required Python libraries:
    - pandas
    - scikit-learn
    - matplot lib
    - seaborn
    - graphviz (ensure graphviz is located in System PATH)

Install dependencies using:
```bash
pip install pandas scikit-learn matplotlib seaborn graphviz
```

### Data Preparation
#### Input Files
The following files, given by the Wharton competition, are used:
1. Team Region Groups.csv: Contains team-region mappings.
2. games_2022.csv: Historical game statistics from the 2022 season.
3. East Regional Games to predict.csv: Matchups to predict (East Regional games).

#### Data Processing
 - Merges team region data with game statistics
 - Processes numeric features, given by the competition, and training the model with them
- Combines home/away game data into a single dataset with feature prefixes to abide by conventions in files given by the competition

### Usage
1. Run the script:
```bash
python main.py
```

#### Outputs
- East_Predictions.csv: Contains predicted winning percentages for home teams.
- decision_tree.pdf: Visualisation of one decision tree from the Random Forest.
- A bar plot displaying sorted winning percentages (shown interactively).

### Model details
- Trained with several values given by the competition
- Random Forest Classifier with default parameters
- 80% of data was used for training; 20% was used for testing the model (80-20 train-test-split)

**Author:** Franco Novoa
**Team:** Markham Masters, Perú