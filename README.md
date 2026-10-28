# Student Performance Prediction Using Statistical Machine Learning

## Project overview
This SML PBL project demonstrates how statistical and machine-learning regression models can estimate student exam scores from study habits and academic indicators.

## Features
- Generates a reproducible synthetic dataset of 300 student records.
- Explores the relationship between study hours and exam scores.
- Trains Linear Regression, Decision Tree Regression, and Random Forest Regression.
- Compares models using MAE, RMSE, and R-squared (R²).
- Plots model comparisons and actual versus predicted scores.
- Predicts an example student's exam score.
- Exports the dataset and model comparison table as CSV files.

## Input features
- Study hours per day
- Attendance percentage
- Previous score
- Assignments completed
- Sleep hours

## Requirements
- R
- RStudio or Posit Cloud
- `rpart` package
- `randomForest` package

The R script attempts to install `rpart` and `randomForest` if they are not already installed.

## How to run
1. Open `student_performance_prediction.R` in RStudio or Posit Cloud.
2. Run the full script using **Source**.
3. Review the printed model metrics and plots.
4. The script creates `student_performance_dataset.csv` and `model_comparison.csv` in the current working directory.

## Important note
The dataset is synthetic and intended for educational demonstration only. Results are calculated when the R script is run and may vary slightly by package version. They should not be interpreted as a validated real-world student assessment tool.
