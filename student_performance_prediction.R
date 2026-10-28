# ============================================================
# SML PBL PROJECT
# Student Performance Prediction Using Statistical Machine Learning
# Language: R | Environment: RStudio / Posit Cloud
# ============================================================

# 1. Generate a reproducible sample dataset
set.seed(42)
n <- 300

df <- data.frame(
  Study_Hours = runif(n, 1, 8),
  Attendance = runif(n, 50, 100),
  Previous_Score = runif(n, 30, 95),
  Assignments_Completed = sample(1:10, n, replace = TRUE),
  Sleep_Hours = runif(n, 4, 9)
)

noise <- rnorm(n, mean = 0, sd = 5)

df$Exam_Score <- pmax(0, pmin(100,
  3.0 * df$Study_Hours +
  0.15 * df$Attendance +
  0.35 * df$Previous_Score +
  1.2 * df$Assignments_Completed +
  0.8 * df$Sleep_Hours +
  noise
))

cat("FIRST FIVE STUDENT RECORDS\n")
print(head(df))

cat("\nDATASET SUMMARY\n")
print(summary(df))

# 2. Explore the data
plot(
  df$Study_Hours, df$Exam_Score,
  main = "Study Hours vs Exam Score",
  xlab = "Study Hours per Day",
  ylab = "Exam Score",
  pch = 19,
  col = "steelblue"
)
grid()

# 3. Split features and target into training and testing sets
set.seed(42)
train_index <- sample(
  seq_len(nrow(df)),
  size = floor(0.8 * nrow(df))
)

train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# 4. Train Linear Regression
linear_model <- lm(Exam_Score ~ ., data = train_data)
linear_predictions <- predict(linear_model, newdata = test_data)

# 5. Train Decision Tree
if (!requireNamespace("rpart", quietly = TRUE)) {
  install.packages("rpart", repos = "https://cloud.r-project.org")
}
library(rpart)

tree_model <- rpart(
  Exam_Score ~ .,
  data = train_data,
  method = "anova",
  control = rpart.control(maxdepth = 5, cp = 0.01)
)
tree_predictions <- predict(tree_model, newdata = test_data)

# 6. Train Random Forest
if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages("randomForest", repos = "https://cloud.r-project.org")
}
library(randomForest)

set.seed(42)
forest_model <- randomForest(
  Exam_Score ~ .,
  data = train_data,
  ntree = 100,
  mtry = 2,
  importance = TRUE
)
forest_predictions <- predict(forest_model, newdata = test_data)

# 7. Evaluate models
evaluate_model <- function(actual, predicted) {
  mae <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  r2 <- 1 - sum((actual - predicted)^2) /
    sum((actual - mean(actual))^2)

  c(MAE = mae, RMSE = rmse, R2 = r2)
}

results <- rbind(
  "Linear Regression" = evaluate_model(test_data$Exam_Score, linear_predictions),
  "Decision Tree" = evaluate_model(test_data$Exam_Score, tree_predictions),
  "Random Forest" = evaluate_model(test_data$Exam_Score, forest_predictions)
)

cat("\nMODEL PERFORMANCE COMPARISON\n")
print(round(results, 3))

# 8. Compare MAE and RMSE
barplot(
  t(results[, c("MAE", "RMSE")]),
  beside = TRUE,
  col = c("steelblue", "orange"),
  main = "Regression Model Comparison",
  ylab = "Error (lower is better)",
  legend.text = TRUE,
  args.legend = list(x = "topleft", cex = 0.8)
)

# 9. Compare R-squared values
barplot(
  results[, "R2"],
  names.arg = rownames(results),
  col = "seagreen",
  main = "R-squared Comparison",
  ylab = "R-squared (higher is better)",
  las = 1
)

# 10. Select the model with the lowest test RMSE
best_index <- which.min(results[, "RMSE"])
best_model_name <- rownames(results)[best_index]
cat("\nBEST MODEL BY TEST RMSE:", best_model_name, "\n")

# 11. Actual vs predicted plot for the best model
prediction_list <- list(
  "Linear Regression" = linear_predictions,
  "Decision Tree" = tree_predictions,
  "Random Forest" = forest_predictions
)
best_predictions <- prediction_list[[best_model_name]]

plot(
  test_data$Exam_Score, best_predictions,
  xlab = "Actual Score",
  ylab = "Predicted Score",
  main = paste("Actual vs Predicted:", best_model_name),
  pch = 19,
  col = "steelblue",
  xlim = c(0, 100),
  ylim = c(0, 100)
)
abline(0, 1, col = "red", lty = 2, lwd = 2)
grid()

# 12. Predict a score for a sample student
new_student <- data.frame(
  Study_Hours = 4,
  Attendance = 85,
  Previous_Score = 70,
  Assignments_Completed = 8,
  Sleep_Hours = 7
)

if (best_model_name == "Linear Regression") {
  predicted_score <- predict(linear_model, newdata = new_student)
} else if (best_model_name == "Decision Tree") {
  predicted_score <- predict(tree_model, newdata = new_student)
} else {
  predicted_score <- predict(forest_model, newdata = new_student)
}

cat("\nSAMPLE STUDENT DETAILS\n")
print(new_student)
cat("\nIllustrative predicted exam score:",
    round(as.numeric(predicted_score), 2), "out of 100\n")

# 13. Save project outputs in the working directory
write.csv(df, "student_performance_dataset.csv", row.names = FALSE)

results_df <- data.frame(
  Model = rownames(results),
  MAE = results[, "MAE"],
  RMSE = results[, "RMSE"],
  R2 = results[, "R2"],
  row.names = NULL
)
write.csv(results_df, "model_comparison.csv", row.names = FALSE)

cat("\nDataset and model comparison CSV files saved successfully.\n")
