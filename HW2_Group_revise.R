# import the dataset
library(dplyr)
xyz <- read.csv("C:/Users/Ruth Hsu/Desktop/MS/UMN/Course/6131 Introduction to Business Analytics in R/HW2/XYZData.csv", stringsAsFactors = TRUE)

# check whether there is any missing data: no
xyz[!complete.cases(xyz), ]

# Remove non-predictive columns (user id) and convert class
xyz <- xyz %>% select(-user_id) %>%
  mutate_at(c("adopter","male","good_country"), as.factor)

# randomly permutate rows of the entire dataset
set.seed(123)
xyz_rand <- xyz[sample(nrow(xyz)),]

# use 70% training and 30% validation/testing split by row indexes
library(caret)
train_rows <- createDataPartition(y = xyz_rand$adopter, p = 0.70, list = FALSE)
xyz_rand_train <- xyz_rand[train_rows,]
xyz_rand_testValid <- xyz_rand[-train_rows,]
# Further split the remaining data into validation and test, 50% each
testIndex <- createDataPartition(xyz_rand_testValid$adopter, p = 0.5, list = FALSE)
xyz_rand_valid <- xyz_rand_testValid[testIndex, ]
xyz_rand_test <- xyz_rand_testValid[-testIndex, ]

# SMOTE oversampling to balance class distribution
#install.packages("remotes")
#remotes::install_github("dongyuanwu/RSBID")
library(RSBID)
smote_result <- SMOTE_NC(xyz_rand_train, "adopter", k = 5) #The desired percentage of the size of majority samples that the minority samples would be reached in the new dataset. The default is 100.
nrow(smote_result[smote_result$adopter == 1,]) #check the result, 2800 ids have adopter == 1

##1st round
# Train a Decision Tree Model
library(rpart)
set.seed(123) # For reproducibility
#process categorial variable
# Building the Decision Tree
decision_tree <- rpart(adopter ~ ., data = smote_result, method = "class", control = rpart.control(maxdepth = 4))
# Plotting the Decision Tree
plot(decision_tree, uniform=TRUE, compress=TRUE, margin=0.1)# Adjust 'margin' as needed
text(decision_tree, use.n=TRUE, cex=0.8) # Adjust 'cex' to change text size
importance_values <- decision_tree$variable.importance
print(importance_values)
#Evaluate the Decision Tree
# Predict on the validation dataset
tree_predictions <- predict(decision_tree, xyz_rand_valid, type = "class")
# Confusion matrix and accuracy
confusion_matrix <- table(xyz_rand_valid$adopter, tree_predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))


# Random Forest to evaluate the importance and choose the variable
library(randomForest)
set.seed(123) # For reproducibility
rf_xyz <- randomForest(adopter ~ ., data = smote_result, importance = TRUE, ntree = 500)
# Extract and View Variable Importance
importance(rf_xyz) # matrix of importance scores for each variable, including Mean Decrease Accuracy and Mean Decrease Gini
varImpPlot(rf_xyz) # displays a plot of variable importance, showing the top variables based on their importance scores
#Evaluate the Decision Tree
# Predict on the validation dataset
forest_predictions <- predict(rf_xyz, xyz_rand_valid, type = "class")
# Confusion matrix and accuracy
confusion_matrix <- table(xyz_rand_valid$adopter, forest_predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))

#Stepwise
# Load necessary library
library(MASS)
# Assume 'data' is your dataset and 'adopter' is your binary outcome variable
# You should replace 'var1', 'var2', etc., with your actual predictor variable names.
stepwise <- glm(adopter ~ ., data = smote_result, family = binomial)
# Perform stepwise selection
stepwise_model <- step(stepwise, direction = "forward")
# View the summary of the final model
summary(stepwise_model)
#Evaluate
# Make predictions on the validation dataset
predictions <- predict(stepwise_model, newdata = xyz_rand_valid, type = "response")
# Convert probabilities to binary outcomes using a threshold (e.g., 0.5)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
# Evaluate performance
# Create a confusion matrix
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(xyz_rand_valid$adopter))
# Print the confusion matrix and associated metrics
print(confusion_matrix)
# Calculate and plot ROC curve and AUC
library(pROC)
roc_curve <- roc(xyz_rand_valid$adopter, predictions)
plot(roc_curve)
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

##2nd round
#delete variable choose in 1st round
smote_result2 <- select(smote_result, -friend_cnt, -avg_friend_male, 
                       -delta_avg_friend_male, -delta_friend_country_cnt, 
                       -delta_subscriber_friend_cnt, -delta_posts, 
                       -delta_playlists, -delta_shouts, -delta_good_country)
xyz_rand_testValid2 <- select(xyz_rand_testValid, -friend_cnt, -avg_friend_male, 
                        -delta_avg_friend_male, -delta_friend_country_cnt, 
                        -delta_subscriber_friend_cnt, -delta_posts, 
                        -delta_playlists, -delta_shouts, -delta_good_country)

# Correlation to evaluate
# Load necessary library
library(ltm)
# Select the relevant continuous variables from the dataframe
continuous_vars <- smote_result2[, c("age", "avg_friend_age", "friend_country_cnt",
                                     "subscriber_friend_cnt", "songsListened",
                                     "lovedTracks", "playlists", "delta_songsListened",
                                     "delta_lovedTracks", "tenure")]
xyz_rand_valid2 <- xyz_rand_testValid[testIndex, ]
# Calculate Pearson correlations
pearson_corr <- cor(continuous_vars, use = "complete.obs")
print(pearson_corr)
#Visualize the Correlation Matrix
library(corrplot)
# Visualize the Pearson correlation matrix using a heatmap
corrplot(pearson_corr, method = "color",
         type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 25, 
         addCoef.col = "black") # Add correlation coefficients
# Calculate Point Biserial correlations
attach(smote_result2)
pb_corr1 <- biserial.cor(age, male)
pb_corr2 <- biserial.cor(age, good_country)
pb_corr3 <- biserial.cor(avg_friend_age, male)
pb_corr4 <- biserial.cor(avg_friend_age, good_country)
pb_corr5 <- biserial.cor(friend_country_cnt, male)
pb_corr6 <- biserial.cor(friend_country_cnt, good_country)
pb_corr7 <- biserial.cor(subscriber_friend_cnt, male)
pb_corr8 <- biserial.cor(subscriber_friend_cnt, good_country)
pb_corr9 <- biserial.cor(songsListened, male)
pb_corr10 <- biserial.cor(songsListened, good_country)
pb_corr11 <- biserial.cor(lovedTracks, male)
pb_corr12 <- biserial.cor(lovedTracks, good_country)
pb_corr13 <- biserial.cor(playlists, male)
pb_corr14 <- biserial.cor(playlists, good_country)
pb_corr15 <- biserial.cor(playlists, male)
pb_corr16 <- biserial.cor(playlists, good_country)
pb_corr17 <- biserial.cor(delta_songsListened, male)
pb_corr18 <- biserial.cor(delta_songsListened, good_country)
pb_corr19 <- biserial.cor(tenure, male)
pb_corr20 <- biserial.cor(tenure, good_country)
# Create a correlation table
correlation_table <- matrix(NA, nrow = 4, ncol = 4,
                            dimnames = list(c("cont_var1", "cont_var2", "cat_var1", "cat_var2"),
                                            c("cont_var1", "cont_var2", "cat_var1", "cat_var2")))
# Fill in the table
correlation_table["cont_var1", "cont_var1"] <- 1
correlation_table["cont_var2", "cont_var2"] <- 1
correlation_table["cont_var1", "cont_var2"] <- pearson_corr[1, 2]
correlation_table["cont_var2", "cont_var1"] <- pearson_corr[1, 2]
correlation_table["cont_var1", "cat_var1"] <- pb_corr1
correlation_table["cont_var1", "cat_var2"] <- pb_corr2
correlation_table["cont_var2", "cat_var1"] <- pb_corr3
correlation_table["cont_var2", "cat_var2"] <- pb_corr4
# Print the correlation table
print(correlation_table)
# Create a matrix for point biserial correlations
pb_corr_matrix <- matrix(c(pb_corr1, pb_corr2, pb_corr3, pb_corr4, pb_corr5,
                           pb_corr6, pb_corr7, pb_corr8, pb_corr9, pb_corr10,
                           pb_corr11, pb_corr12, pb_corr13, pb_corr14, pb_corr15,
                           pb_corr16, pb_corr17, pb_corr18, pb_corr19, pb_corr20),
                         nrow = 10, byrow = TRUE)
# Assign row and column names for clarity
rownames(pb_corr_matrix) <- c("age", "avg_friend_age", "friend_country_cnt",
                              "subscriber_friend_cnt", "songsListened",
                              "lovedTracks", "playlists", "delta_songsListened",
                              "delta_lovedTracks", "tenure")
colnames(pb_corr_matrix) <- c("male", "good_country")
# Load necessary library
install.packages("pheatmap")
library(pheatmap)
# Create the heatmap
pheatmap(pb_corr_matrix, display_numbers = TRUE, 
         cluster_rows = FALSE, cluster_cols = FALSE, 
         main = "Point Biserial Correlation Heatmap")

# Random Forest to evaluate the importance and choose the variable
library(randomForest)
set.seed(123) # For reproducibility
rf_xyz2 <- randomForest(adopter ~ ., data = smote_result2, importance = TRUE, ntree = 500)
# Extract and View Variable Importance
importance(rf_xyz2) # matrix of importance scores for each variable, including Mean Decrease Accuracy and Mean Decrease Gini
varImpPlot(rf_xyz2) # displays a plot of variable importance, showing the top variables based on their importance scores
#Evaluate the Decision Tree
# Predict on the validation dataset
forest_predictions <- predict(rf_xyz2, xyz_rand_testValid2, type = "class")
# Confusion matrix and accuracy
confusion_matrix <- table(xyz_rand_testValid2$adopter, forest_predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))

#Stepwise
# Load necessary library
library(MASS)
# Assume 'data' is your dataset and 'adopter' is your binary outcome variable
# You should replace 'var1', 'var2', etc., with your actual predictor variable names.
stepwise2 <- glm(adopter ~ ., data = smote_result2, family = binomial)
# Perform stepwise selection
stepwise_model2 <- step(stepwise2, direction = "forward")
# View the summary of the final model
summary(stepwise_model2)
#Evaluate
# Make predictions on the validation dataset
predictions <- predict(stepwise_model2, newdata = xyz_rand_testValid2, type = "response")
# Convert probabilities to binary outcomes using a threshold (e.g., 0.5)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
# Evaluate performance
# Create a confusion matrix
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(xyz_rand_testValid2$adopter))
# Print the confusion matrix and associated metrics
print(confusion_matrix)
# Calculate and plot ROC curve and AUC
library(pROC)
roc_curve <- roc(xyz_rand_testValid2$adopter, predictions)
plot(roc_curve)
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

#Building the Logistic Regression Model
smote_result4 <- smote_result %>%
  dplyr::select(male, avg_friend_age, subscriber_friend_cnt, songsListened, 
                playlists, delta_lovedTracks, tenure, adopter)
xyz_rand_valid4 <- xyz_rand_valid %>%
  dplyr::select(male, avg_friend_age, subscriber_friend_cnt, songsListened, 
                playlists, delta_lovedTracks, tenure, adopter)
xyz_rand_test4 <- xyz_rand_test %>%
  dplyr::select(male, avg_friend_age, subscriber_friend_cnt, songsListened, 
                playlists, delta_lovedTracks, tenure, adopter)
regression_model <- glm(adopter ~ male + avg_friend_age + subscriber_friend_cnt
                        + songsListened + playlists + delta_lovedTracks 
                        + tenure, data = smote_result4, family = binomial)

#Validating the Logistic Regression Model
# Load necessary library for evaluation
library(ROCR)
library(caret)
# Predict probabilities on validation data
validation_probs <- predict(regression_model, newdata = xyz_rand_valid4, type = "response")
str(xyz_rand_valid4$outcome)
# ROC and AUC
pred <- prediction(validation_probs, xyz_rand_valid4$adopter)
perf <- performance(pred, "tpr", "fpr")
auc <- performance(pred, "auc")
auc_value <- auc@y.values[[1]]
# F1 Score
predicted_classes <- ifelse(validation_probs > 0.5, 1, 0)
conf_matrix <- confusionMatrix(factor(predicted_classes), xyz_rand_valid4$adopter)
f1_score <- conf_matrix$byClass["F1"]

#Final Testing of the Logistic Regression Model
# Predict probabilities on test data
test_probs <- predict(regression_model, newdata = xyz_rand_test4, type = "response")
# Threshold conversion to likely adopters
likely_adopters <- ifelse(test_probs > 0.5, 1, 0)  # Adjust threshold if needed
# Show likely adopters
xyz_rand_test4$likely_adopter <- likely_adopters
# ROC and AUC for test data
test_pred <- prediction(test_probs, xyz_rand_test4$adopter)
test_perf <- performance(test_pred, "tpr", "fpr")
test_auc <- performance(test_pred, "auc")
test_auc_value <- test_auc@y.values[[1]]
# F1 Score for test data
test_predicted_classes <- ifelse(test_probs > 0.5, 1, 0)
test_conf_matrix <- confusionMatrix(factor(test_predicted_classes), xyz_rand_test4$adopter)
test_f1_score <- test_conf_matrix$byClass["F1"]

#Predict on New Data
# Assuming logistic regression performed best and you have new data 'new_data'
new_data_probs <- predict(regression_model, newdata = new_data, type = "response")
# Threshold conversion to likely adopters
likely_adopters <- ifelse(new_data_probs > 0.5, 1, 0)  # Adjust threshold if needed
# Show likely adopters
new_data$likely_adopter <- likely_adopters