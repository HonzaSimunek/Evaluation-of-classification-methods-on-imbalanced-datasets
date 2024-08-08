##### Packages loading

library(dplyr)
library(stats)
library(caret)
library(pROC)
library(PRROC)
library(ROCR)
library(e1071)
library(smotefamily)
library(ROSE)
library(ranger)
library(stargazer)
library(cowplot)
library(car)
library(MASS)
library(broom)

rm(list = ls())


##### Data loading

### 1) Myocardial infarction complications

data <- read.table("C://Users//Jan.Simunek//Downloads//myocardial+infarction+complications//MI.data", header = FALSE, sep = ",")

### Removing index column
data <- data[ , !(names(data) %in% c("V1"))]

# Changing ? to NAs
data[data == "?"] <- NA

# Frequencies of missing values in each column
na_per_column <- colSums(is.na(data))
na_per_column

# Percentage of missing values in each column
missing_percent <- colMeans(is.na(data)) * 100

# Identification of columns with more than 5% missing values
columns_to_remove <- names(data)[missing_percent > 5]

# Columns which will be removed
columns_to_remove

# Columns removing
data <- data[, !(names(data) %in% columns_to_remove)]

# Remaining rows with NAs removing
data <- na.omit(data)

# Characters to factors conversion
data <- data.frame(lapply(data, function(x) {
  if (is.character(x)) {
    return(as.factor(x))
  } else {
    return(x)
  }
}))

# Checking the structure after conversion
str(data)


# Function to remove factors with only one level
single_level_factors <- sapply(data, function(x) {
  if (is.factor(x)) {
    length(levels(x)) == 1
  } else {
    FALSE
  }
})

# Removal of factors with only one level
data <- data[, !single_level_factors]

# Creating a model and starting aliasing
alias_info <- alias(lm(V114 ~ ., data = data))
print(alias_info)

# Aliased variables extraction
alias_vars <- rownames(alias_info$Complete)
print(alias_vars)

data$V2 <- as.integer(data$V2)

# Multicollinearity testing using VIF
vif_test <- vif(lm(V114 ~ ., data = data))
vif_test



str(data)

data$V113 <- as.factor(data$V113)
data$V114 <- as.factor(data$V114)
data$V115 <- as.factor(data$V115)
data$V116 <- as.factor(data$V116)
data$V117 <- as.factor(data$V117)
data$V118 <- as.factor(data$V118)
data$V119 <- as.factor(data$V119)
data$V120 <- as.factor(data$V120)
data$V121 <- as.factor(data$V121)
data$V122 <- as.factor(data$V122)
data$V123 <- as.factor(data$V123)
data$V124 <- as.factor(data$V124)
str(data)


table(data$V114)
unique(data$V114)



data <- data %>% mutate(across(where(is.integer), as.numeric))

data$V114 <- as.numeric(as.character(data$V114))

# Stepwise regression for variable selection
stepwise_model <- stepAIC(lm(V114 ~ ., data = data), direction = "both", trace = FALSE)

summary(stepwise_model)

# Getting selected variables using the broom package
tidy_model <- tidy(stepwise_model)
selected_variables <- unique(tidy_model$term)
selected_variables <- selected_variables[selected_variables != "(Intercept)"]

# Select original variables before encoding
# Extract original variable names from original data
original_variables <- colnames(data)

get_original_variables <- function(encoded_vars, original_data) {
  original_vars <- character()
  for (var in encoded_vars) {
    # Trying to find the original name
    base_var <- strsplit(var, ":|\\.|\\^|\\*|\\+|\\-")[[1]][1]
    if (base_var %in% colnames(original_data)) {
      original_vars <- c(original_vars, base_var)
    } else {
      # Search in all variables if it is a dummy variable
      for (orig_var in colnames(original_data)) {
        if (startsWith(var, orig_var)) {
          original_vars <- c(original_vars, orig_var)
          break
        }
      }
    }
  }
  return(unique(original_vars))
}

# Getting the original variables
selected_original_variables <- get_original_variables(selected_variables, data)
original_variables
selected_original_variables
selected_variables
not_selected_variables <- setdiff(original_variables, selected_original_variables)
not_selected_variables
selected_original_variables <- c("V10", "V27", "V42", "V49", "V100", "V107", "V113", "V115")
selected_original_variables
# Adding V114 to the selection
selected_original_variables <- c(selected_original_variables, "V114")

# Selection of original variables from the dataset
data <- data[, selected_original_variables]

data$V114 <- as.factor(data$V114)

# DATA PREPARATION
set.seed(3)
data_trainIndex <- createDataPartition(data$V114, p = .7, 
                                       list = FALSE, 
                                       times = 1)
data_Train <- data[ data_trainIndex,]
data_Test  <- data[-data_trainIndex,]

# True values
true_classes <- as.factor(data_Test$V114)
target_variable <- "V114"

table(data_Train$V114)
table(data$V114)
# Ensuring that V114 is a factor
data_Train$V114 <- as.factor(data_Train$V114)


# Identification of categorical variables
categorical_vars <- names(data)[sapply(data, function(x) is.factor(x) && nlevels(x) > 2)]

# Identification of binary variables
binary_vars <- names(data)[sapply(data, function(x) is.factor(x) && nlevels(x) == 2)]

# Identifictaion of non-categorical and non-binary variables
non_categorical_vars <- setdiff(names(data), c(categorical_vars, binary_vars))

# Identification of numeric variables
numeric_vars <- names(data)[sapply(data, is.numeric)]
dummies <- dummyVars( ~ ., data = data_Train[categorical_vars], fullRank = FALSE)
data_Train_encoded <- predict(dummies, newdata = data_Train[categorical_vars])
data_Train_encoded <- as.data.frame(data_Train_encoded)

# Conversion of binary variables to 0 and 1
data_Train_binary <- data_Train[, binary_vars]
data_Train_binary <- as.data.frame(lapply(data_Train_binary, function(x) as.numeric(as.character(x))))

# Adding back binary variables and non-categorical variables
data_Train_encoded <- cbind(data_Train_encoded, data_Train_binary, data_Train[, non_categorical_vars])
str(data_Train_encoded)

# Separating the explained variable and the explanatory variables
x_train <- data_Train_encoded[, -which(names(data_Train_encoded) == "V114")]
y_train <- data_Train_encoded$V114


# Function for tuning dup_size to achieve a ratio of 0.5 and possibly randomly removing minority class observations
adjust_smote_to_balance <- function(x_train, y_train, K = 5) {
  y_train <- as.factor(y_train)  # Conversion to a factor
  majority_class <- names(which.max(table(y_train)))
  minority_class <- names(which.min(table(y_train)))
  current_ratio <- sum(y_train == minority_class) / length(y_train)
  dup_size <- 1
  
  while (current_ratio < 0.5) {
    smote_result <- SMOTE(x_train, y_train, K = K, dup_size = dup_size)
    data_smote <- smote_result$data
    y_smote <- data_smote$class
    
    current_ratio <- sum(y_smote == minority_class) / nrow(data_smote)
    if (current_ratio >= 0.5) {
      break
    }
    dup_size <- dup_size + 1
  }
  
  if (current_ratio > 0.5) {
    # The exact number of instances for each class to make the ratio 0.5
    target_count <- sum(y_smote == majority_class)
    
    # Random undersampling of a minority class
    minority_indices <- which(y_smote == minority_class)
    sampled_minority_indices <- sample(minority_indices, target_count)
    majority_data <- data_smote[y_smote == majority_class, ]
    minority_data <- data_smote[sampled_minority_indices, ]
    
    # Combining balanced data
    balanced_data <- rbind(minority_data, majority_data)
  } else {
    balanced_data <- data_smote
  }
  
  return(balanced_data)
}

# Data balancing with SMOTE
smote_result <- SMOTE(x_train, y_train, K = 5, dup_size = 1)

# Creating a new dataframe after SMOTE
data_Train_smote_encoded <- adjust_smote_to_balance(x_train, y_train)

str(data_Train_smote_encoded)
table(data_Train_smote_encoded$class)

# Rounding function
is_between_0_and_1 <- function(col, colname) {
  if (colname == "xxx") {
    return(FALSE)
  }
  is.numeric(col) && all(col >= 0 & col <= 1, na.rm = TRUE)
}

binary_columns <- names(data_Train_smote_encoded)[sapply(names(data_Train_smote_encoded), function(colname) is_between_0_and_1(data_Train_smote_encoded[[colname]], colname))]

# Round values in binary columns to 0 or 1
data_Train_smote_encoded$V114 <- as.factor(data_Train_smote_encoded$class)
data_Train_smote_encoded$V114 <- as.numeric(as.character(data_Train_smote_encoded$V114))

data_Train_smote_encoded[binary_columns] <- lapply(data_Train_smote_encoded[binary_columns], function(x) round(x))
data_Train_smote_encoded[binary_vars] <- lapply(data_Train_smote_encoded[binary_vars], function(x) round(x))
data_Train_smote_encoded[binary_columns] <- lapply(data_Train_smote_encoded[binary_columns], function(x) as.factor(x))
data_Train_smote_encoded[binary_vars] <- lapply(data_Train_smote_encoded[binary_vars], function(x) as.factor(x))

# Checking the resulting data
str(data_Train_smote_encoded)

data_Train_smote_encoded$class <- NULL

# Checking class distribution after SMOTE
table(data_Train_smote_encoded$V114)


# Function for inverse one-hot encoding
inverse_one_hot <- function(encoded_data, original_data, categorical_vars, target_var) {
  encoded_data <- as.data.frame(encoded_data)
  
  # Temporarily removing the target variable so that it does not change during inverse one-hot encoding
  target_data <- encoded_data[[target_var]]
  encoded_data[[target_var]] <- NULL
  
  for (cat_var in categorical_vars) {
    prefixes <- grep(paste0("^", cat_var, "\\."), colnames(encoded_data), value = TRUE)
    
    if (length(prefixes) > 0) {
      original_levels <- gsub(paste0(cat_var, "\\."), "", prefixes)
      original_levels <- gsub("\\.", " ", original_levels)  # The assumption that the original data uses spaces
      
      encoded_data[[cat_var]] <- factor(apply(encoded_data[, prefixes], 1, function(x) original_levels[which.max(x)]), levels = original_levels)
      
      encoded_data <- encoded_data[, !colnames(encoded_data) %in% prefixes]
    }
  }
  
  # Conversion of all numeric variables to numeric type
  numeric_vars <- setdiff(names(original_data), c(categorical_vars, target_var))
  encoded_data <- encoded_data %>%
    mutate(across(all_of(numeric_vars), as.numeric))
  
  # Adding back the target variable
  encoded_data[[target_var]] <- target_data
  
  return(encoded_data)
}

# Using the inverse one-hot encoding function
data_Train_smote <- inverse_one_hot(data_Train_smote_encoded, data_Train, categorical_vars, "V114")

# Checking the structure of the resulting dataframe
str(data_Train_smote)


# Function to remove one category from one-hot encoded data
remove_one_category <- function(encoded_data, categorical_vars) {
  for (cat_var in categorical_vars) {
    # Modified regular expression that will search for columns starting with a categorical variable regardless of the separator
    prefixes <- grep(paste0("^", cat_var), colnames(encoded_data), value = TRUE)
    
    if (length(prefixes) > 0) {
      # Removal of the first category
      encoded_data <- encoded_data[, !colnames(encoded_data) %in% prefixes[1]]
    }
  }
  return(encoded_data)
}

# Removing one category from one-hot encoded data
data_Train_encoded[binary_vars] <- lapply(data_Train_encoded[binary_vars], function(x) x - 1)
data_Train_encoded[binary_vars] <- lapply(data_Train_encoded[binary_vars], function(x) as.factor(x))
data_Train_encoded[binary_columns] <- lapply(data_Train_encoded[binary_columns], function(x) as.factor(x))
data_Train_encoded[binary_vars] <- lapply(data_Train_encoded[binary_vars], function(x) as.factor(x))
data_Train_encoded <- remove_one_category(data_Train_encoded, categorical_vars)
str(data_Train_encoded)

data_Train_smote_encoded <- remove_one_category(data_Train_smote_encoded, categorical_vars)
str(data_Train_smote_encoded)

# Function for data type conversion and correct setting of factor levels for V114
convert_types <- function(train_data, test_data, target_var) {
  # Looping through each column in the test data
  for (col in names(test_data)) {
    # Checking if the column exists in the train data
    if (col %in% names(train_data)) {
      # Converting the train data column to the same type as the test data column
      if (is.factor(test_data[[col]])) {
        train_data[[col]] <- as.factor(train_data[[col]])
        # Setting the same levels as in the test data
        levels(train_data[[col]]) <- levels(test_data[[col]])
      } else if (is.character(test_data[[col]])) {
        train_data[[col]] <- as.character(train_data[[col]])
      } else if (is.integer(test_data[[col]])) {
        train_data[[col]] <- as.integer(train_data[[col]])
      } else if (is.numeric(test_data[[col]])) {
        train_data[[col]] <- as.numeric(train_data[[col]])
      }
    }
  }
  
  # Setting the correct levels for target variable V114
  if (is.factor(test_data[[target_var]])) {
    if (all(levels(train_data[[target_var]]) == c("1", "2"))) {
      levels(train_data[[target_var]]) <- c("0", "1")
    } else {
      levels(train_data[[target_var]]) <- levels(test_data[[target_var]])
    }
  }
  
  return(train_data)
}


data_Train[binary_vars] <- lapply(data_Train[binary_vars], function(x) as.factor(x))
data_Test[binary_vars] <- lapply(data_Test[binary_vars], function(x) as.factor(x))
data_Train_smote[binary_vars] <- lapply(data_Train_smote[binary_vars], function(x) as.numeric(x))
data_Train_smote[binary_vars] <- lapply(data_Train_smote[binary_vars], function(x) x - 1)
data_Train_smote[binary_vars] <- lapply(data_Train_smote[binary_vars], function(x) as.factor(x))

# Conversion of data types in data_Train_smote_encoded by data_Test
data_Train_smote <- convert_types(data_Train_smote, data_Test, "V114")
unique(data_Train[categorical_vars])
unique(data_Train_smote[categorical_vars])
# Structure check after conversion
str(data_Train_smote)
str(data_Train)
str(data_Test)

# Checking V114 values
table(data_Train_smote$V114)

# Assume that V114 is a factor
data_Train_smote$V114 <- as.factor(data_Train_smote$V114)

# Checking current factor levels
levels(data_Train_smote$V114)

# Remapping factor levels
levels(data_Train_smote$V114) <- c("0", "1")

# Result check
table(data_Train_smote$V114)

data_Test$V114 <- as.factor(data_Test$V114)

# UNDERSAMPLING

# Data balancing using undersampling
data_Train_under <- ovun.sample(V114 ~ ., data = data_Train, method = "under", N = min(table(data_Train$V114)) * 2, seed = 42)$data
data_Train_under$V114 <- as.factor(data_Train_under$V114)

# Checking class distribution after undersampling
table(data_Train_under$V114)

# One-hot encoding of categorical variables
dummies <- dummyVars( ~ ., data = data_Train_under[categorical_vars], fullRank = TRUE)
data_Train_under_encoded <- predict(dummies, newdata = data_Train_under[categorical_vars])
data_Train_under_encoded <- as.data.frame(data_Train_under_encoded)

# Conversion of binary variables to 0 and 1
data_Train_binary <- data_Train_under[, binary_vars]
data_Train_binary <- as.data.frame(lapply(data_Train_binary, function(x) as.numeric(as.character(x))))

# Adding back binary variables and non-categorical variables
data_Train_under_encoded <- cbind(data_Train_under_encoded, data_Train_binary, data_Train_under[, non_categorical_vars])

convert_levels <- function(df) {
  for (col in names(df)) {
    if (is.factor(df[[col]])) {
      if (all(levels(df[[col]]) %in% c("-1", "0"))) {
        levels(df[[col]]) <- ifelse(levels(df[[col]]) == "-1", "0", "1")
      } else if (all(levels(df[[col]]) %in% c("0", "1"))) {
      }
    }
  }
  return(df)
}
str(data_Train_encoded)
# Applying the feature to the data
data_Train_encoded <- convert_levels(data_Train_encoded)
str(data_Train_encoded)


data_Train_under_encoded <- convert_types(data_Train_under_encoded, data_Train_encoded, "V114")
str(data_Train_under_encoded)

# OVERSAMPLING

# Data balancing using oversampling
data_Train_over <- ovun.sample(V114 ~ ., data = data_Train, method = "over", N = max(table(data_Train$V114)) * 2, seed = 42)$data
data_Train_over$V114 <- as.factor(data_Train_over$V114)

# Checking class distribution after oversampling
table(data_Train_over$V114)

# One-hot encoding of categorical variables
dummies <- dummyVars( ~ ., data = data_Train_over[categorical_vars], fullRank = TRUE)
data_Train_over_encoded <- predict(dummies, newdata = data_Train_over[categorical_vars])
data_Train_over_encoded <- as.data.frame(data_Train_over_encoded)

# Conversion of binary variables to 0 and 1
data_Train_binary <- data_Train_over[, binary_vars]
data_Train_binary <- as.data.frame(lapply(data_Train_binary, function(x) as.numeric(as.character(x))))

# Adding back binary variables and non-categorical variables
data_Train_over_encoded <- cbind(data_Train_over_encoded, data_Train_binary, data_Train_over[, non_categorical_vars])
data_Train_over_encoded <- convert_types(data_Train_over_encoded, data_Train_encoded, "V114")
str(data_Train_over_encoded)
# TEST DATA ENCODING


# One-hot encoding of categorical variables
dummies <- dummyVars(~ ., data = data_Test[categorical_vars], fullRank = TRUE)
data_Test_encoded <- predict(dummies, newdata = data_Test[categorical_vars])
data_Test_encoded <- as.data.frame(data_Test_encoded)

data_Test_binary <- data_Test[, binary_vars]
data_Test_binary <- as.data.frame(lapply(data_Test_binary, function(x) as.numeric(as.character(x))))

# Adding back binary variables and non-categorical variables
data_Test_encoded <- cbind(data_Test_encoded, data_Test_binary, data_Test[, non_categorical_vars])


str(data_Test_encoded)
data_Test_encoded <- convert_types(data_Test_encoded, data_Train_encoded, "V114")
str(data_Test_encoded)

data_Train_smote <- convert_types(data_Train_smote, data_Train, "V114")
str(data_Train_smote)

str(data)
str(data_Test)
str(data_Test_encoded)
str(data_Train)
str(data_Train_encoded)
str(data_Train_smote)
str(data_Train_smote_encoded)
str(data_Train_under)
str(data_Train_under_encoded)
str(data_Train_over)
str(data_Train_over_encoded)















# Threshold range setting
thresholds <- seq(0.01, 0.99, by = 0.01)

# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### LOGIT - imbalanced

start_time <- Sys.time()
logit <- glm(V114 ~ ., data = data_Train, family = binomial(link = "logit"))
end_time <- Sys.time()
duration <- end_time - start_time
duration
summary(logit)

# Prediction on the test dataset
logit_predictions <- predict(logit, newdata = data_Test, type = "response")

# Looping across the thresholds
for (threshold in thresholds) {
  # Prediction with the current threshold
  logit_predicted_classes <- ifelse(logit_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  logit_confusionmatrix <- confusionMatrix(factor(logit_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  logit_precision <- logit_confusionmatrix$byClass["Pos Pred Value"]
  logit_recall <- logit_confusionmatrix$byClass["Sensitivity"]
  logit_f1 <- 2 * (logit_precision * logit_recall) / (logit_precision + logit_recall)
  
  # Saving the best results
  if (!is.na(logit_f1) && logit_f1 > best_f1) {
    best_f1 <- logit_f1
    best_threshold <- threshold
    best_precision <- logit_precision
    best_recall <- logit_recall
    best_confusionmatrix <- logit_confusionmatrix
  }
}

# Overwriting variables to best values
logit_threshold <- best_threshold
logit_f1 <- best_f1
logit_precision <- best_precision
logit_recall <- best_recall
logit_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", logit_threshold, "\n")
cat("Precision: ", logit_precision, "\n")
cat("Recall: ", logit_recall, "\n")
cat("F1 Score: ", logit_f1, "\n")

# AUC-ROC
logit_roc_curve <- roc(true_classes, logit_predictions)
logit_auc_roc <- auc(logit_roc_curve)
cat("AUC-ROC: ", logit_auc_roc, "\n")

# Data for ggplot
logit_roc_data <- data.frame(
  fpr = rev(1 - logit_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(logit_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "Logit",
  data_type = "Imbalanced"
)

# ROC curve plotting using ggplot2
ggplot(logit_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(logit_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_logit_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
logit_pr_curve <- pr.curve(scores.class0 = logit_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
logit_auc_pr <- logit_pr_curve$auc.integral
cat("AUC-PR: ", logit_auc_pr, "\n")

# Data for ggplot
logit_pr_data <- data.frame(
  recall = logit_pr_curve$curve[, 1],  # Recall
  precision = logit_pr_curve$curve[, 2],  # Precision
  method = "Logit",
  data_type = "Imbalanced"
)

# PR curve plotting using ggplot2
ggplot(logit_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(logit_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_logit_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)






# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### LOGIT - smote

start_time <- Sys.time()
logit_smote <- glm(V114 ~ ., data = data_Train_smote, family = binomial(link = "logit"))
end_time <- Sys.time()
duration <- end_time - start_time
duration
summary(logit_smote)

# Prediction on the test dataset
logit_smote_predictions <- predict(logit_smote, newdata = data_Test, type = "response")

# Loop # Loop # Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  logit_smote_predicted_classes <- ifelse(logit_smote_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  logit_smote_confusionmatrix <- confusionMatrix(factor(logit_smote_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  logit_smote_precision <- logit_smote_confusionmatrix$byClass["Pos Pred Value"]
  logit_smote_recall <- logit_smote_confusionmatrix$byClass["Sensitivity"]
  logit_smote_f1 <- 2 * (logit_smote_precision * logit_smote_recall) / (logit_smote_precision + logit_smote_recall)
  
  # Saving the best results
  if (!is.na(logit_smote_f1) && logit_smote_f1 > best_f1) {
    best_f1 <- logit_smote_f1
    best_threshold <- threshold
    best_precision <- logit_smote_precision
    best_recall <- logit_smote_recall
    best_confusionmatrix <- logit_smote_confusionmatrix
  }
}

# Overwriting variables to best values
logit_smote_threshold <- best_threshold
logit_smote_f1 <- best_f1
logit_smote_precision <- best_precision
logit_smote_recall <- best_recall
logit_smote_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", logit_smote_threshold, "\n")
cat("Precision: ", logit_smote_precision, "\n")
cat("Recall: ", logit_smote_recall, "\n")
cat("F1 Score: ", logit_smote_f1, "\n")

# AUC-ROC
logit_smote_roc_curve <- roc(true_classes, logit_smote_predictions)
logit_smote_auc_roc <- auc(logit_smote_roc_curve)
cat("AUC-ROC: ", logit_smote_auc_roc, "\n")

# Data for ggplot
logit_smote_roc_data <- data.frame(
  fpr = rev(1 - logit_smote_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(logit_smote_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "Logit",
  data_type = "SMOTE"
)

# ROC curve plotting using ggplot2
ggplot(logit_smote_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(logit_smote_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_logit_smote_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
logit_smote_pr_curve <- pr.curve(scores.class0 = logit_smote_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
logit_smote_auc_pr <- logit_smote_pr_curve$auc.integral
cat("AUC-PR: ", logit_smote_auc_pr, "\n")

# Data for ggplot
logit_smote_pr_data <- data.frame(
  recall = logit_smote_pr_curve$curve[, 1],  # Recall
  precision = logit_smote_pr_curve$curve[, 2],  # Precision
  method = "Logit",
  data_type = "SMOTE"
)

# PR curve plotting using ggplot2
ggplot(logit_smote_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(logit_smote_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_logit_smote_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)






data_Train_under <- data_Train_under %>%
  mutate(across(!all_of(target_variable), as.numeric))

data_Test_under <- data_Test %>%
  mutate(across(!all_of(target_variable), as.numeric))

# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### LOGIT - undersampling

start_time <- Sys.time()
logit_under <- glm(V114 ~ ., data = data_Train_under, family = binomial(link = "logit"))
end_time <- Sys.time()
duration <- end_time - start_time
duration
summary(logit_under)

# Prediction on the test dataset
logit_under_predictions <- predict(logit_under, newdata = data_Test_under, type = "response")

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  logit_under_predicted_classes <- ifelse(logit_under_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  logit_under_confusionmatrix <- confusionMatrix(factor(logit_under_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  logit_under_precision <- logit_under_confusionmatrix$byClass["Pos Pred Value"]
  logit_under_recall <- logit_under_confusionmatrix$byClass["Sensitivity"]
  logit_under_f1 <- 2 * (logit_under_precision * logit_under_recall) / (logit_under_precision + logit_under_recall)
  
  # Saving the best results
  if (!is.na(logit_under_f1) && logit_under_f1 > best_f1) {
    best_f1 <- logit_under_f1
    best_threshold <- threshold
    best_precision <- logit_under_precision
    best_recall <- logit_under_recall
    best_confusionmatrix <- logit_under_confusionmatrix
  }
}

# Overwriting variables to best values
logit_under_threshold <- best_threshold
logit_under_f1 <- best_f1
logit_under_precision <- best_precision
logit_under_recall <- best_recall
logit_under_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", logit_under_threshold, "\n")
cat("Precision: ", logit_under_precision, "\n")
cat("Recall: ", logit_under_recall, "\n")
cat("F1 Score: ", logit_under_f1, "\n")

# AUC-ROC
logit_under_roc_curve <- roc(true_classes, logit_under_predictions)
logit_under_auc_roc <- auc(logit_under_roc_curve)
cat("AUC-ROC: ", logit_under_auc_roc, "\n")

# Data for ggplot
logit_under_roc_data <- data.frame(
  fpr = rev(1 - logit_under_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(logit_under_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "Logit",
  data_type = "Undersampling"
)

# ROC curve plotting using ggplot2
ggplot(logit_under_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(logit_under_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_logit_under_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
logit_under_pr_curve <- pr.curve(scores.class0 = logit_under_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
logit_under_auc_pr <- logit_under_pr_curve$auc.integral
cat("AUC-PR: ", logit_under_auc_pr, "\n")

# Data for ggplot
logit_under_pr_data <- data.frame(
  recall = logit_under_pr_curve$curve[, 1],  # Recall
  precision = logit_under_pr_curve$curve[, 2],  # Precision
  method = "Logit",
  data_type = "Undersampling"
)

# PR curve plotting using ggplot2
ggplot(logit_under_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(logit_under_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_logit_under_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)






# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### LOGIT - oversampling

start_time <- Sys.time()
logit_over <- glm(V114 ~ ., data = data_Train_over, family = binomial(link = "logit"))
end_time <- Sys.time()
duration <- end_time - start_time
duration
summary(logit_over)

# Prediction on the test dataset
logit_over_predictions <- predict(logit_over, newdata = data_Test, type = "response")

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  logit_over_predicted_classes <- ifelse(logit_over_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  logit_over_confusionmatrix <- confusionMatrix(factor(logit_over_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  logit_over_precision <- logit_over_confusionmatrix$byClass["Pos Pred Value"]
  logit_over_recall <- logit_over_confusionmatrix$byClass["Sensitivity"]
  logit_over_f1 <- 2 * (logit_over_precision * logit_over_recall) / (logit_over_precision + logit_over_recall)
  
  # Saving the best results
  if (!is.na(logit_over_f1) && logit_over_f1 > best_f1) {
    best_f1 <- logit_over_f1
    best_threshold <- threshold
    best_precision <- logit_over_precision
    best_recall <- logit_over_recall
    best_confusionmatrix <- logit_over_confusionmatrix
  }
}

# Overwriting variables to best values
logit_over_threshold <- best_threshold
logit_over_f1 <- best_f1
logit_over_precision <- best_precision
logit_over_recall <- best_recall
logit_over_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", logit_over_threshold, "\n")
cat("Precision: ", logit_over_precision, "\n")
cat("Recall: ", logit_over_recall, "\n")
cat("F1 Score: ", logit_over_f1, "\n")

# AUC-ROC
logit_over_roc_curve <- roc(true_classes, logit_over_predictions)
logit_over_auc_roc <- auc(logit_over_roc_curve)
cat("AUC-ROC: ", logit_over_auc_roc, "\n")

# Data for ggplot
logit_over_roc_data <- data.frame(
  fpr = rev(1 - logit_under_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(logit_under_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "Logit",
  data_type = "Oversampling"
)

# ROC curve plotting using ggplot2
ggplot(logit_over_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(logit_over_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_logit_over_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
logit_over_pr_curve <- pr.curve(scores.class0 = logit_over_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
logit_over_auc_pr <- logit_over_pr_curve$auc.integral
cat("AUC-PR: ", logit_over_auc_pr, "\n")

# Data for ggplot
logit_over_pr_data <- data.frame(
  recall = logit_over_pr_curve$curve[, 1],  # Recall
  precision = logit_over_pr_curve$curve[, 2],  # Precision
  method = "Logit",
  data_type = "Oversampling"
)

# PR curve plotting using ggplot2
ggplot(logit_over_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(logit_over_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_logit_over_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

























# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### RANDOM FOREST imbalanced #####
start_time <- Sys.time()
randomforest <- ranger(V114 ~ ., data = data_Train, num.trees = 100, probability = TRUE)
end_time <- Sys.time()
duration <- end_time - start_time
cat("Doba trénování RF na imbalanced datech: ", duration, "\n")

# Prediction on the test dataset
rf_predictions <- predict(randomforest, data = data_Test)$predictions[, 2]

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  rf_predicted_classes <- ifelse(rf_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  rf_confusionmatrix <- confusionMatrix(factor(rf_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  rf_precision <- rf_confusionmatrix$byClass["Pos Pred Value"]
  rf_recall <- rf_confusionmatrix$byClass["Sensitivity"]
  rf_f1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)
  
  # Saving the best results
  if (!is.na(rf_f1) && rf_f1 > best_f1) {
    best_f1 <- rf_f1
    best_threshold <- threshold
    best_precision <- rf_precision
    best_recall <- rf_recall
    best_confusionmatrix <- rf_confusionmatrix
  }
}

# Overwriting variables to best values
rf_threshold <- best_threshold
rf_f1 <- best_f1
rf_precision <- best_precision
rf_recall <- best_recall
rf_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", rf_threshold, "\n")
cat("Precision: ", rf_precision, "\n")
cat("Recall: ", rf_recall, "\n")
cat("F1 Score: ", rf_f1, "\n")

# AUC-ROC
rf_roc_curve <- roc(true_classes, rf_predictions)
rf_auc_roc <- auc(rf_roc_curve)
cat("Random Forest AUC-ROC: ", rf_auc_roc, "\n")

# Data for ggplot
rf_roc_data <- data.frame(
  fpr = rev(1 - rf_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(rf_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "Random Forest",
  data_type = "Imbalanced"
)

# ROC curve plotting using ggplot2
ggplot(rf_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(rf_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_rf_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
rf_pr_curve <- pr.curve(scores.class0 = rf_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
rf_auc_pr <- rf_pr_curve$auc.integral
cat("Random Forest AUC-PR: ", rf_auc_pr, "\n")

# Data for ggplot
rf_pr_data <- data.frame(
  recall = rf_pr_curve$curve[, 1],  # Recall
  precision = rf_pr_curve$curve[, 2],  # Precision
  method = "Random Forest",
  data_type = "Imbalanced"
)

# PR curve plotting using ggplot2
ggplot(rf_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(rf_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_rf_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)






# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### RANDOM FOREST smote #####
start_time <- Sys.time()
randomforest_smote <- ranger(V114 ~ ., data = data_Train_smote, num.trees = 100, probability = TRUE)
end_time <- Sys.time()
duration <- end_time - start_time
cat("Doba trénování RF na SMOTE datech: ", duration, "\n")

# Prediction on the test dataset
rf_smote_predictions <- predict(randomforest_smote, data = data_Test)$predictions[, 2]

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  rf_smote_predicted_classes <- ifelse(rf_smote_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  rf_smote_confusionmatrix <- confusionMatrix(factor(rf_smote_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  rf_smote_precision <- rf_smote_confusionmatrix$byClass["Pos Pred Value"]
  rf_smote_recall <- rf_smote_confusionmatrix$byClass["Sensitivity"]
  rf_smote_f1 <- 2 * (rf_smote_precision * rf_smote_recall) / (rf_smote_precision + rf_smote_recall)
  
  # Saving the best results
  if (!is.na(rf_smote_f1) && rf_smote_f1 > best_f1) {
    best_f1 <- rf_smote_f1
    best_threshold <- threshold
    best_precision <- rf_smote_precision
    best_recall <- rf_smote_recall
    best_confusionmatrix <- rf_smote_confusionmatrix
  }
}

# Overwriting variables to best values
rf_smote_threshold <- best_threshold
rf_smote_f1 <- best_f1
rf_smote_precision <- best_precision
rf_smote_recall <- best_recall
rf_smote_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", rf_smote_threshold, "\n")
cat("Precision: ", rf_smote_precision, "\n")
cat("Recall: ", rf_smote_recall, "\n")
cat("F1 Score: ", rf_smote_f1, "\n")

# AUC-ROC
rf_smote_roc_curve <- roc(true_classes, rf_smote_predictions)
rf_smote_auc_roc <- auc(rf_smote_roc_curve)
cat("Random Forest AUC-ROC: ", rf_smote_auc_roc, "\n")

# Data for ggplot
rf_smote_roc_data <- data.frame(
  fpr = rev(1 - rf_smote_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(rf_smote_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "Random Forest",
  data_type = "SMOTE"
)

# ROC curve plotting using ggplot2
ggplot(rf_smote_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(rf_smote_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_rf_smote_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
rf_smote_pr_curve <- pr.curve(scores.class0 = rf_smote_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
rf_smote_auc_pr <- rf_smote_pr_curve$auc.integral
cat("Random Forest AUC-PR: ", rf_smote_auc_pr, "\n")

# Data for ggplot
rf_smote_pr_data <- data.frame(
  recall = rf_smote_pr_curve$curve[, 1],  # Recall
  precision = rf_smote_pr_curve$curve[, 2],  # Precision
  method = "Random Forest",
  data_type = "SMOTE"
)

# PR curve plotting using ggplot2
ggplot(rf_smote_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(rf_smote_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_rf_smote_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)





# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### RANDOM FOREST under #####
start_time <- Sys.time()
randomforest_under <- ranger(V114 ~ ., data = data_Train_under, num.trees = 100, probability = TRUE)
end_time <- Sys.time()
duration <- end_time - start_time
cat("Doba trénování RF na UNDER datech: ", duration, "\n")

# Prediction on the test dataset
rf_under_predictions <- predict(randomforest_under, data = data_Test)$predictions[, 2]

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  rf_under_predicted_classes <- ifelse(rf_under_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  rf_under_confusionmatrix <- confusionMatrix(factor(rf_under_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  rf_under_precision <- rf_under_confusionmatrix$byClass["Pos Pred Value"]
  rf_under_recall <- rf_under_confusionmatrix$byClass["Sensitivity"]
  rf_under_f1 <- 2 * (rf_under_precision * rf_under_recall) / (rf_under_precision + rf_under_recall)
  
  # Saving the best results
  if (!is.na(rf_under_f1) && rf_under_f1 > best_f1) {
    best_f1 <- rf_under_f1
    best_threshold <- threshold
    best_precision <- rf_under_precision
    best_recall <- rf_under_recall
    best_confusionmatrix <- rf_under_confusionmatrix
  }
}

# Overwriting variables to best values
rf_under_threshold <- best_threshold
rf_under_f1 <- best_f1
rf_under_precision <- best_precision
rf_under_recall <- best_recall
rf_under_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", rf_under_threshold, "\n")
cat("Precision: ", rf_under_precision, "\n")
cat("Recall: ", rf_under_recall, "\n")
cat("F1 Score: ", rf_under_f1, "\n")

# AUC-ROC
rf_under_roc_curve <- roc(true_classes, rf_under_predictions)
rf_under_auc_roc <- auc(rf_under_roc_curve)
cat("Random Forest AUC-ROC: ", rf_under_auc_roc, "\n")

# Data for ggplot
rf_under_roc_data <- data.frame(
  fpr = rev(1 - rf_under_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(rf_under_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "Random Forest",
  data_type = "Undersampling"
)

# ROC curve plotting using ggplot2
ggplot(rf_under_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(rf_under_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_rf_under_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
rf_under_pr_curve <- pr.curve(scores.class0 = rf_under_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
rf_under_auc_pr <- rf_under_pr_curve$auc.integral
cat("Random Forest AUC-PR: ", rf_under_auc_pr, "\n")

# Data for ggplot
rf_under_pr_data <- data.frame(
  recall = rf_under_pr_curve$curve[, 1],  # Recall
  precision = rf_under_pr_curve$curve[, 2],  # Precision
  method = "Random Forest",
  data_type = "Undersampling"
)

# PR curve plotting using ggplot2
ggplot(rf_under_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(rf_under_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_rf_under_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)







# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

##### RANDOM FOREST over #####
start_time <- Sys.time()
randomforest_over <- ranger(V114 ~ ., data = data_Train_over, num.trees = 100, probability = TRUE)
end_time <- Sys.time()
duration <- end_time - start_time
cat("Doba trénování RF na OVER datech: ", duration, "\n")

# Prediction on the test dataset
rf_over_predictions <- predict(randomforest_over, data = data_Test)$predictions[, 2]
# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  rf_over_predicted_classes <- ifelse(rf_over_predictions > threshold, 1, 0)
  
  # Creating the confusion matrix
  rf_over_confusionmatrix <- confusionMatrix(factor(rf_over_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  rf_over_precision <- rf_over_confusionmatrix$byClass["Pos Pred Value"]
  rf_over_recall <- rf_over_confusionmatrix$byClass["Sensitivity"]
  rf_over_f1 <- 2 * (rf_over_precision * rf_over_recall) / (rf_over_precision + rf_over_recall)
  
  # Saving the best results
  if (!is.na(rf_over_f1) && rf_over_f1 > best_f1) {
    best_f1 <- rf_over_f1
    best_threshold <- threshold
    best_precision <- rf_over_precision
    best_recall <- rf_over_recall
    best_confusionmatrix <- rf_over_confusionmatrix
  }
}

# Overwriting variables to best values
rf_over_threshold <- best_threshold
rf_over_f1 <- best_f1
rf_over_precision <- best_precision
rf_over_recall <- best_recall
rf_over_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", rf_over_threshold, "\n")
cat("Precision: ", rf_over_precision, "\n")
cat("Recall: ", rf_over_recall, "\n")
cat("F1 Score: ", rf_over_f1, "\n")

# AUC-ROC
rf_over_roc_curve <- roc(true_classes, rf_over_predictions)
rf_over_auc_roc <- auc(rf_over_roc_curve)
cat("Random Forest AUC-ROC: ", rf_over_auc_roc, "\n")

# Data for ggplot
rf_over_roc_data <- data.frame(
  fpr = rev(1 - rf_over_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(rf_over_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "Random Forest",
  data_type = "Oversampling"
)

# ROC curve plotting using ggplot2
ggplot(rf_over_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(rf_over_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_rf_over_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
rf_over_pr_curve <- pr.curve(scores.class0 = rf_over_predictions, weights.class0 = as.numeric(as.character(true_classes)), curve = TRUE)
rf_over_auc_pr <- rf_over_pr_curve$auc.integral
cat("Random Forest AUC-PR: ", rf_over_auc_pr, "\n")

# Data for ggplot
rf_over_pr_data <- data.frame(
  recall = rf_over_pr_curve$curve[, 1],  # Recall
  precision = rf_over_pr_curve$curve[, 2],  # Precision
  method = "Random Forest",
  data_type = "Oversampling"
)

# PR curve plotting using ggplot2
ggplot(rf_over_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(rf_over_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_rf_over_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)
































# HDDT
# functions to create and use Hellinger distance decision tree (HDDT)
# written by: Kaustubh Patil - MIT Neuroecon lab (C) 2015

# DISCLAIMER
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# LICENSE
# CREATIVE COMMONS Attribution-NonCommercial 2.5 Generic (CC BY-NC 2.5)
# https://creativecommons.org/licenses/by-nc/2.5/

# References:
# Hellinger distance decision trees are robust and skew-insensitive, 
# ... David A. Cieslak, T. Ryan Hoens, Nitesh V. Chawla and W. Philip Kegelmeyer, Data Min Knowl Disc 2011
# https://www3.nd.edu/~dial/papers/DMKD11.pdf
# Learning Decision Trees for Unbalanced Data, David A. Cieslak and Nitesh V. Chawla, ECML 2008
# https://www3.nd.edu/~dial/papers/ECML08.pdf

# build a Hellinger distance decision tree
# it is a recursive function that calls itself with subsets
# of training data that matches the decision criterion
# using a list to create the tree structure
#
# Input
# X (matrix/data frame): training data, features/independent variables
#                       The columns of X must be either numeric or factor
# y (vector)           : training data, labels/dependent variable
# C (integer)          : minimum size of the training set at a node to attempt a split
# labels (vector)      : allowed labels [optional]
#
# Value
# node (list)          : the root node of the deicison tree

# Several changes have been made to following functions written by Kaustubh Patil
# for example, returning probabilities instead of predicted classes

HDDT <- function(X, y, C, labels=unique(y)) {
  
  if(is.null(labels) || length(labels)==0) labels <- unique(y)  
  
  node <- list() # when called for first time, this will be the root
  node$C <- C
  node$labels <- labels
  
  if(length(unique(y))==1 || length(y) < C) {
    # calculate counts and frequencies
    # use Laplace smoothing, by adding 1 to count of each label
    y <- c(y, labels)
    node$count <- sort(table(y), decreasing=TRUE)
    node$freq  <- node$count/sum(node$count)
    # get the label of this leaf node
    node$label <- as.integer(names(node$count)[1])
    return(node)
  }
  else { # recursion
    # get Hellinger distance and their max
    # use for loop insread of apply as it will convert data.frame to a matrix and mess up column classes
    # e.g. factor will get coerced into character
    HD <- list()
    for(i in 1:ncol(X)) HD[[i]] <- HDDT_dist(X[,i],y=y,labels=labels)    
    hd <- sapply(HD, function(x) {return(x$d)})
    i  <- which(hd==max(hd))[1] # just taking the first 
    
    # save node attributes
    node$i    <- i
    node$v    <- HD[[i]]$v
    node$type <- HD[[i]]$type
    node$d    <- HD[[i]]$d
    
    if(node$type=="factor") {
      j <- X[,i]==node$v
      node$childLeft  <- HDDT(X[j,], y[j], C, labels)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels)
    }
    else if(node$type=="numeric") {
      j <- X[,i]<=node$v
      node$childLeft  <- HDDT(X[j,], y[j], C, labels)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels)      
    }
  }
  
  return(node) # returns root node
}

# given the root node as returned by the HDDT function and
# new data X return predictions
#
# Input
# root (list)           : root node as returned by the function HDDT
# X (matrix/data frame) : new data, features/independent variables
#
# Value
# y (integer vector)    : predicted labels for X
# modified HDDT_predict function to ensure correct return of frequencies
HDDT_predict <- function(root, X, return_prob = FALSE) {
  y <- rep(NA, nrow(X))
  probs <- rep(NA, nrow(X)) # added variable for probabilities
  for (i in 1:nrow(X)) {
    node <- root
    while (!is.null(node$v)) {
      if (node$type == "factor") {
        if (X[i, node$i] == node$v) node <- node$childLeft
        else node <- node$childRight
      } else if (node$type == "numeric") {
        if (X[i, node$i] <= node$v) node <- node$childLeft
        else node <- node$childRight
      } else stop("unknown node type: ", node$type)
    }
    stopifnot(!is.null(node$label))
    y[i] <- node$label
    if (return_prob) {
      if ("1" %in% names(node$freq)) {
        probs[i] <- node$freq[which(names(node$freq) == "1")]
      } else {
        probs[i] <- 0 # if class "1" is not in the frequencies, the probability is 0
      }
    }
  }
  
  if (return_prob) {
    return(probs) # returning probabilities, if required
  } else {
    return(y) # returning a binary prediction
  }
}


# given a feature vector calculate Hellinger distance
# it takes care of both discrete and continuous attributes
# also returns the "value" of the feature that is used as decision criterion
# and the "type" pf the feature which is either factor as numeric
# ONLY WORKS WITH BINARY LABELS
HDDT_dist <- function(f, y, labels=unique(y)) {  
  i1 <- y==labels[1]
  i0 <- y==labels[2]
  T1 <- sum(i1)
  T0 <- sum(i0)
  val <- NA
  hellinger <- -1
  
  cl <- class(f)  
  if(cl=="factor") {    
    for(v in levels(f)) {
      Tfv1 <- sum(i1 & f==v)
      Tfv0 <- sum(i0 & f==v)
      
      Tfw1 <- T1 - Tfv1
      Tfw0 <- T0 - Tfv0
      cur_value <- ( sqrt(Tfv1 / T1) - sqrt(Tfv0 / T0) )^2 + ( sqrt(Tfw1 / T1) - sqrt(Tfw0 / T0) )^2
      
      if(cur_value > hellinger) {
        hellinger <- cur_value
        val <- v
      }
    }
  }
  else if(cl=="numeric") {
    fs <- sort(unique(f))
    for(v in fs) {
      Tfv1 <- sum(i1 & f<=v)
      Tfv0 <- sum(i0 & f<=v)
      
      Tfw1 <- T1 - Tfv1
      Tfw0 <- T0 - Tfv0
      cur_value <- ( sqrt(Tfv1 / T1) - sqrt(Tfv0 / T0) )^2 + ( sqrt(Tfw1 / T1) - sqrt(Tfw0 / T0) )^2
      
      if(cur_value > hellinger) {
        hellinger <- cur_value
        val <- v
      }
    }
  }
  else stop("unknown class: ", cl)
  
  return(list(d=sqrt(hellinger), v=val, type=cl))
}


HDDT <- function(X, y, C, labels=unique(y), max_depth = 1000000, current_depth = 1) {
  if(is.null(labels) || length(labels)==0) labels <- unique(y)  
  
  node <- list()
  node$C <- C
  node$labels <- labels
  
  print(paste("Current depth:", current_depth, "Number of samples:", length(y)))
  
  if(length(unique(y)) == 1 || length(y) < C || current_depth > max_depth) {
    y <- c(y, labels)
    node$count <- sort(table(y), decreasing=TRUE)
    node$freq  <- node$count/sum(node$count)
    node$label <- as.integer(names(node$count)[1])
    print(paste("Creating leaf node with label:", node$label))
    return(node)
  } else {
    HD <- list()
    for(i in 1:ncol(X)) {
      HD[[i]] <- HDDT_dist(X[,i], y=y, labels=labels)
      print(paste("Calculated Hellinger distance for column", i, "value:", HD[[i]]$v, "distance:", HD[[i]]$d))
    }
    hd <- sapply(HD, function(x) {return(x$d)})
    max_hd <- max(hd)
    
    if(max_hd == 0) {
      print("Max Hellinger distance is 0, creating leaf node")
      y <- c(y, labels)
      node$count <- sort(table(y), decreasing=TRUE)
      node$freq  <- node$count/sum(node$count)
      node$label <- as.integer(names(node$count)[1])
      return(node)
    }
    
    i  <- which(hd == max_hd)[1]
    
    node$i    <- i
    node$v    <- HD[[i]]$v
    node$type <- HD[[i]]$type
    node$d    <- HD[[i]]$d
    
    print(paste("Splitting on column:", i, "with value:", node$v, "of type:", node$type, "Hellinger distance:", node$d))
    
    if(node$type == "factor") {
      j <- X[,i] == node$v
      print(paste("Factor split: left", sum(j), "right", sum(!j)))
      node$childLeft  <- HDDT(X[j,], y[j], C, labels, max_depth, current_depth + 1)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels, max_depth, current_depth + 1)
    }
    else if(node$type == "numeric") {
      j <- X[,i] <= node$v
      print(paste("Numeric split: left", sum(j), "right", sum(!j)))
      node$childLeft  <- HDDT(X[j,], y[j], C, labels, max_depth, current_depth + 1)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels, max_depth, current_depth + 1)
    }
  }
  
  print("Returning node")
  return(node)
}


# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

###### HDDT imbalanced

data_Train_encoded <- data_Train_encoded %>% mutate(across(where(is.integer), as.numeric))
data_Test_encoded <- data_Test_encoded %>% mutate(across(where(is.integer), as.numeric))

# Extracting features and labels
X_train <- data_Train_encoded[, -which(names(data_Train_encoded) == "V114")]
y_train <- data_Train_encoded$V114
X_test <- data_Test_encoded[, -which(names(data_Test_encoded) == "V114")]
y_test <- data_Test_encoded$V114
y_test <- as.numeric(as.character(y_test))

# Training the HDDT model
start_time <- Sys.time()
hddt_model <- HDDT(X_train, y_train, C = 5, max_depth = 1000000)
end_time <- Sys.time()
cat("Training time:", end_time - start_time, "\n")

# Making predictions on the test dataset
hddt_probabilities <- HDDT_predict(hddt_model, X_test, return_prob = TRUE)
summary(hddt_probabilities)
hist(hddt_probabilities, main = "Histogram of Predicted Probabilities", xlab = "Probability")

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  hddt_predicted_classes <- ifelse(hddt_probabilities > threshold, 1, 0)
  
  # Creating the confusion matrix
  hddt_confusionmatrix <- confusionMatrix(factor(hddt_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  hddt_precision <- hddt_confusionmatrix$byClass["Pos Pred Value"]
  hddt_recall <- hddt_confusionmatrix$byClass["Sensitivity"]
  hddt_f1 <- 2 * (hddt_precision * hddt_recall) / (hddt_precision + hddt_recall)
  
  # Saving the best results
  if (!is.na(hddt_f1) && hddt_f1 > best_f1) {
    best_f1 <- hddt_f1
    best_threshold <- threshold
    best_precision <- hddt_precision
    best_recall <- hddt_recall
    best_confusionmatrix <- hddt_confusionmatrix
  }
}

# Overwriting variables to best values
hddt_threshold <- best_threshold
hddt_f1 <- best_f1
hddt_precision <- best_precision
hddt_recall <- best_recall
hddt_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", hddt_threshold, "\n")
cat("Precision: ", hddt_precision, "\n")
cat("Recall: ", hddt_recall, "\n")
cat("F1 Score: ", hddt_f1, "\n")

# AUC-ROC
hddt_roc_curve <- roc(as.numeric(y_test), as.numeric(hddt_probabilities))
hddt_auc_roc <- auc(hddt_roc_curve)
cat("HDDT AUC-ROC: ", hddt_auc_roc, "\n")

# Data for ggplot
hddt_roc_data <- data.frame(
  fpr = rev(1 - hddt_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(hddt_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "HDDT",
  data_type = "Imbalanced"
)

# Plotting ROC curve using ggplot2
ggplot(hddt_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(hddt_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_hddt_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
y_test <- as.numeric(as.character(y_test))
hddt_pr_curve <- pr.curve(scores.class0 = hddt_probabilities, weights.class0 = as.numeric(y_test), curve = TRUE)
hddt_auc_pr <- hddt_pr_curve$auc.integral
cat("HDDT AUC-PR: ", hddt_auc_pr, "\n")

# Data for ggplot
hddt_pr_data <- data.frame(
  recall = hddt_pr_curve$curve[, 1],  # Recall
  precision = hddt_pr_curve$curve[, 2],  # Precision
  method = "HDDT",
  data_type = "Imbalanced"
)

# Plotting PR curve using ggplot2
ggplot(hddt_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(hddt_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_hddt_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)






# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

###### HDDT smote

# Extracting features and labels
X_train_smote <- data_Train_smote_encoded[, -which(names(data_Train_smote_encoded) == "V114")]
y_train_smote <- data_Train_smote_encoded$V114
X_test <- data_Test_encoded[, -which(names(data_Test_encoded) == "V114")]
y_test <- data_Test_encoded$V114
y_test <- as.numeric(as.character(y_test))


# Training the HDDT model
start_time <- Sys.time()
hddt_smote <- HDDT(X_train_smote, y_train_smote, C = 5)
end_time <- Sys.time()
cat("Training time:", end_time - start_time, "\n")

# Making predictions on the test dataset
hddt_smote_probabilities <- HDDT_predict(hddt_smote, X_test, return_prob = TRUE)
summary(hddt_smote_probabilities)
hist(hddt_smote_probabilities, main = "Histogram of Predicted Probabilities", xlab = "Probability")

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  hddt_smote_predicted_classes <- ifelse(hddt_smote_probabilities > threshold, 1, 0)
  
  # Creating the confusion matrix
  hddt_smote_confusionmatrix <- confusionMatrix(factor(hddt_smote_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  hddt_smote_precision <- hddt_smote_confusionmatrix$byClass["Pos Pred Value"]
  hddt_smote_recall <- hddt_smote_confusionmatrix$byClass["Sensitivity"]
  hddt_smote_f1 <- 2 * (hddt_smote_precision * hddt_smote_recall) / (hddt_smote_precision + hddt_smote_recall)
  
  # Saving the best results
  if (!is.na(hddt_smote_f1) && hddt_smote_f1 > best_f1) {
    best_f1 <- hddt_smote_f1
    best_threshold <- threshold
    best_precision <- hddt_smote_precision
    best_recall <- hddt_smote_recall
    best_confusionmatrix <- hddt_smote_confusionmatrix
  }
}

# Overwriting variables to best values
hddt_smote_threshold <- best_threshold
hddt_smote_f1 <- best_f1
hddt_smote_precision <- best_precision
hddt_smote_recall <- best_recall
hddt_smote_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", hddt_smote_threshold, "\n")
cat("Precision: ", hddt_smote_precision, "\n")
cat("Recall: ", hddt_smote_recall, "\n")
cat("F1 Score: ", hddt_smote_f1, "\n")

# AUC-ROC
hddt_smote_roc_curve <- roc(as.numeric(y_test), as.numeric(hddt_smote_probabilities))
hddt_smote_auc_roc <- auc(hddt_smote_roc_curve)
cat("HDDT AUC-ROC: ", hddt_smote_auc_roc, "\n")

# Data for ggplot
hddt_smote_roc_data <- data.frame(
  fpr = rev(1 - hddt_smote_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(hddt_smote_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "HDDT",
  data_type = "SMOTE"
)

# Plotting ROC curve using ggplot2
ggplot(hddt_smote_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(hddt_smote_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_hddt_smote_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
y_test <- as.numeric(as.character(y_test))
hddt_smote_pr_curve <- pr.curve(scores.class0 = hddt_smote_probabilities, weights.class0 = as.numeric(y_test), curve = TRUE)
hddt_smote_auc_pr <- hddt_smote_pr_curve$auc.integral
cat("HDDT AUC-PR: ", hddt_smote_auc_pr, "\n")

# Data for ggplot
hddt_smote_pr_data <- data.frame(
  recall = hddt_smote_pr_curve$curve[, 1],  # Recall
  precision = hddt_smote_pr_curve$curve[, 2],  # Precision
  method = "HDDT",
  data_type = "SMOTE"
)

# Plotting PR curve using ggplot2
ggplot(hddt_smote_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(hddt_smote_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_hddt_smote_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)








# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

###### HDDT under

# Extracting features and labels
X_train_under <- data_Train_under_encoded[, -which(names(data_Train_under_encoded) == "V114")]
y_train_under <- data_Train_under_encoded$V114
X_test <- data_Test_encoded[, -which(names(data_Test_encoded) == "V114")]
y_test <- data_Test_encoded$V114
y_test <- as.numeric(as.character(y_test))


# Training the HDDT model
start_time <- Sys.time()
hddt_under <- HDDT(X_train_under, y_train_under, C = 5)
end_time <- Sys.time()
cat("Training time:", end_time - start_time, "\n")

# Making predictions on the test dataset
hddt_under_probabilities <- HDDT_predict(hddt_under, X_test, return_prob = TRUE)
summary(hddt_under_probabilities)
hist(hddt_under_probabilities, main = "Histogram of Predicted Probabilities", xlab = "Probability")

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  hddt_under_predicted_classes <- ifelse(hddt_under_probabilities > threshold, 1, 0)
  
  # Creating the confusion matrix
  hddt_under_confusionmatrix <- confusionMatrix(factor(hddt_under_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  hddt_under_precision <- hddt_under_confusionmatrix$byClass["Pos Pred Value"]
  hddt_under_recall <- hddt_under_confusionmatrix$byClass["Sensitivity"]
  hddt_under_f1 <- 2 * (hddt_under_precision * hddt_under_recall) / (hddt_under_precision + hddt_under_recall)
  
  # Saving the best results
  if (!is.na(hddt_under_f1) && hddt_under_f1 > best_f1) {
    best_f1 <- hddt_under_f1
    best_threshold <- threshold
    best_precision <- hddt_under_precision
    best_recall <- hddt_under_recall
    best_confusionmatrix <- hddt_under_confusionmatrix
  }
}

# Overwriting variables to best values
hddt_under_threshold <- best_threshold
hddt_under_f1 <- best_f1
hddt_under_precision <- best_precision
hddt_under_recall <- best_recall
hddt_under_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", hddt_under_threshold, "\n")
cat("Precision: ", hddt_under_precision, "\n")
cat("Recall: ", hddt_under_recall, "\n")
cat("F1 Score: ", hddt_under_f1, "\n")

# AUC-ROC
hddt_under_roc_curve <- roc(as.numeric(y_test), as.numeric(hddt_under_probabilities))
hddt_under_auc_roc <- auc(hddt_under_roc_curve)
cat("HDDT AUC-ROC: ", hddt_under_auc_roc, "\n")

# Data for ggplot
hddt_under_roc_data <- data.frame(
  fpr = rev(1 - hddt_under_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(hddt_under_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "HDDT",
  data_type = "Undersampling"
)

# Plotting ROC curve using ggplot2
ggplot(hddt_under_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(hddt_under_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_hddt_under_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
y_test <- as.numeric(as.character(y_test))
hddt_under_pr_curve <- pr.curve(scores.class0 = hddt_under_probabilities, weights.class0 = as.numeric(y_test), curve = TRUE)
hddt_under_auc_pr <- hddt_under_pr_curve$auc.integral
cat("HDDT AUC-PR: ", hddt_under_auc_pr, "\n")

# Data for ggplot
hddt_under_pr_data <- data.frame(
  recall = hddt_under_pr_curve$curve[, 1],  # Recall
  precision = hddt_under_pr_curve$curve[, 2],  # Precision
  method = "HDDT",
  data_type = "Undersampling"
)

# Plotting PR curve using ggplot2
ggplot(hddt_under_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(hddt_under_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_hddt_under_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)














# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

###### HDDT over

# Extracting features and labels
X_train_over <- data_Train_over_encoded[, -which(names(data_Train_over_encoded) == "V114")]
y_train_over <- data_Train_over_encoded$V114
X_test <- data_Test_encoded[, -which(names(data_Test_encoded) == "V114")]
y_test <- data_Test_encoded$V114
y_test <- as.numeric(as.character(y_test))


# Training the HDDT model
start_time <- Sys.time()
hddt_over <- HDDT(X_train_over, y_train_over, C = 5)
end_time <- Sys.time()
cat("Training time:", end_time - start_time, "\n")

# Making predictions on the test dataset
hddt_over_probabilities <- HDDT_predict(hddt_over, X_test, return_prob = TRUE)
summary(hddt_over_probabilities)
hist(hddt_over_probabilities, main = "Histogram of Predicted Probabilities", xlab = "Probability")

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  hddt_over_predicted_classes <- ifelse(hddt_over_probabilities > threshold, 1, 0)
  
  # Creating the confusion matrix
  hddt_over_confusionmatrix <- confusionMatrix(factor(hddt_over_predicted_classes), factor(true_classes))
  
  # Precision, Recall and F1 Score
  hddt_over_precision <- hddt_over_confusionmatrix$byClass["Pos Pred Value"]
  hddt_over_recall <- hddt_over_confusionmatrix$byClass["Sensitivity"]
  hddt_over_f1 <- 2 * (hddt_over_precision * hddt_over_recall) / (hddt_over_precision + hddt_over_recall)
  
  # Saving the best results
  if (!is.na(hddt_over_f1) && hddt_over_f1 > best_f1) {
    best_f1 <- hddt_over_f1
    best_threshold <- threshold
    best_precision <- hddt_over_precision
    best_recall <- hddt_over_recall
    best_confusionmatrix <- hddt_over_confusionmatrix
  }
}

# Overwriting variables to best values
hddt_over_threshold <- best_threshold
hddt_over_f1 <- best_f1
hddt_over_precision <- best_precision
hddt_over_recall <- best_recall
hddt_over_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", hddt_over_threshold, "\n")
cat("Precision: ", hddt_over_precision, "\n")
cat("Recall: ", hddt_over_recall, "\n")
cat("F1 Score: ", hddt_over_f1, "\n")

# AUC-ROC
hddt_over_roc_curve <- roc(as.numeric(y_test), as.numeric(hddt_over_probabilities))
hddt_over_auc_roc <- auc(hddt_over_roc_curve)
cat("HDDT AUC-ROC: ", hddt_over_auc_roc, "\n")

# Data for ggplot
hddt_over_roc_data <- data.frame(
  fpr = rev(1 - hddt_over_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(hddt_over_roc_curve$sensitivities),      # True Positive Rate (TPR)
  method = "HDDT",
  data_type = "Oversampling"
)

# Plotting ROC curve using ggplot2
ggplot(hddt_over_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(hddt_over_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_hddt_over_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
y_test <- as.numeric(as.character(y_test))
hddt_over_pr_curve <- pr.curve(scores.class0 = hddt_over_probabilities, weights.class0 = as.numeric(y_test), curve = TRUE)
hddt_over_auc_pr <- hddt_over_pr_curve$auc.integral
cat("HDDT AUC-PR: ", hddt_over_auc_pr, "\n")

# Data for ggplot
hddt_over_pr_data <- data.frame(
  recall = hddt_over_pr_curve$curve[, 1],  # Recall
  precision = hddt_over_pr_curve$curve[, 2],  # Precision
  method = "HDDT",
  data_type = "Oversampling"
)

# Plotting PR curve using ggplot2
ggplot(hddt_over_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(hddt_over_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_hddt_over_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)





























### LOGIT QUANTILE

# Determination of the quantile for the 5% highest values
logit_quantile <- quantile(logit_predictions, 0.95)

# Prediction on the test dataset
logit_predicted_classes_quantile <- ifelse(logit_predictions <= logit_quantile, 0, 1)

# Creating the confusion matrix
logit_confusionmatrix_quantile <- confusionMatrix(factor(logit_predicted_classes_quantile), factor(true_classes))
logit_confusionmatrix_quantile

# Precision, Recall and F1 Score
logit_precision_quantile <- logit_confusionmatrix_quantile$byClass["Pos Pred Value"]
logit_recall_quantile <- logit_confusionmatrix_quantile$byClass["Sensitivity"]
logit_f1_quantile <- 2 * (logit_precision_quantile * logit_recall_quantile) / (logit_precision_quantile + logit_recall_quantile)

cat("Precision: ", logit_precision_quantile, "\n")
cat("Recall: ", logit_recall_quantile, "\n")
cat("F1 Score: ", logit_f1_quantile, "\n")















### RANDOM FOREST QUANTILE

# Determination of the quantile for the 5% highest values
rf_quantile <- quantile(rf_predictions, 0.95)

# Prediction on the test dataset
rf_predicted_classes_quantile <- ifelse(rf_predictions <= rf_quantile, 0, 1)

# Creating the confusion matrix
rf_confusionmatrix_quantile <- confusionMatrix(factor(rf_predicted_classes_quantile), factor(true_classes))
rf_confusionmatrix_quantile

# Precision, Recall and F1 Score
rf_precision_quantile <- rf_confusionmatrix_quantile$byClass["Pos Pred Value"]
rf_recall_quantile <- rf_confusionmatrix_quantile$byClass["Sensitivity"]
rf_f1_quantile <- 2 * (rf_precision_quantile * rf_recall_quantile) / (rf_precision_quantile + rf_recall_quantile)

cat("Precision: ", rf_precision_quantile, "\n")
cat("Recall: ", rf_recall_quantile, "\n")
cat("F1 Score: ", rf_f1_quantile, "\n")



### HDDT QUANTILE

# Determination of the quantile for the 5% highest values
hddt_quantile <- quantile(hddt_probabilities, 0.95)

# Prediction on the test dataset
hddt_predicted_classes_quantile <- ifelse(hddt_probabilities <= hddt_quantile, 0, 1)

# Creating the confusion matrix
hddt_confusionmatrix_quantile <- confusionMatrix(factor(hddt_predicted_classes_quantile), factor(true_classes))
hddt_confusionmatrix_quantile

# Precision, Recall and F1 Score
hddt_precision_quantile <- hddt_confusionmatrix_quantile$byClass["Pos Pred Value"]
hddt_recall_quantile <- hddt_confusionmatrix_quantile$byClass["Sensitivity"]
hddt_f1_quantile <- 2 * (hddt_precision_quantile * hddt_recall_quantile) / (hddt_precision_quantile + hddt_recall_quantile)

cat("Precision: ", hddt_precision_quantile, "\n")
cat("Recall: ", hddt_recall_quantile, "\n")
cat("F1 Score: ", hddt_f1_quantile, "\n")






















data2 <- data %>%
  mutate(across(-all_of(target_variable), as.numeric))

true_classes_trainingloop <- data$V114

# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

### LOGIT
# LOOCV
total_obs <- nrow(data2)
logit_predictions_trainingloop <- numeric(total_obs)

start_time <- Sys.time()

for (i in 1:total_obs) {
  if (i %% 100 == 0) {
    cat("Processing iteration:", i, "/", total_obs, "\n")
    cat("Elapsed time:", Sys.time() - start_time, "\n")
  }
  
  data_Train_trainingloop <- data2[-i, ]
  data_Test_trainingloop <- data2[i, , drop = FALSE]
  
  # Logistic regression training
  logit_model_trainingloop <- glm(V114 ~ ., data = data_Train_trainingloop, family = binomial)
  logit_predictions_trainingloop[i] <- predict(logit_model_trainingloop, newdata = data_Test_trainingloop, type = "response")
}


# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  logit_trainingloop_predicted_classes <- ifelse(logit_predictions_trainingloop > threshold, 1, 0)
  
  # Creating the confusion matrix
  logit_trainingloop_confusionmatrix <- confusionMatrix(factor(logit_trainingloop_predicted_classes), factor(true_classes_trainingloop))
  
  # Precision, Recall and F1 Score
  logit_trainingloop_precision <- logit_trainingloop_confusionmatrix$byClass["Pos Pred Value"]
  logit_trainingloop_recall <- logit_trainingloop_confusionmatrix$byClass["Sensitivity"]
  logit_trainingloop_f1 <- 2 * (logit_trainingloop_precision * logit_trainingloop_recall) / (logit_trainingloop_precision + logit_trainingloop_recall)
  
  # Saving the best results
  if (!is.na(logit_trainingloop_f1) && logit_trainingloop_f1 > best_f1) {
    best_f1 <- logit_trainingloop_f1
    best_threshold <- threshold
    best_precision <- logit_trainingloop_precision
    best_recall <- logit_trainingloop_recall
    best_confusionmatrix <- logit_trainingloop_confusionmatrix
  }
}

# Overwriting variables to best values
logit_trainingloop_threshold <- best_threshold
logit_trainingloop_f1 <- best_f1
logit_trainingloop_precision <- best_precision
logit_trainingloop_recall <- best_recall
logit_trainingloop_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", logit_trainingloop_threshold, "\n")
cat("Precision: ", logit_trainingloop_precision, "\n")
cat("Recall: ", logit_trainingloop_recall, "\n")
cat("F1 Score: ", logit_trainingloop_f1, "\n")

# AUC-ROC
logit_trainingloop_roc_curve <- roc(true_classes_trainingloop, logit_predictions_trainingloop)
logit_trainingloop_auc_roc <- auc(logit_trainingloop_roc_curve)
cat("AUC-ROC: ", logit_trainingloop_auc_roc, "\n")

# Data for ggplot
logit_trainingloop_roc_data <- data.frame(
  fpr = rev(1 - logit_trainingloop_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(logit_trainingloop_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "Logit",
  data_type = "Imbalanced (Training Loop)"
)

# ROC curve plotting using ggplot2
ggplot(logit_trainingloop_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(logit_trainingloop_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_logit_trainingloop_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
logit_trainingloop_pr_curve <- pr.curve(scores.class0 = logit_predictions_trainingloop, weights.class0 = as.numeric(as.character(true_classes_trainingloop)), curve = TRUE)
logit_trainingloop_auc_pr <- logit_trainingloop_pr_curve$auc.integral
cat("AUC-PR: ", logit_trainingloop_auc_pr, "\n")

# Data for ggplot
logit_trainingloop_pr_data <- data.frame(
  recall = logit_trainingloop_pr_curve$curve[, 1],  # Recall
  precision = logit_trainingloop_pr_curve$curve[, 2],  # Precision
  method = "Logit",
  data_type = "Imbalanced (Training Loop)"
)

# PR curve plotting using ggplot2
ggplot(logit_trainingloop_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(logit_trainingloop_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_logit_trainingloop_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)




# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

### RANDOM FOREST
# LOOCV
total_obs <- nrow(data2)
rf_predictions_trainingloop <- numeric(total_obs)

start_time <- Sys.time()

for (i in 1:total_obs) {
  if (i %% 100 == 0) {
    cat("Processing iteration:", i, "/", total_obs, "\n")
    cat("Elapsed time:", Sys.time() - start_time, "\n")
  }
  
  data_Train_trainingloop <- data2[-i, ]
  data_Test_trainingloop <- data2[i, , drop = FALSE]
  
  # Random Forest training
  randomforest_trainingloop <- ranger(V114 ~ ., data = data_Train_trainingloop, num.trees = 100, probability = TRUE)
  rf_predictions_trainingloop[i] <- predict(randomforest_trainingloop, data = data_Test_trainingloop)$predictions[, 2]
}



# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  rf_trainingloop_predicted_classes <- ifelse(rf_predictions_trainingloop > threshold, 1, 0)
  
  # Creating the confusion matrix
  rf_trainingloop_confusionmatrix <- confusionMatrix(factor(rf_trainingloop_predicted_classes), factor(true_classes_trainingloop))
  
  # Precision, Recall and F1 Score
  rf_trainingloop_precision <- rf_trainingloop_confusionmatrix$byClass["Pos Pred Value"]
  rf_trainingloop_recall <- rf_trainingloop_confusionmatrix$byClass["Sensitivity"]
  rf_trainingloop_f1 <- 2 * (rf_trainingloop_precision * rf_trainingloop_recall) / (rf_trainingloop_precision + rf_trainingloop_recall)
  
  # Saving the best results
  if (!is.na(rf_trainingloop_f1) && rf_trainingloop_f1 > best_f1) {
    best_f1 <- rf_trainingloop_f1
    best_threshold <- threshold
    best_precision <- rf_trainingloop_precision
    best_recall <- rf_trainingloop_recall
    best_confusionmatrix <- rf_trainingloop_confusionmatrix
  }
}

# Overwriting variables to best values
rf_trainingloop_threshold <- best_threshold
rf_trainingloop_f1 <- best_f1
rf_trainingloop_precision <- best_precision
rf_trainingloop_recall <- best_recall
rf_trainingloop_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", rf_trainingloop_threshold, "\n")
cat("Precision: ", rf_trainingloop_precision, "\n")
cat("Recall: ", rf_trainingloop_recall, "\n")
cat("F1 Score: ", rf_trainingloop_f1, "\n")

# AUC-ROC
rf_trainingloop_roc_curve <- roc(true_classes_trainingloop, rf_predictions_trainingloop)
rf_trainingloop_auc_roc <- auc(rf_trainingloop_roc_curve)
cat("AUC-ROC: ", rf_trainingloop_auc_roc, "\n")

# Data for ggplot
rf_trainingloop_roc_data <- data.frame(
  fpr = rev(1 - rf_trainingloop_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(rf_trainingloop_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "Random Forest",
  data_type = "Imbalanced (Training Loop)"
)

# ROC curve plotting using ggplot2
ggplot(rf_trainingloop_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(rf_trainingloop_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_rf_trainingloop_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
rf_trainingloop_pr_curve <- pr.curve(scores.class0 = rf_predictions_trainingloop, weights.class0 = as.numeric(as.character(true_classes_trainingloop)), curve = TRUE)
rf_trainingloop_auc_pr <- rf_trainingloop_pr_curve$auc.integral
cat("AUC-PR: ", rf_trainingloop_auc_pr, "\n")

# Data for ggplot
rf_trainingloop_pr_data <- data.frame(
  recall = rf_trainingloop_pr_curve$curve[, 1],  # Recall
  precision = rf_trainingloop_pr_curve$curve[, 2],  # Precision
  method = "Random Forest",
  data_type = "Imbalanced (Training Loop)"
)

# PR curve plotting using ggplot2
ggplot(rf_trainingloop_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(rf_trainingloop_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_rf_trainingloop_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)








# Variables for storing the best results
best_threshold <- NULL
best_f1 <- 0
best_precision <- NULL
best_recall <- NULL
best_confusionmatrix <- NULL

### HDDT
# LOOCV
total_obs <- nrow(data2)
hddt_probabilities_trainingloop <- numeric(total_obs)

start_time <- Sys.time()

for (i in 1:total_obs) {
  if (i %% 100 == 0) {
    cat("Processing iteration:", i, "/", total_obs, "\n")
    cat("Elapsed time:", Sys.time() - start_time, "\n")
  }
  
  data_Train_trainingloop <- data2[-i, ]
  data_Test_trainingloop <- data2[i, , drop = FALSE]
  
  # Separation of the explained variable and explanatory variables
  X_train_trainingloop <- data_Train_trainingloop[, -which(names(data_Train_trainingloop) == target_variable)]
  y_train_trainingloop <- data_Train_trainingloop[[target_variable]]
  X_test_trainingloop <- data_Test_trainingloop[, -which(names(data_Test_trainingloop) == target_variable), drop = FALSE]
  
  # HDDT training
  hddt_trainingloop <- HDDT(X_train_trainingloop, y_train_trainingloop, C = 5)
  hddt_probabilities_trainingloop[i] <- HDDT_predict(hddt_trainingloop, X_test_trainingloop, return_prob = TRUE)
}

# Looping across the thresholds

for (threshold in thresholds) {
  # Prediction with the current threshold
  hddt_trainingloop_predicted_classes <- ifelse(hddt_probabilities_trainingloop > threshold, 1, 0)
  
  # Creating the confusion matrix
  hddt_trainingloop_confusionmatrix <- confusionMatrix(factor(hddt_trainingloop_predicted_classes), factor(true_classes_trainingloop))
  
  # Precision, Recall and F1 Score
  hddt_trainingloop_precision <- hddt_trainingloop_confusionmatrix$byClass["Pos Pred Value"]
  hddt_trainingloop_recall <- hddt_trainingloop_confusionmatrix$byClass["Sensitivity"]
  hddt_trainingloop_f1 <- 2 * (hddt_trainingloop_precision * hddt_trainingloop_recall) / (hddt_trainingloop_precision + hddt_trainingloop_recall)
  
  # Saving the best results
  if (!is.na(hddt_trainingloop_f1) && hddt_trainingloop_f1 > best_f1) {
    best_f1 <- hddt_trainingloop_f1
    best_threshold <- threshold
    best_precision <- hddt_trainingloop_precision
    best_recall <- hddt_trainingloop_recall
    best_confusionmatrix <- hddt_trainingloop_confusionmatrix
  }
}

# Overwriting variables to best values
hddt_trainingloop_threshold <- best_threshold
hddt_trainingloop_f1 <- best_f1
hddt_trainingloop_precision <- best_precision
hddt_trainingloop_recall <- best_recall
hddt_trainingloop_confusionmatrix <- best_confusionmatrix

cat("Threshold: ", hddt_trainingloop_threshold, "\n")
cat("Precision: ", hddt_trainingloop_precision, "\n")
cat("Recall: ", hddt_trainingloop_recall, "\n")
cat("F1 Score: ", hddt_trainingloop_f1, "\n")

# AUC-ROC
hddt_trainingloop_roc_curve <- roc(true_classes_trainingloop, hddt_probabilities_trainingloop)
hddt_trainingloop_auc_roc <- auc(hddt_trainingloop_roc_curve)
cat("AUC-ROC: ", hddt_trainingloop_auc_roc, "\n")

# Data for ggplot
hddt_trainingloop_roc_data <- data.frame(
  fpr = rev(1 - hddt_trainingloop_roc_curve$specificities),  # 1 - specificity is False Positive Rate (FPR)
  tpr = rev(hddt_trainingloop_roc_curve$sensitivities),       # True Positive Rate (TPR)
  method = "HDDT",
  data_type = "Imbalanced (Training Loop)"
)

# ROC curve plotting using ggplot2
ggplot(hddt_trainingloop_roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.1, label = paste("AUC =", round(hddt_trainingloop_auc_roc, 3)), color = "blue")

# Saving the plot
ggsave("1_myocardial_hddt_trainingloop_roc.png", plot = last_plot(), dpi = 300, width = 9, height = 5)

# AUC-PR
hddt_trainingloop_pr_curve <- pr.curve(scores.class0 = hddt_probabilities_trainingloop, weights.class0 = as.numeric(as.character(true_classes_trainingloop)), curve = TRUE)
hddt_trainingloop_auc_pr <- hddt_trainingloop_pr_curve$auc.integral
cat("AUC-PR: ", hddt_trainingloop_auc_pr, "\n")

# Data for ggplot
hddt_trainingloop_pr_data <- data.frame(
  recall = hddt_trainingloop_pr_curve$curve[, 1],  # Recall
  precision = hddt_trainingloop_pr_curve$curve[, 2],  # Precision
  method = "HDDT",
  data_type = "Imbalanced (Training Loop)"
)

# PR curve plotting using ggplot2
ggplot(hddt_trainingloop_pr_data, aes(x = recall, y = precision)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC =", round(hddt_trainingloop_auc_pr, 3)), color = "red")

# Saving the plot
ggsave("1_myocardial_hddt_trainingloop_pr.png", plot = last_plot(), dpi = 300, width = 9, height = 5)















# Initializing data frame for results
results <- data.frame(Method = character(), Data = character(), Precision = numeric(), Recall = numeric(), F1 = numeric(), AUROC = numeric(), AUPRC = numeric(), target_ratio = numeric(), Threshold = numeric(), stringsAsFactors = FALSE)

# Function for calculating the proportion of values of the explained variable
calculate_target_ratio <- function(data, target_var) {
  table(data[[target_var]]) / nrow(data)
}

# Function for adding results to data frame
add_results <- function(method, data_type, precision, recall, f1, auc_roc, auc_pr, train_data, target_var, threshold = NA) {
  target_ratio <- calculate_target_ratio(train_data, target_var)
  new_row <- data.frame(
    Method = method, 
    Data = data_type, 
    Precision = round(precision, 3), 
    Recall = round(recall, 3), 
    F1 = round(f1, 3), 
    AUROC = round(auc_roc, 3), 
    AUPRC = round(auc_pr, 3), 
    target_ratio = round(target_ratio[1], 3),
    Threshold = round(threshold, 2)
  )
  return(new_row)
}

# Results adding
results <- rbind(results, add_results("Logit", "Imbalanced", logit_precision, logit_recall, logit_f1, logit_auc_roc, logit_auc_pr, data_Train, "V114", logit_threshold))
results <- rbind(results, add_results("Logit", "SMOTE", logit_smote_precision, logit_smote_recall, logit_smote_f1, logit_smote_auc_roc, logit_smote_auc_pr, data_Train_smote, "V114", logit_smote_threshold))
results <- rbind(results, add_results("Logit", "Undersampling", logit_under_precision, logit_under_recall, logit_under_f1, logit_under_auc_roc, logit_under_auc_pr, data_Train_under, "V114", logit_under_threshold))
results <- rbind(results, add_results("Logit", "Oversampling", logit_over_precision, logit_over_recall, logit_over_f1, logit_over_auc_roc, logit_over_auc_pr, data_Train_over, "V114", logit_over_threshold))

# Adding a specific threshold for "Logit Imbalanced (Quantile)"
logit_quantile_threshold <- round(ifelse(is.infinite(min(logit_predictions[logit_predicted_classes_quantile == 1])), NA, min(logit_predictions[logit_predicted_classes_quantile == 1])), 2)
results <- rbind(results, add_results("Logit", "Imbalanced (Quantile 0.95)", logit_precision_quantile, logit_recall_quantile, logit_f1_quantile, logit_auc_roc, logit_auc_pr, data_Train, "V114", logit_quantile_threshold))
results <- rbind(results, add_results("Logit", "Imbalanced (Training Loop)", logit_trainingloop_precision, logit_trainingloop_recall, logit_trainingloop_f1, logit_trainingloop_auc_roc, logit_trainingloop_auc_pr, data, "V114", logit_trainingloop_threshold))

results <- rbind(results, add_results("Random Forest", "Imbalanced", rf_precision, rf_recall, rf_f1, rf_auc_roc, rf_auc_pr, data_Train, "V114", rf_threshold))
results <- rbind(results, add_results("Random Forest", "SMOTE", rf_smote_precision, rf_smote_recall, rf_smote_f1, rf_smote_auc_roc, rf_smote_auc_pr, data_Train_smote, "V114", rf_smote_threshold))
results <- rbind(results, add_results("Random Forest", "Undersampling", rf_under_precision, rf_under_recall, rf_under_f1, rf_under_auc_roc, rf_under_auc_pr, data_Train_under, "V114", rf_under_threshold))
results <- rbind(results, add_results("Random Forest", "Oversampling", rf_over_precision, rf_over_recall, rf_over_f1, rf_over_auc_roc, rf_over_auc_pr, data_Train_over, "V114", rf_over_threshold))

# Adding a specific threshold for "RF Imbalanced (Quantile)"
rf_quantile_threshold <- round(ifelse(is.infinite(min(rf_predictions[rf_predicted_classes_quantile == 1])), NA, min(rf_predictions[rf_predicted_classes_quantile == 1])), 2)
results <- rbind(results, add_results("Random Forest", "Imbalanced (Quantile 0.95)", rf_precision_quantile, rf_recall_quantile, rf_f1_quantile, rf_auc_roc, rf_auc_pr, data_Train, "V114", rf_quantile_threshold))
results <- rbind(results, add_results("Random Forest", "Imbalanced (Training Loop)", rf_trainingloop_precision, rf_trainingloop_recall, rf_trainingloop_f1, rf_trainingloop_auc_roc, rf_trainingloop_auc_pr, data, "V114", rf_trainingloop_threshold))

results <- rbind(results, add_results("HDDT", "Imbalanced", hddt_precision, hddt_recall, hddt_f1, hddt_auc_roc, hddt_auc_pr, data_Train, "V114", hddt_threshold))
results <- rbind(results, add_results("HDDT", "SMOTE", hddt_smote_precision, hddt_smote_recall, hddt_smote_f1, hddt_smote_auc_roc, hddt_smote_auc_pr, data_Train_smote, "V114", hddt_smote_threshold))
results <- rbind(results, add_results("HDDT", "Undersampling", hddt_under_precision, hddt_under_recall, hddt_under_f1, hddt_under_auc_roc, hddt_under_auc_pr, data_Train_under, "V114", hddt_under_threshold))
results <- rbind(results, add_results("HDDT", "Oversampling", hddt_over_precision, hddt_over_recall, hddt_over_f1, hddt_over_auc_roc, hddt_over_auc_pr, data_Train_over, "V114", hddt_over_threshold))

# Adding a specific threshold for "HDDT Imbalanced (Quantile)"
hddt_quantile_threshold <- round(ifelse(is.infinite(min(hddt_probabilities_trainingloop[hddt_predicted_classes_quantile == 1])), NA, min(hddt_probabilities_trainingloop[hddt_predicted_classes_quantile == 1])), 2)
results <- rbind(results, add_results("HDDT", "Imbalanced (Quantile 0.95)", hddt_precision_quantile, hddt_recall_quantile, hddt_f1_quantile, hddt_auc_roc, hddt_auc_pr, data_Train, "V114", hddt_quantile_threshold))
results <- rbind(results, add_results("HDDT", "Imbalanced (Training Loop)", hddt_trainingloop_precision, hddt_trainingloop_recall, hddt_trainingloop_f1, hddt_trainingloop_auc_roc, hddt_trainingloop_auc_pr, data, "V114", hddt_trainingloop_threshold))



# Printing the results
print(results)
results <- as.data.frame(results)

# Creating LaTeX table from data frame results
stargazer(
  results,
  type = "latex",
  title = "Results of Different Methods",
  summary = FALSE,
  rownames = FALSE,
  out = "results_table.tex",
  digits = 3
)


logit_confusionmatrix
logit_smote_confusionmatrix
logit_under_confusionmatrix
logit_over_confusionmatrix
logit_confusionmatrix_quantile
logit_trainingloop_confusionmatrix
rf_confusionmatrix
rf_smote_confusionmatrix
rf_under_confusionmatrix
rf_over_confusionmatrix
rf_confusionmatrix_quantile
rf_trainingloop_confusionmatrix
hddt_confusionmatrix
hddt_smote_confusionmatrix
hddt_under_confusionmatrix
hddt_over_confusionmatrix
hddt_confusionmatrix_quantile
hddt_trainingloop_confusionmatrix


















### LOGIT ROC
# Combining data into one data frame
logit_combined_roc_data <- bind_rows(logit_roc_data, logit_smote_roc_data, logit_under_roc_data, logit_over_roc_data, logit_trainingloop_roc_data)

# Plotting the combined ROC chart
logit_combined_roc <- ggplot(logit_combined_roc_data, aes(x = fpr, y = tpr, color = data_type)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Data Type") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Imbalanced" = "#F8766D", "Oversampling" = "#00BA38", "SMOTE" = "#00BFC4", "Undersampling" = "#C77CFF", "Imbalanced (Training Loop)" = "orange")) +
  annotate("text", x = 0.6, y = 0.40, label = paste("Imbalanced AUC =", round(logit_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.30, label = paste("SMOTE AUC =", round(logit_smote_auc_roc, 3)), color = "#00BFC4", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Undersampling AUC =", round(logit_under_auc_roc, 3)), color = "#C77CFF", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("Oversampling AUC =", round(logit_over_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.00, label = paste("Imbalanced (Training Loop) AUC =", round(logit_trainingloop_auc_roc, 3)), color = "orange", size = 8)

logit_combined_roc

# Saving the chart
ggsave("1_myocardial_logit_combined_roc.png", plot = logit_combined_roc, dpi = 300, width = 9, height = 5)


### LOGIT PR
# Combining data into one data frame
logit_combined_pr_data <- bind_rows(logit_pr_data, logit_smote_pr_data, logit_under_pr_data, logit_over_pr_data, logit_trainingloop_pr_data)

# Plotting the combined PR chart
logit_combined_pr <- ggplot(logit_combined_pr_data, aes(x = recall, y = precision, color = data_type)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Data Type") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Imbalanced" = "#F8766D", "Oversampling" = "#00BA38", "SMOTE" = "#00BFC4", "Undersampling" = "#C77CFF", "Imbalanced (Training Loop)" = "orange")) +
  annotate("text", x = 0.6, y = 0.90, label = paste("Imbalanced AUC =", round(logit_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.80, label = paste("SMOTE AUC =", round(logit_smote_auc_pr, 3)), color = "#00BFC4", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Undersampling AUC =", round(logit_under_auc_pr, 3)), color = "#C77CFF", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("Oversampling AUC =", round(logit_over_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.50, label = paste("Imbalanced (Training Loop) AUC =", round(logit_trainingloop_auc_pr, 3)), color = "orange", size = 8)

logit_combined_pr

# Saving the chart
ggsave("1_myocardial_logit_combined_pr.png", plot = logit_combined_pr, dpi = 300, width = 9, height = 5)








### RANDOM FOREST ROC
# Combining data into one data frame
rf_combined_roc_data <- bind_rows(rf_roc_data, rf_smote_roc_data, rf_under_roc_data, rf_over_roc_data, rf_trainingloop_roc_data)

# Plotting the combined ROC chart
rf_combined_roc <- ggplot(rf_combined_roc_data, aes(x = fpr, y = tpr, color = data_type)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Data Type") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Imbalanced" = "#F8766D", "Oversampling" = "#00BA38", "SMOTE" = "#00BFC4", "Undersampling" = "#C77CFF", "Imbalanced (Training Loop)" = "orange")) +
  annotate("text", x = 0.6, y = 0.40, label = paste("Imbalanced AUC =", round(rf_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.30, label = paste("SMOTE AUC =", round(rf_smote_auc_roc, 3)), color = "#00BFC4", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Undersampling AUC =", round(rf_under_auc_roc, 3)), color = "#C77CFF", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("Oversampling AUC =", round(rf_over_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.00, label = paste("Imbalanced (Training Loop) AUC =", round(rf_trainingloop_auc_roc, 3)), color = "orange", size = 8)

rf_combined_roc

# Saving the chart
ggsave("1_myocardial_rf_combined_roc.png", plot = rf_combined_roc, dpi = 300, width = 9, height = 5)


### RF PR
# Combining data into one data frame
rf_combined_pr_data <- bind_rows(rf_pr_data, rf_smote_pr_data, rf_under_pr_data, rf_over_pr_data, rf_trainingloop_pr_data)

# Plotting the combined PR chart
rf_combined_pr <- ggplot(rf_combined_pr_data, aes(x = recall, y = precision, color = data_type)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Data Type") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Imbalanced" = "#F8766D", "Oversampling" = "#00BA38", "SMOTE" = "#00BFC4", "Undersampling" = "#C77CFF", "Imbalanced (Training Loop)" = "orange")) +
  annotate("text", x = 0.6, y = 0.90, label = paste("Imbalanced AUC =", round(rf_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.80, label = paste("SMOTE AUC =", round(rf_smote_auc_pr, 3)), color = "#00BFC4", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Undersampling AUC =", round(rf_under_auc_pr, 3)), color = "#C77CFF", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("Oversampling AUC =", round(rf_over_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.50, label = paste("Imbalanced (Training Loop) AUC =", round(rf_trainingloop_auc_pr, 3)), color = "orange", size = 8)

rf_combined_pr

# Saving the chart
ggsave("1_myocardial_rf_combined_pr.png", plot = rf_combined_pr, dpi = 300, width = 9, height = 5)






### HDDT ROC
# Combining data into one data frame
hddt_combined_roc_data <- bind_rows(hddt_roc_data, hddt_smote_roc_data, hddt_under_roc_data, hddt_over_roc_data, hddt_trainingloop_roc_data)

# Plotting the combined ROC chart
hddt_combined_roc <- ggplot(hddt_combined_roc_data, aes(x = fpr, y = tpr, color = data_type)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Data Type") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Imbalanced" = "#F8766D", "Oversampling" = "#00BA38", "SMOTE" = "#00BFC4", "Undersampling" = "#C77CFF", "Imbalanced (Training Loop)" = "orange")) +
  annotate("text", x = 0.6, y = 0.40, label = paste("Imbalanced AUC =", round(hddt_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.30, label = paste("SMOTE AUC =", round(hddt_smote_auc_roc, 3)), color = "#00BFC4", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Undersampling AUC =", round(hddt_under_auc_roc, 3)), color = "#C77CFF", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("Oversampling AUC =", round(hddt_over_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.00, label = paste("Imbalanced (Training Loop) AUC =", round(hddt_trainingloop_auc_roc, 3)), color = "orange", size = 8)

hddt_combined_roc

# Saving the chart
ggsave("1_myocardial_hddt_combined_roc.png", plot = hddt_combined_roc, dpi = 300, width = 9, height = 5)


### HDDT PR
# Combining data into one data frame
hddt_combined_pr_data <- bind_rows(hddt_pr_data, hddt_smote_pr_data, hddt_under_pr_data, hddt_over_pr_data, hddt_trainingloop_pr_data)

# Plotting the combined PR chart
hddt_combined_pr <- ggplot(hddt_combined_pr_data, aes(x = recall, y = precision, color = data_type)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Data Type") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Imbalanced" = "#F8766D", "Oversampling" = "#00BA38", "SMOTE" = "#00BFC4", "Undersampling" = "#C77CFF", "Imbalanced (Training Loop)" = "orange")) +
  annotate("text", x = 0.6, y = 0.90, label = paste("Imbalanced AUC =", round(hddt_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.80, label = paste("SMOTE AUC =", round(hddt_smote_auc_pr, 3)), color = "#00BFC4", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Undersampling AUC =", round(hddt_under_auc_pr, 3)), color = "#C77CFF", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("Oversampling AUC =", round(hddt_over_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.50, label = paste("Imbalanced (Training Loop) AUC =", round(hddt_trainingloop_auc_pr, 3)), color = "orange", size = 8)

hddt_combined_pr

# Saving the chart
ggsave("1_myocardial_hddt_combined_pr.png", plot = hddt_combined_pr, dpi = 300, width = 9, height = 5)





##### ROC
### IMBALANCED DATA
# Combining data into one data frame
combined_imbalanced_roc_data <- bind_rows(logit_roc_data, rf_roc_data, hddt_roc_data)

# Plotting the combined ROC chart
combined_imbalanced_roc <- ggplot(combined_imbalanced_roc_data, aes(x = fpr, y = tpr, color = method)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.30, label = paste("Logit AUC =", round(logit_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Random Forest AUC =", round(rf_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("HDDT AUC =", round(hddt_auc_roc, 3)), color = "#00BFC4", size = 8)

combined_imbalanced_roc

# Saving the chart
ggsave("1_myocardial_combined_imbalanced_roc.png", plot = combined_imbalanced_roc, dpi = 300, width = 9, height = 5)





### SMOTE DATA
# Combining data into one data frame
combined_smote_roc_data <- bind_rows(logit_smote_roc_data, rf_smote_roc_data, hddt_smote_roc_data)

# Plotting the combined ROC chart
combined_smote_roc <- ggplot(combined_smote_roc_data, aes(x = fpr, y = tpr, color = method)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.30, label = paste("Logit AUC =", round(logit_smote_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Random Forest AUC =", round(rf_smote_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("HDDT AUC =", round(hddt_smote_auc_roc, 3)), color = "#00BFC4", size = 8)

combined_smote_roc

# Saving the chart
ggsave("1_myocardial_combined_smote_roc.png", plot = combined_smote_roc, dpi = 300, width = 9, height = 5)





### UNDERSAMPLED DATA
# Combining data into one data frame
combined_under_roc_data <- bind_rows(logit_under_roc_data, rf_under_roc_data, hddt_under_roc_data)

# Plotting the combined ROC chart
combined_under_roc <- ggplot(combined_under_roc_data, aes(x = fpr, y = tpr, color = method)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.30, label = paste("Logit AUC =", round(logit_under_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Random Forest AUC =", round(rf_under_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("HDDT AUC =", round(hddt_under_auc_roc, 3)), color = "#00BFC4", size = 8)

combined_under_roc

# Saving the chart
ggsave("1_myocardial_combined_under_roc.png", plot = combined_under_roc, dpi = 300, width = 9, height = 5)





### OVERSAMPLED DATA
# Combining data into one data frame
combined_over_roc_data <- bind_rows(logit_over_roc_data, rf_over_roc_data, hddt_over_roc_data)

# Plotting the combined ROC chart
combined_over_roc <- ggplot(combined_over_roc_data, aes(x = fpr, y = tpr, color = method)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.30, label = paste("Logit AUC =", round(logit_over_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Random Forest AUC =", round(rf_over_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("HDDT AUC =", round(hddt_over_auc_roc, 3)), color = "#00BFC4", size = 8)

combined_over_roc

# Saving the chart
ggsave("1_myocardial_combined_over_roc.png", plot = combined_over_roc, dpi = 300, width = 9, height = 5)





### IMBALANCED quantile DATA
# Combining data into one data frame
combined_trainingloop_roc_data <- bind_rows(logit_trainingloop_roc_data, rf_trainingloop_roc_data, hddt_trainingloop_roc_data)

# Plotting the combined ROC chart
combined_trainingloop_roc <- ggplot(combined_trainingloop_roc_data, aes(x = fpr, y = tpr, color = method)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.30, label = paste("Logit AUC =", round(logit_trainingloop_auc_roc, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.20, label = paste("Random Forest AUC =", round(rf_trainingloop_auc_roc, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.10, label = paste("HDDT AUC =", round(hddt_trainingloop_auc_roc, 3)), color = "#00BFC4", size = 8)

combined_trainingloop_roc

# Saving the chart
ggsave("1_myocardial_combined_trainingloop_roc.png", plot = combined_trainingloop_roc, dpi = 300, width = 9, height = 5)










##### PR
### IMBALANCED DATA
# Combining data into one data frame
combined_imbalanced_pr_data <- bind_rows(logit_pr_data, rf_pr_data, hddt_pr_data)

# Plotting the combined PR chart
combined_imbalanced_pr <- ggplot(combined_imbalanced_pr_data, aes(x = recall, y = precision, color = method)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.80, label = paste("Logit AUC =", round(logit_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Random Forest AUC =", round(rf_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("HDDT AUC =", round(hddt_auc_pr, 3)), color = "#00BFC4", size = 8)

combined_imbalanced_pr

# Saving the chart
ggsave("1_myocardial_combined_imbalanced_pr.png", plot = combined_imbalanced_pr, dpi = 300, width = 9, height = 5)





### SMOTE DATA
# Combining data into one data frame
combined_smote_pr_data <- bind_rows(logit_smote_pr_data, rf_smote_pr_data, hddt_smote_pr_data)

# Plotting the combined PR chart
combined_smote_pr <- ggplot(combined_smote_pr_data, aes(x = recall, y = precision, color = method)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.80, label = paste("Logit AUC =", round(logit_smote_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Random Forest AUC =", round(rf_smote_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("HDDT AUC =", round(hddt_smote_auc_pr, 3)), color = "#00BFC4", size = 8)

combined_smote_pr

# Saving the chart
ggsave("1_myocardial_combined_smote_pr.png", plot = combined_smote_pr, dpi = 300, width = 9, height = 5)





### UNDERSAMPLED DATA
# Combining data into one data frame
combined_under_pr_data <- bind_rows(logit_under_pr_data, rf_under_pr_data, hddt_under_pr_data)

# Plotting the combined PR chart
combined_under_pr <- ggplot(combined_under_pr_data, aes(x = recall, y = precision, color = method)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.80, label = paste("Logit AUC =", round(logit_under_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Random Forest AUC =", round(rf_under_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("HDDT AUC =", round(hddt_under_auc_pr, 3)), color = "#00BFC4", size = 8)

combined_under_pr

# Saving the chart
ggsave("1_myocardial_combined_under_pr.png", plot = combined_under_pr, dpi = 300, width = 9, height = 5)





### OVERSAMPLED DATA
# Combining data into one data frame
combined_over_pr_data <- bind_rows(logit_over_pr_data, rf_over_pr_data, hddt_over_pr_data)

# Plotting the combined PR chart
combined_over_pr <- ggplot(combined_over_pr_data, aes(x = recall, y = precision, color = method)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.80, label = paste("Logit AUC =", round(logit_over_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Random Forest AUC =", round(rf_over_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("HDDT AUC =", round(hddt_over_auc_pr, 3)), color = "#00BFC4", size = 8)

combined_over_pr

# Saving the chart
ggsave("1_myocardial_combined_over_pr.png", plot = combined_over_pr, dpi = 300, width = 9, height = 5)








### IMBALANCED TRAINING LOOP DATA
# Combining data into one data frame
combined_trainingloop_pr_data <- bind_rows(logit_trainingloop_pr_data, rf_trainingloop_pr_data, hddt_trainingloop_pr_data)

# Plotting the combined PR chart
combined_trainingloop_pr <- ggplot(combined_trainingloop_pr_data, aes(x = recall, y = precision, color = method)) +
  geom_line(size = 1) +
  labs(title = "",
       x = "Recall",
       y = "Precision",
       color = "Method") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 16)) +
  scale_color_manual(values = c("Logit" = "#F8766D", "Random Forest" = "#00BA38", "HDDT" = "#00BFC4")) +
  annotate("text", x = 0.6, y = 0.80, label = paste("Logit AUC =", round(logit_trainingloop_auc_pr, 3)), color = "#F8766D", size = 8) +
  annotate("text", x = 0.6, y = 0.70, label = paste("Random Forest AUC =", round(rf_trainingloop_auc_pr, 3)), color = "#00BA38", size = 8) +
  annotate("text", x = 0.6, y = 0.60, label = paste("HDDT AUC =", round(hddt_trainingloop_auc_pr, 3)), color = "#00BFC4", size = 8)

combined_trainingloop_pr

# Saving the chart
ggsave("1_myocardial_combined_trainingloop_pr.png", plot = combined_trainingloop_pr, dpi = 300, width = 9, height = 5)


