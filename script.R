library(tidyverse)
library(lubridate)
library(tidyverse)
library(caret)
library(e1071)
library(xgboost)
library(pROC)
library(SHAPforxgboost)

# Load raw data
df_raw <- read_csv("huggingface.csv")
summary(df_raw)

# Parse downloads and likes from combined string format
parse_number <- function(s) {
  if (is.na(s) || s == "") return(NA)
  s <- str_trim(s)
  
  multiplier <- case_when(
    str_detect(s, "K$|k$") ~ 1000,
    str_detect(s, "M$") ~ 1000000,
    str_detect(s, "B$") ~ 1000000000,
    TRUE ~ 1
  )
  
  number <- as.numeric(str_remove(s, "[KMBkmb]$"))
  return(number * multiplier)
}

df_parsed <- df_raw %>%
  mutate(
    downloads_str = str_extract(`downloads and likes`, "^[0-9.]+[KMBkmb]?"),
    likes_str = str_trim(str_extract(`downloads and likes`, "[0-9.]+[KMBkmb]?$")),
    downloads = map_dbl(downloads_str, parse_number),
    likes = map_dbl(likes_str, parse_number)
  )

# data quality 
cat("Dataset dimensions:", nrow(df_parsed))
cat("Downloads parsed:", sum(!is.na(df_parsed$downloads)))
cat("Likes parsed:", sum(!is.na(df_parsed$likes)))

# distribution
downloads_clean <- df_parsed$downloads[!is.na(df_parsed$downloads)]
cat("Downloads summary:\n")
cat("  Mean:", format(mean(downloads_clean), big.mark=","))
cat("  Median:", format(median(downloads_clean), big.mark=","))
cat("  Max:", format(max(downloads_clean), big.mark=","))
cat("  Skewness:", round(e1071::skewness(downloads_clean), 2))

# Is likes actually informative for adoption?
cat("Likes vs Adoption:\n")
high_adoption <- df_clean %>% filter(is_high_adoption == TRUE)
low_adoption <- df_clean %>% filter(is_high_adoption == FALSE)

cat("  Mean likes (High adoption):", round(mean(high_adoption$likes), 1), "\n")
cat("  Mean likes (Low adoption):", round(mean(low_adoption$likes), 1), "\n")
cat("  Median likes (High):", median(high_adoption$likes), "\n")
cat("  Median likes (Low):", median(low_adoption$likes), "\n\n")

# Test if significantly different
t_test <- t.test(high_adoption$likes, low_adoption$likes)
cat("T-test p-value:", format(t_test$p.value, scientific = TRUE), "\n")


quantile(df_clean$downloads, probs = c(0.5, 0.75, 0.9, 0.95, 0.99))

# binary balance
median_downloads <- median(downloads_clean)
pct_above <- mean(downloads_clean > median_downloads) * 100
cat("Binary target (is_high_adoption):\n")
cat("  Above median:", round(pct_above, 1), "%\n")
cat("  Below median:", round(100 - pct_above, 1), "%\n\n")

write_csv(df_parsed, "huggingface_parsed.csv")

# ============================================================================
# Feature Engineering
# ============================================================================

df <- read_csv("huggingface_parsed.csv")

# author from url
df <- df %>%
  mutate(
    author = str_extract(`model url`, "(?<=co/)([^/]+)"),
    author = if_else(is.na(author), title, author)
  )

# author type
df <- df %>%
  mutate(
    author_type = case_when(
      tolower(author) %in% c("google", "facebook", "openai", "microsoft", 
                             "meta", "nvidia", "amazon", "ibm", "deepmind", 
                             "anthropic", "apple") ~ "big_tech",
      tolower(author) %in% c("huggingface", "sentence-transformers", 
                             "CompVis", "stabilityai", "runwayml", 
                             "huggingface", "sentence-transformers", "compvis", "stabilityai", "runwayml",
                             "openai", "anthropic", "cohere", "adept", "inflection",
                             "mistralai", "together", "aleph-alpha", "ai21labs",
                             "eleutherai", "bigscience", "bigcode", "tiiuae",
                             "cerebras", "lightricks", "midjourney", "character-ai",
                             "laion", "eleuther", "salesforce", "databricks",
                             "mosaicml", "nomic", "h2oai") ~ "ai_company",
      str_detect(tolower(author), "^[a-z]+[0-9]*$") ~ "individual",
      TRUE ~ "organization"
    )
  )

# models per author
author_counts <- df %>%
  group_by(author) %>%
  summarise(n_models_by_author = n()) %>%
  ungroup()

df <- df %>%
  left_join(author_counts, by = "author")

# model family
df <- df %>%
  mutate(
    model_family = case_when(
      str_detect(tolower(title), "\\bbert\\b") & 
        !str_detect(tolower(title), "roberta|albert") ~ "BERT",
      str_detect(tolower(title), "roberta") ~ "RoBERTa",
      str_detect(tolower(title), "albert") ~ "ALBERT",
      str_detect(tolower(title), "\\bgpt") ~ "GPT",
      str_detect(tolower(title), "\\bt5\\b") ~ "T5",
      str_detect(tolower(title), "\\bbart\\b") ~ "BART",
      str_detect(tolower(title), "stable.diffusion") ~ "Stable_Diffusion",
      str_detect(tolower(title), "llama") ~ "LLaMA",
      str_detect(tolower(title), "whisper") ~ "Whisper",
      str_detect(tolower(title), "clip|\\bvit\\b|vision") ~ "Vision",
      str_detect(tolower(title), "xlm") ~ "XLM",
      TRUE ~ "Other"
    )
  )

# model size
df <- df %>%
  mutate(
    model_size = case_when(
      str_detect(tolower(title), "tiny") ~ "tiny",
      str_detect(tolower(title), "mini") ~ "mini",
      str_detect(tolower(title), "small") ~ "small",
      str_detect(tolower(title), "\\bbase\\b") ~ "base",
      str_detect(tolower(title), "medium") ~ "medium",
      str_detect(tolower(title), "xxl") ~ "xxl",
      str_detect(tolower(title), "xlarge|\\bxl\\b") ~ "xlarge",
      str_detect(tolower(title), "\\blarge\\b") & 
        !str_detect(tolower(title), "xlarge") ~ "large",
      TRUE ~ "unknown"
    )
  )

# domain
df <- df %>%
  mutate(
    domain = case_when(
      str_detect(tolower(title), "bio|medical|clinical|health|pubmed") ~ "medical",
      str_detect(tolower(title), "legal|law") ~ "legal",
      str_detect(tolower(title), "financial|finbert|finance") ~ "finance",
      str_detect(tolower(title), "code|programming|codex") ~ "code",
      str_detect(tolower(title), "news") ~ "news",
      str_detect(tolower(title), "social|twitter") ~ "social_media",
      str_detect(tolower(title), "scientific|sci") ~ "scientific",
      TRUE ~ "general"
    )
  )

# language
df <- df %>%
  mutate(
    language = case_when(
      str_detect(tolower(title), "multilingual|xlm|mbert|m-bert") ~ "multilingual",
      str_detect(tolower(title), "chinese|zh-|zh_") ~ "chinese",
      str_detect(tolower(title), "japanese|ja-|ja_") ~ "japanese",
      str_detect(tolower(title), "french|fr-|fr_") ~ "french",
      str_detect(tolower(title), "german|de-|de_") ~ "german",
      str_detect(tolower(title), "spanish|es-|es_") ~ "spanish",
      str_detect(tolower(title), "arabic|ar-|ar_") ~ "arabic",
      str_detect(tolower(title), "russian|ru-|ru_") ~ "russian",
      TRUE ~ "english"
    )
  )

df <- df %>%
  mutate(
    is_finetuned = str_detect(tolower(title), 
                              "finetune|fine-tune|finetuned|fine-tuned|ft-"),
    is_distilled = str_detect(tolower(title), "distil")
  )

# dates
df <- df %>%
  mutate(
    updated_date = case_when(
      str_detect(updated, "days? ago") ~ {
        days_ago <- as.numeric(str_extract(updated, "\\d+"))
        Sys.Date() - days(days_ago)
      },
      str_detect(updated, "months? ago") ~ {
        months_ago <- as.numeric(str_extract(updated, "\\d+"))
        Sys.Date() - months(months_ago)
      },
      str_detect(updated, "years? ago") ~ {
        years_ago <- as.numeric(str_extract(updated, "\\d+"))
        Sys.Date() - years(years_ago)
      },
      TRUE ~ mdy(updated)
    )
  )

df <- df %>%
  mutate(
    is_high_adoption = downloads > 15 # 75th percentile
  )

# more eda

df %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 40) +
  facet_wrap(~ name, scales = "free")

write_csv(df, "huggingface_final.csv")




set.seed(123)

# final dataset after feature
df <- read_csv("huggingface_final.csv")



# ============================================================================
# Train test split
# ============================================================================

# remove rows w NA in target
df_clean <- df %>% 
  filter(!is.na(is_high_adoption)) %>%
  filter(!is.na(model_age_days))

cor(select(df_clean, where(is.numeric)), use = "complete.obs")


# stratified split
train_index <- createDataPartition(df_clean$is_high_adoption, 
                                   p = 0.8, 
                                   list = FALSE)

train_data <- df_clean[train_index, ]
test_data <- df_clean[-train_index, ]

# convert to factor
train_data$is_high_adoption <- factor(train_data$is_high_adoption, 
                                      levels = c(FALSE, TRUE),
                                      labels = c("Low", "High"))
test_data$is_high_adoption <- factor(test_data$is_high_adoption,
                                     levels = c(FALSE, TRUE),
                                     labels = c("Low", "High"))

cat("Split complete:\n")
cat("  Train:", nrow(train_data), "\n")
cat("  Test:", nrow(test_data), "\n")
cat("  Train balance:", round(mean(train_data$is_high_adoption == "High")*100, 1), "% High\n")
cat("  Test balance:", round(mean(test_data$is_high_adoption == "High")*100, 1), "% High\n\n")

# ============================================================================
# features
# ============================================================================

feature_cols_no_leakage <- c(
  "n_models_by_author",  # Organizational
  "model_age_days",      # Temporal
  "author_type",         # Organizational
  "model_family",        # Technical
  "model_size",          # Technical
  "domain",              # Specialization
  "language",            # Specialization
  "is_finetuned",        # Technical
  "is_distilled"         # Technical
)

# sync factor levels between train and test

categorical_vars <- c("author_type", "model_family", "model_size", "domain", "language")

for(col in categorical_vars) {
  all_levels <- unique(c(as.character(train_data[[col]]), 
                         as.character(test_data[[col]])))
  train_data[[col]] <- factor(train_data[[col]], levels = all_levels)
  test_data[[col]] <- factor(test_data[[col]], levels = all_levels)
}

# ============================================================================
# NAIVE BAYES MODEL
# ============================================================================

train_nb <- train_data %>%
  mutate(
    log_n_models = log1p(n_models_by_author)
  ) %>%
  select(
    log_n_models,       # log-transformed for normality
    model_age_days,     
    author_type, model_family, model_size, domain, language,
    is_finetuned, is_distilled,
    is_high_adoption
  )

test_nb <- test_data %>%
  mutate(
    log_n_models = log1p(n_models_by_author)
  ) %>%
  select(
    log_n_models, model_age_days,
    author_type, model_family, model_size, domain, language,
    is_finetuned, is_distilled,
    is_high_adoption
  )
for(var in categorical_vars) {
  train_nb[[var]] <- as.factor(train_nb[[var]])
  test_nb[[var]] <- as.factor(test_nb[[var]])
}

# TRAIN
nb_model <- naiveBayes(
  is_high_adoption ~ ., 
  data = train_nb,
  laplace = 1
)

# PREDICT
nb_pred_prob <- predict(nb_model, test_nb, type = "raw")

# find optimal threshold
thresholds <- seq(0.1, 0.9, 0.01)
f1_scores_nb <- numeric(length(thresholds))

for(i in seq_along(thresholds)) {
  t <- thresholds[i]
  pred_temp <- factor(ifelse(nb_pred_prob[, "High"] > t, "High", "Low"), 
                      levels = c("Low", "High"))
  cm_temp <- confusionMatrix(pred_temp, test_nb$is_high_adoption, positive = "High")
  f1_scores_nb[i] <- cm_temp$byClass["F1"]
}

optimal_threshold_nb <- thresholds[which.max(f1_scores_nb)]
max_f1_nb <- max(f1_scores_nb, na.rm = TRUE)

cat("  Optimal threshold:", optimal_threshold_nb, "\n")
cat("  Max F1:", round(max_f1_nb, 3), "\n\n")

# use optimal threshold
nb_pred <- factor(ifelse(nb_pred_prob[, "High"] > optimal_threshold_nb, "High", "Low"), 
                  levels = c("Low", "High"))

# EVAL
nb_cm <- confusionMatrix(nb_pred, test_nb$is_high_adoption, positive = "High")
nb_roc <- roc(test_nb$is_high_adoption, nb_pred_prob[, "High"])

cat("Naive Bayes Confusion Matrix:\n")
print(nb_cm$table)

# metrics
nb_metrics <- data.frame(
  Model = "Naive Bayes",
  Accuracy = nb_cm$overall["Accuracy"],
  Precision = nb_cm$byClass["Precision"],
  Recall = nb_cm$byClass["Sensitivity"],
  F1 = nb_cm$byClass["F1"],
  AUC = auc(nb_roc),
  Threshold = optimal_threshold_nb
)

cat("Naive Bayes Final Performance:\n")
cat("  F1:", round(nb_metrics$F1, 3), "\n")
cat("  AUC:", round(nb_metrics$AUC, 3), "\n")
cat("  Precision:", round(nb_metrics$Precision, 3), "\n")
cat("  Recall:", round(nb_metrics$Recall, 3), "\n\n")

# chi-square independence tests
cat("Testing Naive Bayes Independence Assumption:\n")

categorical_pairs <- combn(categorical_vars, 2, simplify = FALSE)

for(pair in categorical_pairs) {
  tryCatch({
    chi_test <- chisq.test(table(train_nb[[pair[1]]], train_nb[[pair[2]]]))
    cat("  ", pair[1], "vs", pair[2], 
        "| p-value:", format(chi_test$p.value, scientific = TRUE, digits = 3),
        ifelse(chi_test$p.value < 0.001, "STRONG DEPENDENCE", ""), "\n")
  }, error = function(e) {
    cat("  ", pair[1], "vs", pair[2], ": ERROR\n")
  })
}

# ============================================================================
# XGBOOST MODEL
# ============================================================================

dummy_formula <- as.formula(paste("~", paste(feature_cols_no_leakage, collapse = " + "), "- 1"))
train_matrix <- model.matrix(dummy_formula, data = train_data)
test_matrix <- model.matrix(dummy_formula, data = test_data)

# convert
train_label <- as.numeric(train_data$is_high_adoption == "High")
test_label <- as.numeric(test_data$is_high_adoption == "High")

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# class weight
scale_pos_weight_value <- sum(train_label == 0) / sum(train_label == 1)

# SEQUENTIAL HYPERPARAMETER TUNING
depths <- c(4, 6, 8, 10, 12)
best_depth <- 8
best_auc_depth <- 0

for(d in depths) {
  params_test <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = d,
    eta = 0.1,
    scale_pos_weight = scale_pos_weight_value
  )
  
  cv_model <- xgb.cv(
    params = params_test,
    data = dtrain,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_iter <- cv_model$best_iteration
  if(is.null(best_iter) || best_iter < 10) {
    best_iter <- which.max(cv_model$evaluation_log$test_auc_mean)
  }
  
  test_auc <- cv_model$evaluation_log$test_auc_mean[best_iter]
  
  cat("  max_depth =", d, "| AUC =", round(test_auc, 4), 
      "| best_iter =", best_iter, "\n")
  
  if(test_auc > best_auc_depth) {
    best_auc_depth <- test_auc
    best_depth <- d
  }
}

etas <- c(0.01, 0.03, 0.05, 0.1, 0.2)
best_eta <- 0.05
best_auc_eta <- 0

for(e in etas) {
  params_test <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = best_depth,
    eta = e,
    scale_pos_weight = scale_pos_weight_value
  )
  
  cv_model <- xgb.cv(
    params = params_test,
    data = dtrain,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_iter <- cv_model$best_iteration
  if(is.null(best_iter) || best_iter < 10) {
    best_iter <- which.max(cv_model$evaluation_log$test_auc_mean)
  }
  
  test_auc <- cv_model$evaluation_log$test_auc_mean[best_iter]
  
  cat("  eta =", e, "| AUC =", round(test_auc, 4), 
      "| best_iter =", best_iter, "\n")
  
  if(test_auc > best_auc_eta) {
    best_auc_eta <- test_auc
    best_eta <- e
  }
}

subsamples <- c(0.6, 0.7, 0.8, 0.9)
colsamples <- c(0.6, 0.7, 0.8, 0.9)
best_subsample <- 0.7
best_colsample <- 0.7
best_auc_sample <- 0

for(ss in subsamples) {
  for(cs in colsamples) {
    params_test <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = best_depth,
      eta = best_eta,
      subsample = ss,
      colsample_bytree = cs,
      scale_pos_weight = scale_pos_weight_value
    )
    
    cv_model <- xgb.cv(
      params = params_test,
      data = dtrain,
      nrounds = 500,
      nfold = 5,
      early_stopping_rounds = 20,
      verbose = 0
    )
    
    best_iter <- cv_model$best_iteration
    if(is.null(best_iter) || best_iter < 10) {
      best_iter <- which.max(cv_model$evaluation_log$test_auc_mean)
    }
    
    test_auc <- cv_model$evaluation_log$test_auc_mean[best_iter]
    
    if(test_auc > best_auc_sample) {
      best_auc_sample <- test_auc
      best_subsample <- ss
      best_colsample <- cs
      cat("New best - subsample =", ss, "colsample =", cs, 
          "| AUC =", round(test_auc, 4), "\n")
    }
  }
}

gammas <- c(0, 0.1, 0.2, 0.5)
min_child_weights <- c(1, 3, 5, 7)
best_gamma <- 0.1
best_min_child <- 5
best_auc_reg <- 0

for(g in gammas) {
  for(mcw in min_child_weights) {
    params_test <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = best_depth,
      eta = best_eta,
      subsample = best_subsample,
      colsample_bytree = best_colsample,
      gamma = g,
      min_child_weight = mcw,
      scale_pos_weight = scale_pos_weight_value
    )
    
    cv_model <- xgb.cv(
      params = params_test,
      data = dtrain,
      nrounds = 500,
      nfold = 5,
      early_stopping_rounds = 20,
      verbose = 0
    )
    
    best_iter <- cv_model$best_iteration
    if(is.null(best_iter) || best_iter < 10) {
      best_iter <- which.max(cv_model$evaluation_log$test_auc_mean)
    }
    
    test_auc <- cv_model$evaluation_log$test_auc_mean[best_iter]
    
    if(test_auc > best_auc_reg) {
      best_auc_reg <- test_auc
      best_gamma <- g
      best_min_child <- mcw
      cat(" New best - gamma =", g, "min_child_weight =", mcw, 
          "| AUC =", round(test_auc, 4), "\n")
    }
  }
}

cat("Best gamma:", best_gamma, "\n")
cat("Best min_child_weight:", best_min_child, "\n")
cat("AUC:", round(best_auc_reg, 4), "\n\n")

params_final <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_depth,
  eta = best_eta,
  subsample = best_subsample,
  colsample_bytree = best_colsample,
  gamma = best_gamma,
  min_child_weight = best_min_child,
  scale_pos_weight = scale_pos_weight_value
)

cat("Optimal parameters:\n")
cat("  max_depth:", best_depth, "\n")
cat("  eta:", best_eta, "\n")
cat("  subsample:", best_subsample, "\n")
cat("  colsample_bytree:", best_colsample, "\n")
cat("  gamma:", best_gamma, "\n")
cat("  min_child_weight:", best_min_child, "\n")
cat("  scale_pos_weight:", round(scale_pos_weight_value, 3), "\n\n")

# cross-validation to get best number of rounds
cv_model_final <- xgb.cv(
  params = params_final,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 1
)

best_iteration <- cv_model_final$best_iteration
if(is.null(best_iteration) || length(best_iteration) == 0 || best_iteration < 10) {
  best_iteration <- which.max(cv_model_final$evaluation_log$test_auc_mean)
  cat("Using iteration with best test AUC:", best_iteration, "\n")
}

cat("\nBest iteration:", best_iteration, "\n")
cat("Cross-validation AUC:", round(max(cv_model_final$evaluation_log$test_auc_mean), 4), "\n\n")

# FINAL MODEL TRAIN
xgb_model <- xgb.train(
  params = params_final,
  data = dtrain,
  nrounds = best_iteration,
  verbose = 0
)

# PREDICT
xgb_pred_prob <- predict(xgb_model, dtest)

# find optimal threshold
cat("Optimizing decision threshold...\n")
thresholds <- seq(0.1, 0.9, 0.01)
f1_scores <- numeric(length(thresholds))

for(i in seq_along(thresholds)) {
  t <- thresholds[i]
  pred_temp <- factor(ifelse(xgb_pred_prob > t, "High", "Low"), 
                      levels = c("Low", "High"))
  cm_temp <- confusionMatrix(pred_temp, test_data$is_high_adoption, positive = "High")
  f1_scores[i] <- cm_temp$byClass["F1"]
}

optimal_threshold <- thresholds[which.max(f1_scores)]
max_f1 <- max(f1_scores, na.rm = TRUE)

cat("  Optimal threshold:", optimal_threshold, "\n")
cat("  Expected F1:", round(max_f1, 3), "\n\n")

# apply optimal threshold
xgb_pred <- factor(ifelse(xgb_pred_prob > optimal_threshold, "High", "Low"), 
                   levels = c("Low", "High"))

# EVAL
xgb_cm <- confusionMatrix(xgb_pred, test_data$is_high_adoption, positive = "High")
xgb_roc <- roc(test_data$is_high_adoption, xgb_pred_prob)

cat("XGBoost Confusion Matrix:\n")
print(xgb_cm$table)

brier_score <- mean((xgb_pred_prob - test_label)^2)
cat("1. Brier Score:", round(brier_score, 3), "(lower = better, 0.25 = random)\n")

# store metrics
xgb_metrics <- data.frame(
  Model = "XGBoost",
  Accuracy = xgb_cm$overall["Accuracy"],
  Precision = xgb_cm$byClass["Precision"],
  Recall = xgb_cm$byClass["Sensitivity"],
  F1 = xgb_cm$byClass["F1"],
  AUC = auc(xgb_roc),
  Threshold = optimal_threshold
)

cat("XGBoost Final Performance:\n")
cat("  F1:", round(xgb_metrics$F1, 3), "\n")
cat("  AUC:", round(xgb_metrics$AUC, 3), "\n")
cat("  Precision:", round(xgb_metrics$Precision, 3), "\n")
cat("  Recall:", round(xgb_metrics$Recall, 3), "\n\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

comparison <- rbind(nb_metrics, xgb_metrics)
rownames(comparison) <- NULL
print(round(comparison[, -1], 3))
cat("\n")

# ROC
library(ggplot2)

nb_roc_df <- data.frame(
  FPR = 1 - nb_roc$specificities,
  TPR = nb_roc$sensitivities,
  Model = "Naive Bayes"
)

xgb_roc_df <- data.frame(
  FPR = 1 - xgb_roc$specificities,
  TPR = xgb_roc$sensitivities,
  Model = "XGBoost"
)

roc_combined <- rbind(nb_roc_df, xgb_roc_df)

roc_plot <- ggplot(roc_combined, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  scale_color_manual(values = c("Naive Bayes" = "blue", "XGBoost" = "red")) +
  labs(
    title = "ROC Curve Comparison",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = c(0.8, 0.2),
    legend.background = element_blank(),
    legend.key = element_blank()
  ) +
  annotate("text", x = 0.6, y = 0.3, 
           label = paste0("NB AUC = ", round(auc(nb_roc), 3)), 
           color = "blue", size = 4) +
  annotate("text", x = 0.6, y = 0.2, 
           label = paste0("XGB AUC = ", round(auc(xgb_roc), 3)), 
           color = "red", size = 4)

print(roc_plot)


# ============================================================================
# SHAP INTERPRETATION
# ============================================================================

# SHAP on test set
shap_values <- shap.values(xgb_model = xgb_model, 
                           X_train = test_matrix)


shap_long <- shap.prep(shap_contrib = shap_values$shap_score,
                       X_train = test_matrix)

# global feature importance
cat("Top 15 Features (SHAP importance):\n")
shap_importance <- shap_values$mean_shap_score
shap_importance_df <- data.frame(
  Feature = names(shap_importance),
  Importance = shap_importance
) %>%
  arrange(desc(Importance)) %>%
  head(15)

print(shap_importance_df)

# plot

# summary (beeswarm)
shap_summary_plot <- shap.plot.summary(shap_long) + 
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))

print(shap_summary_plot)

library(zip)

files_to_zip <- c(
  "script.R",              
  "huggingface_final.csv"     
)

# Create zip file
zip(
  zipfile = "755273_final.zip",
  files = files_to_zip
)
