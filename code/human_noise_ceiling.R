# Clear environment
rm(list = ls())

# Packages
library(tidyverse)
library(psych) 
library(data.table)
library(bruceR)
library(corrplot)

# 1) helper functions
#from individual ratings, build a rating table (video x rater) for one dimension
get_rating_table <- function(data, dimension,
                             video_col = "video_name",
                             rater_col = "subjectID", 
                             score_col = "likert_response",
                             question_col = "question_name") {
  data %>%
    filter(.data[[question_col]] == dimension) %>%
    select(all_of(c(video_col, rater_col, score_col))) %>%
    pivot_wider(names_from = all_of(rater_col),
                values_from = all_of(score_col))
}

# Row signature: concatenated rater columns to identify non-NA blocks
block_signature <- function(mat, colnames_vec) {
  apply(mat, 1, function(x) {
    ids <- colnames_vec[which(!is.na(x))]
    paste(ids, collapse = "|")
  })
}

# reliability calculation
calculate_reliability <- function(X, method = "cronbach") {
  if (method == "cronbach") {
    return(psych::alpha(X,check.keys = FALSE)$total$raw_alpha)
  } else if (method == "icc") {
    return(psych::ICC(X)$results[1, "ICC"]) 
  } else if (method == "split") {
    n_cols <- ncol(X)
    mid <- ceiling(n_cols/2)
    r <- cor(rowMeans(X[, 1:mid]), rowMeans(X[, (mid+1):n_cols]))
    return((2*r)/(1+r)) # Spearman-Brown correction
  }
  return(NA)
}

# 2) Core function
summarize_blocks_cronbach <- function(individual_rating, model_annotation, dimension_name) {
  # 0. Convert individual ratings to rating table for the given dimension
  rating_table <- get_rating_table(individual_rating, dimension = dimension_name)
  
  # 1. Get numeric matrix and detect blocks
  num <- rating_table %>% select(where(is.numeric)) %>% as.matrix()
  if (ncol(num) == 0) stop("No numeric rater columns found in rating_table.")
  vids <- rating_table$video_name
  
  # 2. Group rows by block signature
  sig <- block_signature(num, colnames(num))
  blocks <- split(seq_len(nrow(num)), sig)
  
  # 3. Process each block
  per_block <- purrr::imap_dfr(blocks, function(row_idx, block_sig) {
    # Get block matrix
    X <- num[row_idx, , drop = FALSE]
    
    # Keep only fully observed columns
    good_cols <- which(colSums(!is.na(X)) == nrow(X))
    X <- X[, good_cols, drop = FALSE]

    # 3.1 Calculate reliability and ceiling
    alpha <- calculate_reliability(X)
    ceiling <- if (is.finite(alpha) && alpha >= 0) sqrt(alpha) else NA_real_
    
    # 3.2 Calculate correlation with model
    human_mean <- rowMeans(X)
    df_block <- tibble(video_name = vids[row_idx],
                      human_mean = human_mean) %>%
      inner_join(model_annotation, by = "video_name")
    
    r <- cor(df_block$human_mean,
                         df_block[[dimension_name]],
                         method = "pearson")
    
    return(tibble(block_id = block_sig,
           alpha = alpha,
           noise_ceiling = ceiling,
           correlation = r))
  })
  
  # 4. Summarize across blocks
  # Fisher-z transform for correlation aggregation
  valid_corr <- is.finite(per_block$correlation)
  z <- atanh(pmax(pmin(per_block$correlation[valid_corr], 0.999999), -0.999999))
  r_mean <- if (length(z)) tanh(mean(z)) else NA_real_
  
  summary <- tibble(
    alpha_mean = mean(per_block$alpha, na.rm = TRUE),
    ceiling_mean = mean(per_block$noise_ceiling, na.rm = TRUE),
    correlation_mean = r_mean
  )
  
  list(by_block = per_block, summary = summary)
}

validate_ratings <- function(dimension_name, block_sig) {
  test_ <- get_rating_table(individual_rating, dimension_name)
  rater_ids <- as.numeric(strsplit(block_sig, "\\|")[[1]])
  test__ <- test_ %>%
    select(video_name, all_of(as.character(rater_ids))) %>%
    filter(if_all(-video_name, ~!is.na(.))) %>%
    mutate(across(-video_name, as.numeric))
  
  # Calculate reliability metrics
  reliability_mat  <- test__ %>%
    select(-video_name) %>%
    as.matrix() 
  
  cronbach_alpha <- calculate_reliability(reliability_mat, "cronbach")
  icc <- calculate_reliability(reliability_mat, "icc")
  
  print(paste("ICC:", round(icc, 3)))
  print(paste("Cronbach's α:", round(cronbach_alpha, 3)))
  
  # Create correlation plot
  corrplot(cor(reliability_mat, use="pairwise.complete.obs"), 
           tl.cex = 0.8,
           title = paste("Inter-rater Correlations:", dimension_name),
           mar = c(0,0,2,0))
  
  # Calculate correlation between mean human rating and model rating
  mean_rating <- rowMeans(reliability_mat)
  
  video_names <- test__$video_name
  model_ratings <- model_annotation %>%
    filter(video_name %in% video_names) %>%
    select(
      video_name,
      all_of(dimension_name)) %>%
    arrange(match(video_name, video_names)) %>%
    pull(all_of(dimension_name))
  
  cor_result <- cor.test(mean_rating, model_ratings)
  print(paste("Correlation with model ratings:", round(cor_result$estimate, 3),
              "(p =", round(cor_result$p.value, 3), ")"))
  
  #return(reliability_mat)
}

# 3) Main
target_dimension <-"communication"
individual_rating <- import("data/video/human_rating/individual_ratings.csv")%>%
  filter(!question_name %in% c("relation", "cooperation", "dominance")) %>%
  mutate(question_name = case_when(
    question_name == "joint" ~ "joint_action", 
    question_name == "distance" ~ "agent_distance",
    question_name == "communicating" ~ "communication",
    question_name == "object" ~ "transitivity",
    TRUE ~ question_name
  )) %>%
  mutate(likert_response = if_else(question_name == "communication", 
                                  - likert_response, 
                                  likert_response))
model_annotation <- import("data/annotation/openclip/ViT-H-14-378-quickgelu/dfn5b/model_annotation.csv")

res <- summarize_blocks_cronbach(
  individual_rating = individual_rating,
  model_annotation = model_annotation,
  dimension_name = target_dimension
)

# Per-block results:
res$by_block   # block_id | alpha | noise_ceiling | correlation
# Overall summary:
res$summary



dimensions <- c("expanse", "transitivity", "agent_distance", 
                "facingness", "joint_action", "communication", "valence", "arousal")
for(dim in dimensions){
target_dimension <-"communication"
individual_rating <- import("data/video/human_rating/individual_ratings.csv")%>%
  filter(!question_name %in% c("relation", "cooperation", "dominance")) %>%
  mutate(question_name = case_when(
    question_name == "joint" ~ "joint_action", 
    question_name == "distance" ~ "agent_distance",
    question_name == "communicating" ~ "communication",
    question_name == "object" ~ "transitivity",
    TRUE ~ question_name
  )) %>%
  mutate(likert_response = if_else(question_name == "communication", 
                                   - likert_response, 
                                   likert_response))
model_annotation <- import("data/annotation/openclip/ViT-H-14-378-quickgelu/dfn5b/model_annotation.csv")

res <- summarize_blocks_cronbach(
  individual_rating = individual_rating,
  model_annotation = model_annotation,
  dimension_name = target_dimension
)

# Per-block results:
print(dim)
print(res$by_block)   # block_id | alpha | noise_ceiling | correlation
}
# Validate
#validate_ratings(target_dimension, "1830|1837|1843|1866|1880|1892|1896|1899|1916|1921")
#validate_ratings(target_dimension, "1|21|23|62|84|85|87|88|95|101")


# Run analysis for all dimensions
dimensions <- c("expanse", "transitivity", "agent_distance", 
                "facingness", "joint_action", "communication", "valence", "arousal")
#
# no data for "indoor", cause this is rated by experimenters

results_list <- lapply(dimensions, function(dim) {
  res <- summarize_blocks_cronbach(
    individual_rating = individual_rating,
    model_annotation = model_annotation, 
    dimension_name = dim
  )
  
  # Extract relevant metrics from summary
  tibble(
    dimension = dim,
    cronbach_alpha = res$summary$alpha_mean,
    noise_ceiling = res$summary$ceiling_mean,
    correlation = res$summary$correlation_mean
  )
})

# Combine results into single table
results_df <- bind_rows(results_list)

# Print formatted table
print(results_df %>%
  mutate(across(where(is.numeric), ~round(., 3))) %>% 
    rename(cronbach_alpha = alpha_mean,
           noise_ceiling = ceiling_mean,
           correlation = correlation_mean)) 

