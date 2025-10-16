#%% objective
# examine the video info:
# 1. activity category
# 2. human rating
# 2.1 average rating
# 2.2 individual rating
# 3. original clips (manually)

#%% packages
library(bruceR)
library(tidyverse)
library(ggplot2)

#%% main
# 1. activity category
action_category <- rbind(
  import("data/video/action_category/test_categories.csv"),
  import("data/video/action_category/train_categories.csv")
)

# per category, count the number of videos
action_category_count <- action_category %>%
  group_by(action_categories) %>%
  summarise(n = n())

# plot the number of videos per category
ggplot(action_category_count %>% 
         arrange(desc(n)) %>% 
         slice(1:10), 
       aes(x = reorder(action_categories, n), y = n)) +
  geom_bar(stat = "identity") +
  xlab("Action Categories") +
  ylab("Number of Videos") +
  ggtitle("Action Category Distribution(Top 10)") +
  theme_minimal() +
  coord_flip()

# 2. human rating

# variables not included in the paper: intimacy, dominance, cooperation
delete_variables <- c("intimacy", "dominance", "cooperation")
variable_order  <- c("indoor", "expanse", "transitivity", "agent_distance", 
"facingness", "joint_action", "communication", "valence", "arousal")

# 2.1 average rating
average_rating <- import("data/video/human_rating/ratings.csv") %>%
  rename(agent_distance = `agent distance`) %>% 
  rename(joint_action = `joint action`) %>% 
  select(-all_of(delete_variables))

# a) examine the distribution of each dimension
dimensions <- variable_order

average_rating_long <- average_rating %>%
  pivot_longer(cols = all_of(dimensions),
                     names_to = "dimension",
                     values_to = "value")
ggplot(average_rating_long, aes(x = value)) +
  facet_wrap(~factor(dimension, levels = variable_order), scales = "free") +
  geom_density(fill = "#1B9E77", alpha = 0.3) +
  geom_histogram(aes(y = ..density..), bins = 30, alpha = 0.3,
                fill = "#D95F02") +
  xlab("Rating") +
  ylab("Density") +
  ggtitle("Distribution of Ratings Across Dimensions") +
  theme_classic() +
  scale_x_continuous(limits = c(-0.1, 1.1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))

# b) correlation matrix between dimensions
cor_matrix <- cor(average_rating[, variable_order])

cor_long <- cor_matrix %>%
  as.data.frame() %>%
  mutate(dim1 = rownames(.)) %>%
  pivot_longer(cols = -dim1, 
              names_to = "dim2", 
              values_to = "correlation")

cor_long_filtered <- cor_long %>%
  mutate(
    dim1 = factor(dim1, levels = variable_order),
    dim2 = factor(dim2, levels = variable_order)
  ) %>%
  filter(as.numeric(dim2) < as.numeric(dim1))

ggplot(cor_long_filtered, aes(x = dim2, y = dim1, fill = correlation)) +
  geom_tile() +
  geom_text(aes(label = ifelse(abs(correlation) > 0.1, 
                              sprintf("%.2f", correlation), "")), 
            size = 3) +
  scale_fill_gradient2(low = "#D95F02", high = "#1B9E77", mid = "white",
                      midpoint = 0, limit = c(-1,1)) +
  theme_bruce() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) +
  ggtitle("Correlation Between Dimensions")


# c) pairwise regressions between dimensions
r_squared_matrix <- matrix(0, length(variable_order), length(variable_order))
rownames(r_squared_matrix) <- variable_order
colnames(r_squared_matrix) <- variable_order

for(i in 1:length(variable_order)) {
  for(j in 1:length(variable_order)) {
    if(i != j) {
      # fit linear regression
      model <- lm(average_rating[[variable_order[i]]] ~ average_rating[[variable_order[j]]])
      # get R-squared
      r_squared_matrix[i,j] <- summary(model)$r.squared
    }
  }
}

# convert to long format
r_squared_long <- r_squared_matrix %>%
  as.data.frame() %>%
  mutate(dim1 = rownames(.)) %>%
  pivot_longer(cols = -dim1,
              names_to = "dim2",
              values_to = "r_squared")

r_squared_filtered <- r_squared_long %>%
  mutate(
    dim1 = factor(dim1, levels = variable_order),
    dim2 = factor(dim2, levels = variable_order)
  ) %>%
  filter(as.numeric(dim2) < as.numeric(dim1))
ggplot(r_squared_filtered, aes(x = dim2, y = fct_rev(dim1), fill = r_squared)) +
  geom_tile() +
  geom_text(aes(label = ifelse(r_squared > 0.1,
                              sprintf("%.2f", r_squared), "")),
            size = 3) +
  scale_fill_gradient2(low = "#EEF4FB", high = "#15306A", limit = c(0,0.5),
                      name = expression(R^2)) +
  theme_bruce() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(), axis.title.y = element_blank()) +
  ggtitle("Explained Variance (R²) Between Dimensions")


# 2.2 individual rating
individual_rating <- import("data/video/human_rating/individual_ratings.csv") %>%
  filter(!question_name %in% c("relation", "cooperation", "dominance")) %>%
  mutate(question_name = case_when(
    question_name == "joint" ~ "joint_action",
    question_name == "distance" ~ "agent_distance",
    question_name == "communicating" ~ "communication",
    question_name == "object" ~ "transitivity",
    TRUE ~ question_name
  ))


# a) per participant
rating_summary <- individual_rating %>%
  group_by(subjectID) %>%
  summarise(
    n_videos = n_distinct(video_name),
    n_dimensions = n_distinct(question_name)
  )

# n_dimensions is always 1, each participant rated only one dimension

# b) per video
video_rating_summary <- individual_rating %>%
  group_by(video_name) %>%
  summarise(
    n_dimensions = n_distinct(question_name),
    n_raters = n_distinct(subjectID)
  )

# n_dimensions is always 11, each video is rated on 11 dimension (`indoor` is rated by experimenters)

rating_data <- bind_rows(
  rating_summary %>% 
    mutate(plot_type = "Videos per Participant") %>%
    rename(value = n_videos),
  video_rating_summary %>% 
    mutate(plot_type = "Raters per Video") %>%
    rename(value = n_raters)
)

ggplot(rating_data, aes(x = value)) +
  geom_histogram(binwidth = 1, fill = "#1B9E77", alpha = 0.7) +
  facet_wrap(~plot_type, scales = "free",ncol=1) +
  theme_bruce()+
 scale_x_continuous(breaks = function(x) seq(floor(min(x)), ceiling(max(x)), by = 1)) +
  theme(
    axis.text = element_text(size = 12),
    strip.text = element_text(size = 14),
    strip.background = element_blank()
  ) +
  labs(x = "Count",
       y = "Frequency") +
  ggtitle("Distribution of Videos and Raters")

# c) number of raters for each video-dimension combination
rater_count_by_video_dim <- individual_rating %>%
  group_by(video_name, question_name) %>%
  summarise(n_raters = n_distinct(subjectID)) %>%
  ungroup()

# d) variance of ratings for each dimension-video combination
rating_variance_by_video_dim <- individual_rating %>%
  group_by(video_name, question_name) %>%
  summarise(rating_var = var(likert_response)) %>%
  ungroup()

dim_order <- rating_variance_by_video_dim %>%
  group_by(question_name) %>%
  summarise(mean_variance = mean(rating_var)) %>%
  arrange(desc(mean_variance)) %>%
  pull(question_name)
  
dim_mean_var <- rating_variance_by_video_dim %>%
  group_by(question_name) %>%
  summarise(mean_var = mean(rating_var)) %>%
  ungroup()
  
ggplot(rating_variance_by_video_dim, 
       aes(x = video_name, 
           y = fct_rev(factor(question_name, levels = dim_order)), 
           fill = rating_var)) +
  geom_tile() +
  scale_fill_gradient(low = "#EEF4FB", high = "#AF435D",
                     name = "Rating\nVariance") +
  geom_text(data = dim_mean_var,
            aes(x = length(unique(rating_variance_by_video_dim$video_name)) + 20,
                y = fct_rev(factor(question_name, levels = dim_order)),
                label = sprintf("%.2f", mean_var)),
            color = "black",
            size = 4,
            inherit.aes = FALSE) +
  annotate("text",
           x = length(unique(rating_variance_by_video_dim$video_name)) + 20,
           y = length(dim_order) + 0.7,
           label = "Average",
           size = 4,
           hjust = 0.5) +
  theme_bruce() +
  xlab("Videos") +
  ylab("Dimensions") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank()
        ) + 
  ggtitle("Rater Inconsistency (Video x Dimension)") +
  coord_cartesian(xlim = c(0, length(unique(rating_variance_by_video_dim$video_name)) + 40),
                  ylim = c(1, length(dim_order) + 0.3))

# rater assignment plot based on previous code (c)
ggplot(rater_count_by_video_dim, aes(x = video_name, y = fct_rev(factor(question_name, levels = dim_order)), fill = n_raters)) +
  geom_tile() +
  scale_fill_gradient(low = "#EEF4FB", high = "#15306A",
                      name = "# Raters") +
  theme_bruce() +
  xlab("Videos") +
  ylab("Dimensions") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank()
        # axis.text.y = element_blank()
  )+
  ggtitle("Rater Assignment (Video x Dimension)")


# examine the distribution of variance across dimensions
rating_variance_long <- rating_variance_by_video_dim %>%
  select(question_name, rating_var)

ggplot(rating_variance_long, aes(x = rating_var)) +
  facet_wrap(~factor(question_name, levels = dim_order), ncol=2, scales = "free") +
  geom_density(fill = "#1B9E77", alpha = 0.3) +
  geom_histogram(aes(y = ..density..), bins = 30, alpha = 0.3,
                fill = "#D95F02") +
  xlab("Rating Variance Per Video") +
  ylab("Density") +
  xlim(0, 0.21) +
  ggtitle("Distribution of Rating Variance") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
   plot.title = element_text(face = "bold"))


