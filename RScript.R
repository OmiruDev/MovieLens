## Omiru Ranshan
## MovieLens Project 
## HarvardX: Data Science - Capstone Project
## https://github.com/OmiruDev/MovieLens

#################################################
# MovieLens Rating Prediction Project Code 
################################################

#### Introduction ####

## Dataset ##

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes for loading required package: tidyverse and package caret
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

# Set longer timeout and use alternative download method
options(timeout = 600) # Increase timeout to 10 minutes

# Try multiple download methods
dl <- tempfile()
tryCatch({
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, mode = "wb", method = "libcurl")
}, error = function(e) {
  tryCatch({
    download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, mode = "wb", method = "libcurl")
  }, error = function(e) {
    tryCatch({
      download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, mode = "wb", method = "auto")
    }, error = function(e) {
      stop("Failed to download the dataset after multiple attempts. Please check your internet connection.")
    })
  })
})

# Read ratings data
ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

# Read movies data
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# The Validation subset will be 10% of the MovieLens data.
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx subset:
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#### Methods and Analysis ####

### Data Analysis ###

# Head
head(edx) %>%
  print.data.frame()

# Total unique movies and users
summary(edx)

# Number of unique movies and users in the edx dataset 
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

# Ratings distribution - FIXED
edx %>%
  ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  scale_y_continuous(breaks = seq(0, 3000000, 500000)) +
  ggtitle("Rating distribution") +
  xlab("Rating") +
  ylab("Count")

# Plot number of ratings per movie - FIXED
edx %>%
  count(movieId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

# Table 20 movies rated only once
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

# Plot number of ratings given by users - FIXED
edx %>%
  count(userId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")

# Plot mean movie ratings given by users - FIXED
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(x = b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  theme_light()

### Modelling Approach ###

## Average movie rating model ##

# Compute the dataset's mean rating
mu <- mean(edx$rating)
mu

# Test results based on simple prediction
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

# Check results
# Save prediction in data frame
rmse_results <- data_frame(method = "Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

## Movie effect model ##

# Simple model taking into account the movie effect b_i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Plot number of movies with the computed b_i - FIXED
movie_avgs %>% 
  ggplot(aes(x = b_i)) +
  geom_histogram(bins = 10, color = "black", fill = "lightblue") +
  xlab("Movie effect (b_i)") +
  ylab("Number of movies") +
  ggtitle("Distribution of movie effects")

# Test and save rmse results 
predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  .$b_i
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie effect model",  
                                     RMSE = model_1_rmse))
# Check results
rmse_results %>% knitr::kable()

## Movie and user effect model ##

user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Plot penalty term user effect - FIXED
user_avgs %>% 
  ggplot(aes(x = b_u)) +
  geom_histogram(bins = 30, color = "black", fill = "lightgreen") +
  xlab("User effect (b_u)") +
  ylab("Number of users") +
  ggtitle("Distribution of user effects")

# Test and save rmse results 
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie and user effect model",  
                                     RMSE = model_2_rmse))

# Check result
rmse_results %>% knitr::kable()

## Regularized movie and user effect model ##

# lambda is a tuning parameter
lambdas <- seq(0, 10, 0.25)

# For each lambda, find b_i & b_u, followed by rating prediction & testing
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot rmses vs lambdas to select the optimal lambda - FIXED
qplot(lambdas, rmses, main = "RMSE vs Lambda", xlab = "Lambda", ylab = "RMSE")

# The optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized movie and user effect model",  
                                     RMSE = min(rmses)))

# Check result
rmse_results %>% knitr::kable()

#### Results ####
# RMSE results overview
rmse_results %>% knitr::kable()

#### Appendix ####
print("Operating System:")
version