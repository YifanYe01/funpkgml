library("mlr3")
library("mlbench")
library("mlr3learners")
library("tidyverse")

# create the function: using linear regression to learn the relationship between independent variable x and the target y
# test different train-test data split ratio from 10% to 90% with the step 10% and compare the mean squared error when using different train-test data sample splits with boxplot

# x is the independent variable in the linear regression model
# y is the target variable


#' Evaluate the performance of different training test split ratios on linear regression model
#'
#' @param x series of numerical values, independent variable in the linear regression model
#' @param y series of numerical values, the target variable in the prediction model
#'
#' @return the resulting box-plot of relationship between share of training data and the mean squared error of prediction
#' @export
#'
#' @examples eval_split_ratio(Boston$dis, Boston$nox)
#'
eval_split_ratio <- function(x, y){

  df <- data.frame(x, y)

  # define task
  task <- mlr3::TaskRegr$new("Test",backend = df, target = "y")

  # define train-test splits
  # here: using 10-fold cross validation, split share from 10% to 90% using 10% step
  repetitions <- 1:10
  split_ratios <- seq(0.1, 0.9, by = 0.1)

  # create resampling objects with holdout strategy, using lapply for efficient computation
  split_strategies <- lapply(split_ratios, function(i) mlr3::rsmp("holdout", ratio = i))

  # train linear learner
  learner <- mlr3::lrn("regr.lm")
  learner$train(task)

  # train linear learners and predict in one step (remember to set a seed)
  set.seed(123)
  results <- list()
  for (i in repetitions) {
    results[[i]] <- lapply(split_strategies, function(i) mlr3::resample(task, learner, i))
  }

  # compute errors in double loop over repetitions and split ratios
  errors <- lapply(
    repetitions,
    function(i) sapply(results[[i]], function(j) j$score()$regr.mse))

  # assemble everything in data.frame and convert to long format for plotting
  errors_df <- as.data.frame(do.call(cbind, errors))
  errors_df$split_ratios <- split_ratios
  errors_df_long <- reshape2::melt(errors_df, id.vars = "split_ratios")
  names(errors_df_long)[2:3] <- c("repetition", "mse")

  # plot errors vs split ratio
  spl_err_plot <- ggplot(
    errors_df_long,
    aes(
      x = as.factor(split_ratios),
      y = mse
    )
  ) +
    geom_boxplot(fill = "lightyellow") +
    theme_classic() +
    labs(x = "share of training samples",
         y = "average MSE")

  return(spl_err_plot)
}

