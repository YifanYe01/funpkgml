
# create the function: using logistic regression to solve the binary classification task
# use this function to compare different miss-classification errors on test data using the following re-sampling strategies:
# 1) 3*10 CV
# 2) 10*3 CV
# 3) a singe holdout split with 90% training data

# the input variable task is the binary classification problem data

#' Compare miss-classification errors on binary classification task using different re-sampling strategies respectively
#'
#' @param task the binary classification machine learning task
#'
#' @return the resulting bar plot showing miss-classification errors when using different resampling strategies
#' @export
#'
#' @import tidyverse
#' @import mlr3
#' @import mlr3benchmark
#' @import mlr3learners
#' @import ggplot2
#'
#' @examples
#' rsmp_mce(tsk("german_credit"))

rsmp_mce <- function(task){
  # create the logistic regression learner
  learner <- lrn("classif.log_reg")

  # create different resampling strategies
  set.seed(123)
  resampling_3x10_cv <- rsmp("repeated_cv", folds = 10, repeats = 3)
  resampling_10x3_cv <- rsmp("repeated_cv", folds = 3, repeats = 10)
  resampling_holdout <- rsmp("holdout", ratio = 0.9)

  # evaluate results from models with different resampling strategies
  result_3x10_cv <- resample(task, learner, resampling_3x10_cv, store_models = TRUE)
  result_10x3_cv <- resample(task, learner, resampling_10x3_cv, store_models = TRUE)
  result_holdout <- resample(task, learner, resampling_holdout, store_models = TRUE)

  # aggregate results over splits (MCE is default) and create a data frame
  rsmp_strategy = c("3x10-CV", "10x3-CV", "Holdout Split 90% Training Data")
  mce_results = c(result_3x10_cv$aggregate(), result_10x3_cv$aggregate(), result_holdout$aggregate())

  mce_df <- data.frame(
    rsmp_strategy,
    mce_results
  )

  # create a bar plot corresponding to the MCE on different resampling strategy
  rsmp_mce_plot <- ggplot(
    mce_df,
    aes(
      x = rsmp_strategy,
      y = mce_results
    )
  ) +
    geom_bar(stat="identity", fill = "lightgreen") +
    theme_classic() +
    labs(x = "Resampling Strategy",
         y = "Missclassification Error on Test Data")

  return(rsmp_mce_plot)

}
