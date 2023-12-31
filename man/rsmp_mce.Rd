% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rsmp_mce.R
\name{rsmp_mce}
\alias{rsmp_mce}
\title{Compare miss-classification errors on binary classification task using different re-sampling strategies respectively}
\usage{
rsmp_mce(task)
}
\arguments{
\item{task}{the binary classification machine learning task}
}
\value{
the resulting bar plot showing miss-classification errors when using different resampling strategies
}
\description{
Compare miss-classification errors on binary classification task using different re-sampling strategies respectively
}
\examples{
rsmp_mce(tsk("german_credit"))
}
