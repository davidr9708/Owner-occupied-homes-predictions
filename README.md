Predictive model for Owner-occupied Boston’s homes
================

`{r global_options, echo=FALSE} knitr::opts_chunk$set(fig.path='Figs/')`
=============================================================================================

**Summary:** This project purpose is build a predictive model for
***corrected median value of owner-occupied Boston’s homes in USD
1000’s*** (`cmedv`). Linear regressions, SVD, and neural networks where
used to trained candidates models, the best of each one was compared to
select the final predictive model.

=============================================================================================

## 1. Model’s Purpose

To predict ***corrected median value of owner-occupied Boston’s homes in
USD 1000’s*** (`cmedv`). The model might not be generalizable to some
towns due to the lack of data from some of them. Review session 4.1.1.
in any model’s folder

## 2. Data

The dataset to use is `BostonHousing2` from `mlbench` library. The
outcome is labelled as `cmedv`. There are not missing values.
`{r setup, message=FALSE, echo=FALSE, cache=TRUE} library(mlbench) data("BostonHousing2") summary(BostonHousing2)`
\## 3. Splitting data The resampling method will be ***stratified
cross-validation***. Even though training data will have the same rows
as the `exploring_data`, I decided to name them differently because
training data will have less predictors (selected predictors).

\`\`\`{r data, message=FALSE} library(caret) raw_data \<- BostonHousing2
outcome \<- raw_data$cmedv

set.seed(2156) trainingRows \<- createDataPartition(outcome, p = .8,
list= FALSE)

# Data to explore

exploring_data \<- raw_data\[trainingRows, \]

    ## 4. Exploring data
    Variables' names were divided into different vectors depending on their variable
    type. `numeric_variable` for numeric data and `factor_variable` for factors. `medv` was excluded because it could lead to a loss of performance due to its relationship with `cmedv` (both are almost the same variables).

    ```{r functions, echo=FALSE}
    library(pacman)
    p_load(MASS, tidyverse, caret, corrplot, mlbench, e1071, ggridges)

    variables_type <- function(data){
      numeric_variable <- vector()
      factor_variable <- vector()  
      
      for(name in colnames(data)){
        if(length(class(data[, name])) > 1 &
           class(data[, name])[1] == 'ordered'){
          factor_variable <- c(factor_variable, name)
        }
        else if(class(data[, name]) %in% c('numeric', 'integer')){
          numeric_variable <- c(numeric_variable, name)
        }
        else{
          factor_variable <- c(factor_variable, name)
        }
      }
        return(list('factors' = factor_variable, 'Numeric' = numeric_variable))
      }

`{r variables, echo=FALSE, cache=TRUE} variables        <- variables_type(exploring_data) numeric_variable <- variables$Numeric[variables$Numeric != 'medv'] factor_variable  <- variables$factors len <- length(levels(exploring_data$town))`

`{r print_numeric} print(numeric_variable)`
`{r print_factor} print(factor_variable)`
