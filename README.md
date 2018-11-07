# UIUC-MCS

This repo is a summary of statistical and machine learning projects completed throughout University of Illinois Master of Computer Science
program. 

## Practical Statistic learning
### Ames house price prediction 
Feature engineering technique is applied, convert categorical variables to

###lendingclub default prediction
Lendingclub is the world's largest peer-to-peer lending platform. The goal of this project is to build a machine learning model to predict 
the probability that a loan will default. The data size is 1.8GB with 1.6 million rows and 150 variables. Features with 30% missing data 
are dropped. Choose those that would be available to potential investors from the remaining features. Then feature engineering technique 
is applied by summarizing statistics and visualizing features against loan status. Then I find success using lightgbm classifier since it
is trained faster than xgboost. A pipeline using mean imputer, standardization is trained using gridsearch. The final log loss achieved 
is 0.447.
