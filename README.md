# University of Illinois, Master of Computer Science program

This repo is a summary of statistical and machine learning projects completed throughout University of Illinois Master of Computer Science
program. 

## Practical Statistic learning
### Ames house price prediction 
This project is to predict the Ames Iowa house price based on housing data. Since there are many categorical variables describing the quality of house features, the major work is focusing on converting them to numbers. I find success in reducing the levels to avoid more
features and prevent overfitting. Finally a xgboost model is trained using RandomizedSearchCV, which is faster by sampling parameters.

[Find the details here](https://github.com/xgao0412/UIUC-MCS/tree/master/ames_house_price)

### lendingclub default prediction
Lendingclub is the world's largest peer-to-peer lending platform. The goal of this project is to build a machine learning model to predict 
the probability that a loan will default. The data size is 1.8GB with 1.6 million rows and 150 variables. Features with 30% missing data 
are dropped.I choose those that would be available to potential investors from the remaining features. Then feature engineering technique 
is applied by summarizing statistics and visualizing features against loan status. I find success using lightgbm classifier since the training is faster than xgboost. A pipeline using mean imputer, standardization along with the classifier is trained using gridsearch. The final log loss achieved is 0.447.

One important thing to note here is both xgboost and lightgbm are based on decision trees. xgboost splits trees level wise or depth wise, this leads to increase complexity and may lead to overfitting. While Lightgbm splits trees leaf wise and can reduce more loss and achieve better result.
