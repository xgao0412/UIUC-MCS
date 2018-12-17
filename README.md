# University of Illinois, Master of Computer Science program

This repo is a summary of statistical and machine learning projects completed throughout University of Illinois Master of Computer Science
program. 

## Course: Practical Statistic learning
#### This course explains svm, logistic regression and other modeling concepts based on the book The Elements of Statistical Learning
### project1 Ames house price prediction 
This project is to predict the Ames Iowa house price based on housing data. Since there are many categorical variables describing the quality of house features, the major work is focusing on converting them to numbers. I find success in reducing the levels to avoid more
features and prevent overfitting. Finally a xgboost model is trained using RandomizedSearchCV, which is faster by sampling parameters.

[Find the details here](https://github.com/xgao0412/UIUC-MCS/tree/master/ames_house_price)

### project2 : lendingclub default prediction
Lendingclub is the world's largest peer-to-peer lending platform. The goal of this project is to build a machine learning model to predict 
the probability that a loan will default. The data size is 1.8GB with 1.6 million rows and 150 variables. Features with 30% missing data 
are dropped.I choose those that would be available to potential investors from the remaining features. Then feature engineering technique 
is applied by summarizing statistics and visualizing features against loan status. I find success using lightgbm classifier since the training is faster than xgboost. A pipeline using mean imputer, standardization along with the classifier is trained using gridsearch. The final log loss achieved is 0.447.

One important thing to note here is both xgboost and lightgbm are based on decision trees. xgboost splits trees level wise or depth wise, this leads to increase complexity and may lead to overfitting. While Lightgbm splits trees leaf wise and can reduce more loss and achieve better result.

[Find the details here](https://github.com/xgao0412/UIUC-MCS/tree/master/lendingclub)

### project3 : Walmart sales forecast
The goal of this project is to build a model that can forecast on a two month basis. In the begining, only 1 year data is provided. Then when the new data comes in, the model will combine the new data and forecast the next two months sales. This project is done by R.The tslm is used to fit linear models to time series including trend and seasonality components. When I have 2 years data, stlf is used to forcast on seasonal adjusted data and add back the seasonal components.
label: online learning

[Find the details here](https://github.com/xgao0412/UIUC-MCS/tree/master/walmart_sales)

### project4 : Movie review sentiment classifier
The goal of this project is to build a model that takes movie review as input and predict sentiment of the text content. I used fread function to get rid of the exceptions when dealing with tsv file. Then I used regular expression to remove punctuations and numbers
A dictionary is built using text2vec package, a word and bigram vector and docment term matrix are created. Apply two sample t-test to two sentiment groups. The feature with higher t-statistics will be selected. High magnitude of t-statistics means for certain feature, the mean of positive sentiment group is close to negative group, thus we can acquire features with more predictive power. Finally, I use 10 folds cross validated of ridge regression to find the lambda value that produces the minimum value of auc. Then a ridge regression model is built based on that. This model achieves above 0.96 in prediction accuracy.

[Find the details here](https://github.com/xgao0412/UIUC-MCS/tree/master/sentiment)

## Course: data visualization
#### This course explains D3, javascript, html techniques. I end up building an interactive visualization website to help users select wines.

[Find the details here](https://github.com/xgao0412/Wine_selection)

## Course: cloud computing application
#### This course explains hadoop, spark, mapreduce.
### project: Fake Review Detection
This project is to build a model to detect potential fake reviews in tripadvisor. 

[Find the details here](https://github.com/xgao0412/Fake_review_detection)
