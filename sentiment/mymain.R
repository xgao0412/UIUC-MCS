
if (!require("pROC")) install.packages("pROC")
if (!require("data.table")) install.packages("data.table") 
if (!require("text2vec")) install.packages("text2vec")
if (!require("glmnet")) install.packages("glmnet")

library(data.table)
library(text2vec)

all = as.data.frame(fread("data.tsv",
                          encoding='Latin-1'))
all$review = gsub('<.*?>', ' ', all$review)
all$review = gsub('[[:digit:]]+', ' ', all$review)
splits = read.table("splits.csv", header = T)



train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]

ids_train = train$new_id
it_train = itoken(train[,'review'],
                  tolower,
                  word_tokenizer,
                  ids = ids_train,
                  progressbar = F)

ids_test = test$new_id
it_test = itoken(test[,'review'],
                 tolower,
                 word_tokenizer,
                 ids = ids_test,
                 progressbar = F)

id_df = read.csv('vocab.txt',header = F)
tmp = gsub("_","_",id_df[,1]) 

it_tmp = itoken(tmp,
                tolower,
                word_tokenizer,
                progressbar = F)

vocab = create_vocabulary(it_tmp,
                          ngram = c(1L, 2L) )

vectorizer = vocab_vectorizer(vocab)

dtm_train = create_dtm(it_train, vectorizer)
dtm_test = create_dtm(it_test, vectorizer)

library(glmnet)
set.seed(500)
NFOLDS = 10
mycv = cv.glmnet(x=dtm_train, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)
myfit = glmnet(x=dtm_train, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)
library(pROC)
logit_pred = predict(myfit, dtm_test, type = "response")

readr::write_csv(data.frame('new_id'=test$new_id,'prob'=logit_pred), 'mysubmission.txt',col_names=F)