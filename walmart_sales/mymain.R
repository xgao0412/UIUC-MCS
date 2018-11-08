

# load necessary packages
if (!require("lubridate")) install.packages("lubridate")
if (!require("fpp2")) install.packages("fpp2")
if (!require("reshape")) install.packages("reshape")
if (!require("plyr")) install.packages("plyr")
if (!require("tidyverse")) install.packages("tidyverse")

library(lubridate)
library(fpp2)
library(reshape)
library(plyr)
library(tidyverse)



mypredict = function(){
        fold_file <- paste0('fold_', t, '.csv')
        new_test <- readr::read_csv(fold_file)
        
        all.stores = unique(test$Store)
        num.stores = length(all.stores)
        train.dates = unique(train$Date)
        num.train.dates = length(train.dates)
        train.frame = data.frame(Date=rep(train.dates, num.stores),
                                 Store=rep(all.stores, each=num.train.dates))
        
        preprocess.svd = function(train, n.comp){
                train[is.na(train)] = 0
                z = svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
                s = diag(z$d[1:n.comp])
                train[, 2:ncol(train)] = z$u %*% s %*% t(z$v)
                train
        }
        
        n.comp = 12 # keep first 12 components
        
        all.dept = unique(train$Dept)
        
        for (d in all.dept){ 
                
                tr.d = train.frame
                tr.d = join(tr.d, train[train$Dept==d, c('Store','Date','Weekly_Sales')])  # perform a left join.
                tr.d = cast(tr.d, Date ~ Store)  # row is Date, col is Store, entries are the sales
                tr.d[is.na(tr.d)]=0
                
                test.dates = unique(new_test$Date)
                num.test.dates = length(test.dates)
                forecast.frame = data.frame(Date=rep(test.dates, num.stores),
                                            Store=rep(all.stores, each=num.test.dates))
                
                
                
                fc.d = forecast.frame
                fc.d$Weekly_Sales = 0
                fc.d = cast(fc.d, Date ~ Store)  # similar as tr.d
                fc.d1 = fc.d
                fc.d2 = fc.d
                
                horizon = nrow(fc.d)  # number of steps ahead to forecast
                if (t<=6){
                        for(j in 2:ncol(tr.d)){ # loop over stores
                                
                                
                                s = ts(tr.d[,j],frequency = 52)
                                fit = tslm(s~trend + season)
                                fc.d[,j] = as.numeric(forecast(fit,h=horizon)$mean)
                                fc.d1[,j] = as.numeric(naive(s,h=horizon)$mean)
                                fc.d2[,j] = as.numeric(meanf(s,h=horizon)$mean)
                                
                                
                                cd = melt(fc.d, id = c('Date','Store'))
                                ce = join(test[which(test$Dept==d & test$Date %in% test.dates), c('Store','Date','Weekly_Pred1')],cd)
                                test[which(test$Dept==d & test$Date %in% cd$Date & test$Store %in% cd$Store), 'Weekly_Pred1']<<-ce$value
                                
                                cd1 = melt(fc.d1, id = c('Date','Store'))
                                ce1 = join(test[which(test$Dept==d & test$Date %in% test.dates), c('Store','Date','Weekly_Pred2')],cd1)
                                test[which(test$Dept==d & test$Date %in% cd1$Date & test$Store %in% cd1$Store), 'Weekly_Pred2']<<-ce1$value
                                
                                cd2 = melt(fc.d2, id = c('Date','Store'))
                                ce2 = join(test[which(test$Dept==d & test$Date %in% test.dates), c('Store','Date','Weekly_Pred3')],cd2)
                                test[which(test$Dept==d & test$Date %in% cd2$Date & test$Store %in% cd2$Store), 'Weekly_Pred3']<<-ce2$value
                                
                        }
                } else{
                        for(j in 2:ncol(tr.d)){ 
                                
                                s = ts(tr.d[, j], frequency = 52)  
                                fc.d1[,j] = as.numeric(naive(s,h=horizon)$mean)
                                fc.d2[,j] = as.numeric(meanf(s,h=horizon)$mean)
                                
                                # apply SVD for tr.d
                                tr.d = preprocess.svd(tr.d, n.comp)
                                s = ts(tr.d[, j], frequency = 52)
                                fc = stlf(s, h=horizon,method='arima')
                                pred = as.numeric(fc$mean)
                                fc.d[, j] = pred
                                
                                
                                cd = melt(fc.d, id = c('Date','Store'))
                                ce = join(test[which(test$Dept==d & test$Date %in% test.dates), c('Store','Date','Weekly_Pred1')],cd)
                                test[which(test$Dept==d & test$Date %in% cd$Date & test$Store %in% cd$Store), 'Weekly_Pred1']<<-ce$value
                                
                                cd1 = melt(fc.d1, id = c('Date','Store'))
                                ce1 = join(test[which(test$Dept==d & test$Date %in% test.dates), c('Store','Date','Weekly_Pred2')],cd1)
                                test[which(test$Dept==d & test$Date %in% cd1$Date & test$Store %in% cd1$Store), 'Weekly_Pred2']<<-ce1$value
                                
                                cd2 = melt(fc.d2, id = c('Date','Store'))
                                ce2 = join(test[which(test$Dept==d & test$Date %in% test.dates), c('Store','Date','Weekly_Pred3')],cd2)
                                test[which(test$Dept==d & test$Date %in% cd2$Date & test$Store %in% cd2$Store), 'Weekly_Pred3']<<-ce2$value
                                
                        }
                }
        }
        train <<-rbind(train,new_test)
        train[is.na(train)] <- 0
}




