## Overview
# Store Item Demand Forecasting Challenge. The purpose of this challenge is to predict 3 months of item sales at different stores.
# The data consists of 5 years worth of store-item sales data having a total of 10 stores with each store having 50 items.
# Throughout this analysis I explore four different models and assess their predictive performance for this forecasting challenge.
# The four models I use are Arima, TBATS, Prophet, and XGBoost Tree.

# explain a little bit about each model choice . . .


## Data Cleaning
# The data from this Kaggle competition comes in great shape. We have access to the date, the store number, the item number, and the number of sales.

# Daily sales do not appear to have a large variance, although there are significantly fewer sales on the 31st day. This could be the case because there are five months with less than 31 days in them.
# From weekly and monthly sales we can see a solid trend. It is also interesting to note that the last week in the year has significantly less sales than any other week in the year. This could be because of the concentration of holidays in the last week of the year (Christams and New Years).
# Yearly sales show an upward trend in sales over time. 
# Sales by store show that some stores perform better than others.
# Sales by time show a more distinct seasonal trend and an overall increase yearly. 

## Feature Engineering
# From the datetime object we can easily create individual day, week, month, and year variables.
# These variables can be used as additional responses when creating the models.
# With models that use seasonality we can recreate these variables using time series objects and setting their corresponding frequnecies.
# For 


## Model Tuning and Performance
# the first model
# the second model
# the third model
# the fourth model

# performance for each model

## Alternative Approaches
# maybe consider doing only 2 models and showing that 2 other models are 
# also feasable options for this analysis


library(forecast)
library(tidyverse)
library(lubridate)
library(parallel)
library(doParallel)
library(prophet)
library(gridExtra)
train <- vroom::vroom("~/Desktop/Kaggle/StoreItemDemand/train.csv")
test <- vroom::vroom("~/Desktop/Kaggle/StoreItemDemand/test.csv")
train$train <- TRUE # add this extra variable to keep track of our separate data sets
test$train <- FALSE
store <- bind_rows(train, test)
samp <- vroom::vroom("~/Desktop/Kaggle/StoreItemDemand/sample_submission.csv")


## Data looks like
## Date, Store num, Item num, Sales, Id
## for training data the date goes from 2013-01-01 to 2017-12-31
## this is for each store and each item number
## i.e. item 1 at store 1 has sales reported for every day from 2013-2017
## next it goes to item 1 at store 2 ... item 1 store 10 ... item 2 store 1 ... etc.

# The submission format is ordered by store then by item 
# i.e. store 1 item 1, store 2 item 1, ..., store 9 item 50, store 10 item 50.

## Feature Engineering

# look for seasonal items including things sold around holidays or fruits/vegetables
#  that are seasonal

# day, month, year
# weekday or weekend
# holidays/holiday season -- or its possible the data is simulated and has no holidays
# in season or not 
# possibly store items that have unusually high sales for that store 

store$item <- factor(store$item)
store$store <- factor(store$store)
store <- store %>% mutate(time=year(date)+yday(date)/365,
                          day=as.factor(day(date)),
                          week=as.factor(week(date)),
                          #weekday=as.factor(wday(date)),
                          month=as.factor(month(date)),
                          #quarter=as.factor(quarter(date)),
                          year=as.factor(year(date)))

## PLOTS ## 
#########
## Total Sales by Day, Week, Month, Year
plot_sales <- function(sales_data, original_data=FALSE) {
  day <- ggplot(data=aggregate(sales~day, sales_data, FUN=sum),
         mapping=aes(x=day, y=sales)) + geom_col() + xlab("day (1-31)") + 
    theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
  
  week <- ggplot(data=aggregate(sales~week, sales_data, FUN=sum),
         mapping=aes(x=week, y=sales)) + geom_col() + xlab("week (1-52)") + 
    theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
  
  month <- ggplot(data=aggregate(sales~month, sales_data, FUN=sum),
         mapping=aes(x=month, y=sales)) + geom_col() 
  
  year <- ggplot(data=aggregate(sales~year, sales_data, FUN=sum),
         mapping=aes(x=year, y=sales)) + geom_col()
  
  store <- ggplot(data=aggregate(sales~store, sales_data, FUN=sum),
         mapping=aes(x=store, y=sales)) + geom_col() 
  
  if (original_data) {
    ## 2016 is a leap year so there is an additional days worth of sales that ends up on the first day in 2017.
    # I will remove this day from the time plot so it doesn't show a spike in sales.
    drop_time <- sales_data %>% filter(day==1, month==1, year==2017) %>% pull(time)
    time <- ggplot(data=aggregate(sales~time, sales_data %>% filter(time!=drop_time, train==TRUE), FUN=sum),
           mapping=aes(x=time, y=sales)) +
      geom_line()
  } else {
    time <- ggplot(data=aggregate(sales~time, sales_data, FUN=sum),
           mapping=aes(x=time, y=sales)) + geom_line()
  }
  grid.arrange(day, week, month, year, store, time, ncol=3)
}

plot_sales(store %>% filter(train==TRUE), original_data=TRUE)
#########

## Daily sales data
# every 7 days is a week
# every 30.4375 days is a month
# every 91.3125 days is a quarter
# every 365.25 days is a year

# for (i in levels(store$item)) {
#   for (j in levels(store$store)) {
#     print(j)
#     print(i)
#   }
# }

## Allow parallel fix for macOS
## WORKAROUND: https://github.com/rstudio/rstudio/issues/6692
## Revert to 'sequential' setup of PSOCK cluster in RStudio Console on macOS and R 4.0.0
if (Sys.getenv("RSTUDIO") == "1" && !nzchar(Sys.getenv("RSTUDIO_TERM")) && 
    Sys.info()["sysname"] == "Darwin" && getRversion() >= "4.0.0") {
  parallel:::setDefaultClusterOptions(setup_strategy = "sequential")
}

# store 1 item 1
# store 2 item 1
# store 3 item 1
# ...
# store 8 item 1
# store 9 item 1
# store 10 item 1

### TBATS model ###
# when using weekly and yearly seasonality:
# private: 13.37716
# public: 14.77270
# when using daily, weekly, monthly, and yearly seasonality:
# private: 13.44081
# public: 14.85055
sales <- c()
for (i in 1) { # for (i in 1:50) {
  for (j in 1) { # for (j in 1:10) {
    print(paste("Store:", j))
    print(paste("Item:", i))
    y <- store %>% filter (store==j, item==i, train==TRUE) %>% pull(sales)
    tbats.mod <- tbats(y, seasonal.periods=c(7, 365.25))
    preds = forecast(tbats.mod, h=90)
    sales <- append(sales, preds$mean)
  }
}

plot(preds)
#sales <- round(sales, 0)
#id <- 0:(length(sales)-1)
#tbats_df <- data.frame(id, sales)
#write.csv(x=tbats_df, row.names=FALSE, file="~/Desktop/sumbission_tbats.2.csv")

### ARIMA model ###
# when using day, week, month, and year
# private: 22.63201
# public: 16.60850
sales <- c()
for (i in 1:1) { # 1:50
  for (j in 1:1) { # 1:10
    print(paste("Store:", j))
    print(paste("Item:", i))    
    y <- store %>% filter (store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(sales)
    day <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(day)
    week <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(week)
    month <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(month)
    year <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(year)
    
    xreg <- data.frame(day, week, month, year)
    xreg <- data.matrix(xreg, rownames.force = NA)
    arima.mod <- Arima(y=y, xreg=xreg) # arima(0,0,0) model
    preds <- forecast(arima.mod, xreg=xreg)
    sales <- append(sales, preds$mean[0:90])
  }
}
plot(preds)
sales <- round(sales, 0)
id <- 0:(length(sales)-1)
arima_df <- data.frame(id, sales)
# write.csv(x=arima_df, row.names=FALSE, file="~/Desktop/sumbission_arima.csv")

### auto.arima model ###
# when using day, week, month, year
# private: 26.85342
# public: 22.95090
sales <- c()
for (i in 1:1) { # 1:50
  for (j in 1:1) { # 1:10
    print(paste("Store:", j))
    print(paste("Item:", i))    
    y <- store %>% filter (store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(sales)
    day <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(day)
    week <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(week)
    month <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(month)
    year <- store %>% filter(store==as.integer(j), item==as.integer(i), train==TRUE) %>% pull(year)
    
    xreg <- data.frame(day, week, month, year)
    xreg <- data.matrix(xreg, rownames.force = NA)
    auto.arima.mod <- auto.arima(y=y, xreg=xreg)
    preds <- forecast(auto.arima.mod, xreg=xreg)
    sales <- append(sales, preds$mean[0:90])
  }
}
plot(preds)
sales <- round(sales, 0)
id <- 0:(length(sales)-1)
autoarima_df <- data.frame(id, sales)
# write.csv(x=autoarima_df, row.names=FALSE, file="~/Desktop/submission_autoarima.csv")


autocl <- parallel::makeCluster(3, setup_strategy = "sequential")
parallel::stopCluster(cl)

library(caret)
library(DataExplorer)
library(plyr)

### XGBoost Tree Model ###
# when using day, week, month, year
# private score: 33.91296
# public score: 32.85681
#tr = trainControl(method="repeatedcv", number=3, repeats=3, search="grid", verbose=TRUE)
xgbTreeGrid <- expand.grid(nrounds = 100,
                       max_depth = 50,
                       eta = 0.3,
                       gamma = 1,
                       colsample_bytree = 0.5,
                       min_child_weight = 1,
                       subsample = 1) 
sales <- c()
for (i in 1) {
  for (j in 1) {
    print(j)
    print(i)
    df <- subset(store, store==as.integer(j) & item==as.integer(i), select=c("train", "sales", "day", "week", "month", "year"))
    xgbTree.mod <- train(form=sales~day+week+month+year,
                             data = df %>% filter(train==TRUE),
                             method = "xgbTree",
                             metric = "RMSE",
                             #trControl=tr,
                             tuneGrid = xgbTreeGrid)
    preds <- predict(xgbTree.mod, newdata=df %>% filter(train==FALSE))
    sales <- append(sales, preds)
  }
}

plot(preds)
#sales <- round(sales, 0)
#id <- 0:(length(sales)-1)
#xgbTree_df <- data.frame(id, sales)
#write.csv(x=df, row.names=FALSE, file="~/Desktop/sumbission_xgbTree.csv")

###### Prophet Model #######
# using day, week, month, year
# private score: 14.07198
# public score: 16.81641
sales <- c()
for (i in 1) { 
  for (j in 1) {
    print(i)
    print(j)
    train <- store %>% filter(item == i & store == j, train==TRUE) %>% mutate(ds = date, y = sales) %>% select(ds, y, day, week, month, year)
    test <- store %>% filter(item == i & store == j, train==FALSE) %>% mutate(ds = date, y = sales) %>% select(ds, y, day, week, month, year)
    m <- prophet(daily.seasonality = TRUE)
    m <- add_regressor(m, "day")
    m <- add_regressor(m, "week")
    m <- add_regressor(m, "month")
    m <- add_regressor(m, "year")
    m <- fit.prophet(m, train)
    forecast <- predict(m, test)
    preds <- forecast$yhat
    sales <- append(sales, preds)
  }
}
plot(preds)
sales <- round(sales, 0)
id <- 0:(length(sales)-1)
df <- data.frame(id, sales)
#write.csv(x=df, row.names=FALSE, file="~/Desktop/submission.4.csv")



### Plot the Predictions ###
arima <- vroom::vroom("~/Desktop/arima_day:week:month:year.csv", delim=",")
new_arima <- merge(test, arima, by=c("id"))
new_arima <- bind_rows(train, new_arima)

autoarima <- vroom::vroom("~/Desktop/submission_autoarima.csv", delim=",")
new_autoarima <- merge(test, autoarima, by=c("id"))
new_autoarima <- bind_rows(train, new_autoarima)

tbats <- vroom::vroom("~/Desktop/submission_tbats.csv", delim=",")
new_tbats <- merge(test, tbats, by=c("id"))
new_tbats <- bind_rows(train, new_tbats)

xgbTree <- vroom::vroom("~/Desktop/submission_xgbTree.csv", delim=",")
new_xgbTree <- merge(test, xgbTree, by=c("id"))
new_xgbTree <- bind_rows(train, new_xgbTree)

prophet <- vroom::vroom("~/Desktop/submission_prophet", delim=",")
new_prophet <- merge(test, prophet, by=c("id"))
new_prophet <- bind_rows(train, new_prophet)

plot_sales(new_arima)
plot_sales(new_autoarima)
plot_sales(new_tbats)
plot_sales(new_xgbTree)
plot_sales(new_prophet)

