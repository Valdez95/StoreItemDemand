
library(forecast)
library(tidyverse)
library(lubridate)
library(parallel)
library(doParallel)
library(prophet)

train <- vroom::vroom("~/Desktop/Kaggle/StoreItemDemand/train.csv")
train$train <- TRUE
test <- vroom::vroom("~/Desktop/Kaggle/StoreItemDemand/test.csv")
test$train <- FALSE
store <- bind_rows(train, test)
samp <- vroom::vroom("~/Desktop/Kaggle/StoreItemDemand/sample_submission.csv")

# with(store, table(item, store))

## Data looks like
## Date, Store num, Item num, Sales, Id
## for training data the date goes from 2013-01-01 to 2017-12-31
## this is for each store and each item number
## i.e. item 1 at store 1 has sales reported for every day from 2013-2017
## next it goes to item 1 at store 2 ... item 1 store 10 ... item 2 store 1 ... etc.

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
                          wday=as.factor(wday(date, label=TRUE)),
                          week=as.factor(week(date)),
                          month=as.factor(month(date)),
                          quarter=as.factor(quarter(date)),
                          year=as.factor(year(date)))

## Sales by Month
ggplot(data=store %>% filter(item==1),
       mapping=aes(x=month(date) %>% as.factor(), y=sales)) + 
  geom_boxplot()

## Sales of item by store
ggplot(data=store %>% filter(item==17),
       mapping=aes(x=date, y=sales, color=as.factor(store))) +
  geom_line()

ggplot(data=store %>% filter(item==17, store==7),
       mapping=aes(x=time, y=sales)) +
  geom_line() + geom_smooth(method='lm')

## linear model with time and month
mt.lm <- lm(sales~month+time, data=(store %>% filter(item==17, store==7)))
fit.vals <- fitted(mt.lm)
plot(x=(store %>% filter(item==17, store==7) %>% pull(time)),
     y=store %>% filter(item==17, store==7) %>% pull(sales), type="l")
lines((store %>% filter(item==17, store==7, !is.na(sales)) %>% pull(time)),
      fit.vals, col="red", lwd=2)

## Weekend effect, holiday effect
ggplot(data=store %>% filter(item==1),
       mapping=aes(x=wday(date, label=TRUE) %>% as.factor(), y=sales)) + 
  geom_boxplot()


## 
ggplot(data=store %>% filter(item==17),
       mapping=aes(x=as.factor(store), y=sales)) +
  geom_boxplot()

## Coleman Plot
cole <- store %>% group_by(store, item) %>%
  summarize(totSales=sum(sales))
ggplot(data=cole, mapping=aes(x=as.factor(item), y=totSales)) +
  geom_col() + facet_wrap(~as.factor(store))

# do a lot of feature engineering
# try to fit a single model for each item in each store 
# then you would be fitting a lot of different models, one for each item and for each store

## SARIMA() model 
# 6 different knobs in a sarima model
# p, d, q (order, lower case) control day to day correlation
    # as they get biggeer, higher correlation comes from day to day
# P, D, Q (seasonal, upper case) control seasonal 
    # as they gete bigger it reflects correlation from seasonal effects (last year vs this year)

# Arima or auto.arima
    # max.p = 5, look at all models from a p with values 0-5 
# y <- store %>% filter(item==1, store==1, train==FALSE)

## Set up variables for modeling
randStore = sample(1:10, 1)
randItem = sample(1:50, 1)

y <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>%
  pull(sales) %>% ts(data=., start=1, frequency=365)
xreg <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(month) %>% ts(data=., start=1, frequency=365)

time <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(time) %>% ts(data=., start=1, frequency=365)
day <- store %>% filter(item==randItem, store==randItem, train==TRUE) %>% 
  pull(day) %>% ts(data=., start=1, frequency=365)
week <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(week) %>% ts(data=., start=1, frequency=365)
month <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(month) %>% ts(data=., start=1, frequency=365)
year <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(year) %>% ts(data=., start=1, frequency=365)
wday <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(wday) %>% ts(data=., start=1, frequency=365)
quarter <- store %>% filter(item==randItem, store==randStore, train==TRUE) %>% 
  pull(quarter) %>% ts(data=., start=1, frequency=365)
xreg <- data.frame(time, day, week, month, year, wday, quarter)
xreg <- data.matrix(xreg5, rownames.force = NA)

## TBATS model
# elapsed time = 19
system.time({tbats.model = tbats(y=y, seasonal.periods=365, num.cores = 3)})
preds = forecast(tbats.model, h=365)
plot(preds)
#preds$mean - (df$sales[44911:45000])

## ARIMA model
system.time({arima.mod <- Arima(y=y, xreg=xreg)})
# elapsed time = .007
preds = forecast(arima.mod, xreg=xreg)
plot(preds)

## auto.arima model
# elapsed time = 10
system.time({a.arima.mod <- auto.arima(y=y, xreg=xreg)})
preds = forecast(a.arima.mod, xreg=xreg)
plot(preds)

# frequency represents seasons, 365 time periods this season 
#  (365 days in a year and we have yearly data)
#xreg and y must have same num rows 

# Arima(y=y, order=c(2,2,2), seasonal=c(0,0,0))
xreg <- store %>% filter(item==1, store==1, train==FALSE) %>% 
  pull(month) %>% ts(data=., start=1, frequency=12)
preds <- forecast(arima.mod, xreg=xreg)

plot(preds)

# highly recommend that we do this by store by item
# for item 1:50 for store 1:10 build a model and predict
sales <- c()
system.time({
  for (i in levels(store$store)) {
    print(i)
    for (j in levels(store$item)) {
      print(j)
      y <- store %>% filter (store==as.integer(i), item==as.integer(j), train==TRUE) %>% pull(sales) %>% ts(data=., start=1, frequency=365)
      #month <- store %>% filter(store==as.integer(i), item==as.integer(j), train==FALSE) %>% pull(month) %>% ts(data=., start=1, frequency=30)
      #arima.mod <- auto.arima(y=y, max.p=2, max.q=2, max.P=1, max.Q=1)
      #arima.mod <- Arima(y=y)
      tbats.mod <- tbats(y)
      preds <- forecast(arima.mod, h=90)
      sales <- append(sales, preds$mean)
    }
  }
})

sales <- round(sales, 0)
id <- 0:(45000-1)
df <- data.frame(id, sales)

write.csv(x=df, row.names=FALSE, file="~/Desktop/sumbission.csv")


# build a lot of explanatory variables (Xs)
# feed that to the arima model

# facebook prophet forecasting uses some clever things... generalized additive models
# you could use some of their stuff to do 

#TBATS -- maybe do something like this ? ? ? 
#flexible seasonal modeling framework
#multiple season time series object
#tbats(y, seasonal.periods=c(7, 365)) ... something like that
#forecast using tbats model 90 days ahead
#forecast(tbats.model, h=90)


### XGBoost Model

library(caret)
library(vroom) 
library(DataExplorer)
library(plyr)
library(h2o)
library(tidyverse)


df <- subset(store, item==1 & store==1, select=date:year)
df$day <- df %>% pull(day) %>% ts(data=., start=1, frequency=365)
str(df$day)

df %>% filter(train==FALSE)

# fit gbm_h2o model using default parameters
# additional features used are year, month, day, hour, and weekday
h2o.init()
tr = trainControl(method="repeatedcv", number=3, repeats=3, search="grid")
store.model <- train(form=sales~day,
                             data = df,
                             method = "gbm_h2o",
                             metric = "RMSE")
                             #trControl=tr)
preds <- predict(store.model, newdata=df %>% filter(train==FALSE))
submission <- data.frame(id=0:44999, sales=preds)

x<-store %>% filter(store==1, item==1, train=="TRUE")
y<-bike %>% filter(id=="train")


#####
df <- subset(store, item==1 & store==1, select=date:year)
df$day <- df %>% pull(day) %>% ts(data=., start=1, frequency=365)
df$week <- df %>% pull(week) %>% ts(data=., start=1, frequency=365)
model <- train(form=sales~time+day+week,
               data = df %>% filter(train==TRUE),
               #method = "gbm_h2o",
               method="xgbLinear",
               gamm=0.5,
               metric = "RMSE")
preds <- predict(model, newdata=df %>% filter(train==FALSE))
sales <- append(sales, preds)

      
##### XGB Linear Model ######

## Allow parallel fix for macOS
## WORKAROUND: https://github.com/rstudio/rstudio/issues/6692
## Revert to 'sequential' setup of PSOCK cluster in RStudio Console on macOS and R 4.0.0
if (Sys.getenv("RSTUDIO") == "1" && !nzchar(Sys.getenv("RSTUDIO_TERM")) && 
    Sys.info()["sysname"] == "Darwin" && getRversion() >= "4.0.0") {
  parallel:::setDefaultClusterOptions(setup_strategy = "sequential")
}
cl <- parallel::makeCluster(3, setup_strategy = "sequential")

#system.time({
sales <- c()
for (i in levels(store$store)) {
  print(i)
  currStore = as.integer(i)
  for (j in levels(store$item)) {
    system.time({
    print(j)
    currItem = as.integer(j)
    
    df <- subset(store, item==currItem & store==currStore, select=date:year)
    df$day <- df %>% pull(day) %>% ts(data=., start=1, frequency=365)
    df$week <- df %>% pull(week) %>% ts(data=., start=1, frequency=365)
    model <- train(form=sales~time+day+week,
                   data = df %>% filter(train==TRUE),
                   method = "xgbLinear",
                   gamma=0.5,
                   metric = "RMSE")
    preds <- predict(model, newdata=df %>% filter(train==FALSE))
    sales <- append(sales, preds)
    })
  }
}
#})

parallel::stopCluster(cl)

#write.csv(x=submission, row.names=FALSE, file="~/Desktop/Kaggle/KaggleBikeShare/gbm_h2oSubmit")

###### Prophet Model

sales <- c()
for (i in (1:10)) {
  print(i)
  currStore = as.integer(i)
  for (j in 1:50) {
    print(j)
    currItem = as.integer(j)

    df <- subset(store, item==currItem & store==currStore, select=c(date,sales))
    stats=data.frame(y=log1p(df$sales) ,ds=df$date)
    stats=aggregate(stats$y,by=list(stats$ds),FUN=sum)
    colnames(stats)<- c("ds","y")
    model_prophet = prophet(stats, daily.seasonality = TRUE)
    summary(model_prophet)
    future = make_future_dataframe(model_prophet, periods = 90)
    forecast = predict(model_prophet, future)
    preds <- xts::last(forecast[, "yhat"],90)
    sales <- append(sales, preds)
  }
}

id <- 0:(45000-1)
df <- data.frame(id, sales)

write.csv(x=df, row.names=FALSE, file="~/Desktop/submission.4.csv")

