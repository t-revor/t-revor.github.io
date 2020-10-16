library(tidyverse)
library(lubridate)
library(xts)
library(imputeTS)
library(vars)
library(MLmetrics)
library(collections)
library(data.table)
library("IRdisplay")
library(repr)
options(warn=-1)

#------------------------------------               

loader = function()
{
  # Dictionary definition
  D=dict()
  
  
  # Historical Series
  hs = read.csv("historical_series_tampons_NEW.csv",
                sep = ";",
                dec = ",",
                header=TRUE)
  
  hs$Data.Rif = as.Date(hs$Data.Rif , format = "%d/%m/%y")
  hs$Standard.Units = as.numeric(hs$Standard.Units)
  hs = data.table(hs)
  products = hs[,sum(Standard.Units),by=floor_date(Data.Rif,"week")][,1]
  colnames(products)[1]="date"
  
  for (i in 1:22)
  {
    product = hs[grep(paste("Product ",i,"$",sep=""), hs$Products), ]
    product = product[,sum(Standard.Units),by=floor_date(Data.Rif,"weeks")]
    colnames(product)[1]="date"
    colnames(product)[2]=paste("p",i,sep="")
    products = merge(products,product,all=TRUE)
    products[is.na(products)] = 0
    productss=products %>% pull(i+1)
    productss = ts(productss,
                   start = decimal_date(ymd("2017-01-01")),
                   frequency = 365.25/7,
                   names = )
    D$set(paste("p",i,sep=""),productss)
  }
  
  
  # Promo Series
  promo = read.csv("hpromo.csv",
                   sep = ",",
                   dec = ".",
                   header=TRUE)
  promo$date = as.Date(promo$date)
  
  for (i in colnames(promo)[-1])
  {
    hpromo = ts(promo[1:117,] %>% pull(i),
                start = decimal_date(ymd("2017-01-01")),
                frequency = 365.25/7)
    D$set(i,hpromo)
  }
  
  
  # Advertising Series
  advertising = read.csv("tampax_GRPs.csv",
                         sep = ";",
                         dec = ",",
                         header=TRUE)
  colnames(advertising)[1]="date"
  colnames(advertising)[2]="advertising"
  advertising$date = as.Date(advertising$date , format = "%d/%m/%y")
  advertising[is.na(advertising)] = 0
  advertising = ts(advertising %>% pull('advertising'),
                   start = decimal_date(ymd("2017-01-01")),
                   frequency = 365.25/7)
  
  D$set("advertising",advertising)
  
  
  # Sellout Series
  sellout = read.csv("volumes_sell_out_tampons.csv",
                     sep = ";",
                     dec = ",",
                     header=TRUE)
  colnames(sellout)[1]="date"
  colnames(sellout)[3]="sellout"
  sellout$date = as.Date(sellout$date , format = "%d/%m/%y")
  sellout = sellout[grep("TAMPONI FATER", sellout$Market.Products), ]
  sellout = ts(sellout %>% pull('sellout'),
               start = decimal_date(ymd("2017-01-01")),
               frequency = 365.25/7)
  
  D$set("sellout",sellout)
  
  
  # Gtrends Series
  gtrends = read.csv("gtrends_R.csv",
                     sep = ",",
                     dec = ".",
                     skip = 1,
                     header=TRUE)
  colnames(gtrends)[1]="date"
  colnames(gtrends)[2]="gtrends"
  gtrends$date = as.Date(gtrends$date)
  gtrends = ts(gtrends %>% pull('gtrends'),
               start = decimal_date(ymd("2017-01-01")),
               frequency = 365.25/7)
  
  D$set("gtrends",gtrends)
  
  
  return(D)
}

#------------------------------------   

variable_selection = function()
{
  if (test_set == TRUE){
    endogenous_train = list()
    endogenous_test = list()
    exogenous_train = list()
    exogenous_test = list()
    j=1
    for (i in endogenous){
      endogenous_train[[j]] = D$get(i)[1:(117-forecast_weeks)]
      endogenous_test[[j]] = D$get(i)[(117-forecast_weeks+1):117]
      j=j+1
    }
    j=1
    for (i in exogenous){
      exogenous_train[[j]] = D$get(i)[1:(117-forecast_weeks)]
      exogenous_test[[j]] = D$get(i)[(117-forecast_weeks+1):117]
      j=j+1
    }
    
  } else {
    endogenous_train = list()
    endogenous_test = list()
    exogenous_train = list()
    exogenous_test = list()
    j=1
    for (i in endogenous){
      endogenous_train[[j]] = D$get(i)[1:117]
      endogenous_test[[j]] = D$get(i)[1:117]
      j=j+1
    }
    j=1
    for (i in exogenous){
      exogenous_train[[j]] = D$get(i)[1:117]
      exogenous_test[[j]] = D$get(i)[118:139]
      j=j+1
    }
  }
  
  names(endogenous_train) = endogenous
  names(endogenous_test) = endogenous
  names(exogenous_train) = exogenous
  names(exogenous_test) = exogenous
  
  variables = list(endogenous_train = endogenous_train,
                   endogenous_test = endogenous_test,
                   exogenous_train = exogenous_train,
                   exogenous_test = exogenous_test)
  return(variables)
}

#------------------------------------   

model_selection = function(types)
{
  VARselect(y =  data.frame(variables$endogenous_train),
            exogen = data.frame(variables$exogenous_train), 
            lag.max = 10,
            type = types)
  
}

#------------------------------------                  

model_fit = function(order,types)
{
  model = VAR(y = data.frame(variables$endogenous_train),
              exogen = data.frame(variables$exogenous_train),
              p = order,
              type = types)
  
  return(model)
}

#------------------------------------   

model_forecast = function(conf)
{
  if (test_set == FALSE){
    forecast = predict(model,
                       dumvar = data.frame(variables$exogenous_test),
                       n.ahead = 22,
                       ci = conf)
  } else {
    forecast = predict(model,
                       dumvar = data.frame(variables$exogenous_test),
                       n.ahead = lengths(variables$endogenous_test)[[1]],
                       ci = conf)
    plot(variables$endogenous_test[[1]],type="l")
    lines(forecast[[1]][[1]][,1],col = "red")
    legend("topright", legend = c("Test sample","Forecast"),col=c("black","red"),lty=1:2, cex=0.8)
    print(paste("MSE for current product is: ",round(MSE(variables$endogenous_test[[1]],forecast[[1]][[1]][,1]))))
    
  }
  return(forecast)
}

#------------------------------------                  

model_diagnostic = function()
{
  options(repr.plot.res = 180)
  plot(model)
}



statistical_tests = function(){
  print(serial.test(model,lags.pt = 12,type="PT.asymptotic"))
  print(arch.test(model,lags.multi=12,multivariate.only=TRUE))
}

