library(quantmod)

df_btc <- read.csv('./data/btc_daily_all.csv', stringsAsFactors=F)
head(df_btc, 5)
#         Date    Open    High     Low   Close Adj.Close Volume
# 1 2010-07-17 0.04951 0.04951 0.04951 0.04951   0.04951      0
# 2 2010-07-18 0.04951 0.08585 0.05941 0.08584   0.08584      5
# 3 2010-07-19 0.08584 0.09307 0.07723 0.08080   0.08080     49
# 4 2010-07-20 0.08080 0.08181 0.07426 0.07474   0.07474     20
# 5 2010-07-21 0.07474 0.07921 0.06634 0.07921   0.07921     42

class(df_btc$Date) # [1] "character"
df_btc$Date <- as.Date(df_btc$Date)
class(df_btc$Date) # [1] "Date"

df_prc <- xts(df_btc[, 6], order.by=df_btc[, 1])
class(df_prc) # [1] "xts" "zoo"

plot(df_prc, auto.grid=T, grid.ticks.on='M')
# pic_01.png

head(df_prc, 2)
#               [,1]
# 2010-07-17 0.04951
# 2010-07-18 0.08584
tail(df_prc, 2)
#               [,1]
# 2018-09-06 6515.42
# 2018-09-07 6411.78

# 2010-07-17         0.04951
# 2018-09-07       6411.78
# total return    (6411.78-0.04951)/0.04951 ~ 129503.7

ret_btc_daily <- (df_prc-lag(df_prc))/lag(df_prc) # this
# or equivalently
# ret_btc_daily <- diff(df_prc)/lag(df_prc)

plot(ret_btc_daily, auto.grid=T, grid.ticks.on='M')
# pic_02.png

mean(ret_btc_daily,na.rm=T)*252
# [1] 1.677684
# annualized return ~ 168%

###########################
# ADF test for stationarity
library(tseries)

# a t-test to see if the regression coefficient is 1
adf.test(df_prc)
#         Augmented Dickey-Fuller Test
# data:  df_prc
# Dickey-Fuller = -2.3488, Lag order = 14, p-value = 0.4306
# alternative hypothesis: stationary

df_logprc <- log(df_prc)
logret_btc_daily <- diff(df_logprc)
logret_btc_daily <- logret_btc_daily[2:length(logret_btc_daily)]

adf.test(logret_btc_daily)
# 	Augmented Dickey-Fuller Test
# data:  logret_btc_daily
# Dickey-Fuller = -13.014, Lag order = 14, p-value = 0.01
# alternative hypothesis: stationary
# Warning message:
# In adf.test(logret_btc_daily) : p-value smaller than printed p-value

df_prc_avg20 = rollmean(df_prc, 20, align='center')
plot(df_prc_avg20, auto.grid=T, grid.ticks.on='M')
# pic_03.png
df_prc_avg100 = rollmean(df_prc, 100, align='center')
plot(df_prc_avg100, auto.grid=T, grid.ticks.on='M')
# pic_04.png

head(ret_btc_daily, 2)
#                 [,1]
# 2010-07-17        NA
# 2010-07-18 0.7337912
ret_btc_daily <- ret_btc_daily[2:length(ret_btc_daily)]
head(ret_btc_daily, 2)
#                   [,1]
# 2010-07-18  0.73379115
# 2010-07-19 -0.05871389

ret_btc_avg20 = rollmean(ret_btc_daily, 20, align='center')
plot(ret_btc_avg20, auto.grid=T, grid.ticks.on='M')
# pic_05.png
ret_btc_avg100 = rollmean(ret_btc_daily, 100, align='center')
plot(ret_btc_avg100, auto.grid=T, grid.ticks.on='M')
# pic_06.png


qqnorm(ret_btc_daily)
qqline(ret_btc_daily)
# pic_07.png

adf.test(ret_btc_daily)
# 	Augmented Dickey-Fuller Test
# data:  ret_btc_daily
# Dickey-Fuller = -13.863, Lag order = 14, p-value = 0.01
# alternative hypothesis: stationary
# Warning message:
# In adf.test(ret_btc_daily) : p-value smaller than printed p-value


##############################
###### AR(1), MA(1), ACF #####
###### AutoCorrelations of BTC
#############################

acf(ret_btc_daily, lag.max=200)
# There are some long term patterns exist
# pic_08.png

result_theta <- rollapply(as.vector(ret_btc_daily), width=100, FUN=acf, lag.max=1, 
                          type='correlation', plot=FALSE, align='right')
head(result_theta, 2)
#      acf       type          n.used lag       series        snames
# [1,] Numeric,2 "correlation" 100    Numeric,2 "data[posns]" NULL  
# [2,] Numeric,2 "correlation" 100    Numeric,2 "data[posns]" NULL 
M <- length(result_theta[, 1])
thetas <- numeric(M)
for (i in 1:M) {
  thetas[i] <- result_theta[i,]$acf[2]  # only care the 2nd term
}

thetas
# [1] -1.800531e-01 -2.267575e-01 -1.653201e-01 -1.933586e-01 -1.749205e-01 -4.005488e-02  2.860432e-02
# [8] -2.140020e-03  1.910667e-02  6.599862e-03  3.078703e-03  6.486601e-02  1.640193e-01  6.204253e-02
# ......

vola_btc_daily <- rollapply(ret_btc_daily, width=100, FUN=sd)
plot(vola_btc_daily, auto.grid=T, grid.ticks.on='M')
# pic_09.png

ret_vola_btc_100 <- vola_btc_daily[100:(M+99)]
ret_vola_btc     <- as.vector(ret_vola_btc_100)
ret_autocorr     <- as.vector(thetas)
ret_daily_100    <- as.vector(rollmean(ret_btc_daily, 100, align='right'))

plot(ret_daily_100, ret_vola_btc, col=rgb(0,0,0, alpha=0.2))
# pic_10.png
plot(ret_vola_btc, ret_autocorr, col=rgb(0,0,0,alpha=0.2))
# pic_11.png
ret_daily_100 <- as.xts(ret_daily_100, order.by=index(vola_btc_daily[100:(M+99)]))
plot(ret_daily_100, auto.grid=T, grid.ticks.on='M')
# pic_12.png
plot(ret_vola_btc_100,  auto.grid=T, grid.ticks.on='M')
# pic_13.png


thetas <- as.xts(thetas, order.by=index(vola_btc_daily[100:(M+99)]))
plot(thetas, auto.grid=T, grid.ticks.on='M')
# pic_14.png


#########################################################
##### Forecast

library(forecast)

adf.test(ret_vola_btc_100)  # likely to be stationary
# What if we fit an arima model by calling auto.arima

# 	Augmented Dickey-Fuller Test
# data:  ret_vola_btc_100
# Dickey-Fuller = -3.6424, Lag order = 14, p-value = 0.02858
# alternative hypothesis: stationary

auto.arima(ret_vola_btc_100)  # default kpss test rejects the null hypothesis of stationary time series
# Series: ret_vola_btc_100 
# ARIMA(4,1,0)                    # <-- it chose it for you
# Coefficients:
#          ar1     ar2     ar3     ar4
#       0.0095  0.0258  0.0055  0.1393
# s.e.  0.0185  0.0185  0.0185  0.0185
# 
# sigma^2 estimated as 4.533e-05:  log likelihood=10296.16
# AIC=-20582.31   AICc=-20582.29   BIC=-20552.5

# explain:
# ARIMA(4,1,0) -> 1 = need 1 difference (d parameter)
#              -> p = 4 and q = 0
# but the p-value says it is stationary!!!!!! why!?!?
# because its default method is KPSS test, not adf...

auto.arima(ret_vola_btc_100, test='adf')
# Series: ret_vola_btc_100 
# ARIMA(4,1,0)
# Coefficients:
#          ar1     ar2     ar3     ar4
#       0.0095  0.0258  0.0055  0.1393
# s.e.  0.0185  0.0185  0.0185  0.0185
# 
# sigma^2 estimated as 4.533e-05:  log likelihood=10296.16
# AIC=-20582.31   AICc=-20582.29   BIC=-20552.5

# sample the volatility series to avoid samples using the same raw data
# because our data is 100-day average, but we do vola by day
# so it's always 99% itself

dim(ret_vola_btc_100)
# [1] 2875    1
sample <- seq(100, 2785, 100)
ret_vola_sampled <- ret_vola_btc_100[sample]

auto.arima(ret_vola_sampled, test='adf', trace=T)
#  ARIMA(2,0,2) with non-zero mean : Inf
#  ARIMA(0,0,0) with non-zero mean : -64.84805
#  ARIMA(1,0,0) with non-zero mean : -62.6995
#  ARIMA(0,0,1) with non-zero mean : -62.75421
#  ARIMA(0,0,0) with zero mean     : -49.88286
#  ARIMA(1,0,1) with non-zero mean : -60.1154
# Best model: ARIMA(0,0,0) with non-zero mean 
# 
# Series: ret_vola_sampled 
# ARIMA(0,0,0) with non-zero mean 
# 
# Coefficients:
#       mean
#       0.0635
# s.e.  0.0129
# 
# sigma^2 estimated as 0.004661:  log likelihood=34.67
# AIC=-65.35   AICc=-64.85   BIC=-62.76
# 
# vola <-> AR(0), vola tends not to have autocorrelation
# high vola tend not to come together
# 0.0635 means that today high, tomorror high vola very not likely

auto.arima(ret_vola_sampled, trace=T)
#  ARIMA(2,0,2) with non-zero mean : Inf
#  ARIMA(0,0,0) with non-zero mean : -64.84805
#  ARIMA(1,0,0) with non-zero mean : -62.6995
#  ARIMA(0,0,1) with non-zero mean : -62.75421
#  ARIMA(0,0,0) with zero mean     : -49.88286
#  ARIMA(1,0,1) with non-zero mean : -60.1154
# Best model: ARIMA(0,0,0) with non-zero mean 
# 
# Series: ret_vola_sampled 
# ARIMA(0,0,0) with non-zero mean 
# 
# Coefficients:
#         mean
#       0.0635
# s.e.  0.0129
# 
# sigma^2 estimated as 0.004661:  log likelihood=34.67
# AIC=-65.35   AICc=-64.85   BIC=-62.76

auto.arima(ret_vola_sampled^2, test='adf', trace=T)   # Don't forget Box-Cox!
#  ARIMA(2,0,2) with non-zero mean : Inf
#  ARIMA(0,0,0) with non-zero mean : -115.5926
#  ARIMA(1,0,0) with non-zero mean : -113.0493
#  ARIMA(0,0,1) with non-zero mean : -113.0493
#  ARIMA(0,0,0) with zero mean     : -115.2142
#  ARIMA(1,0,1) with non-zero mean : -110.2746
# Best model: ARIMA(0,0,0) with non-zero mean 
# 
# Series: ret_vola_sampled^2 
# ARIMA(0,0,0) with non-zero mean 
# 
# Coefficients:
#         mean
#       0.0085
# s.e.  0.0050
# 
# sigma^2 estimated as 0.0007116:  log likelihood=60.05
# AIC=-116.09   AICc=-115.59   BIC=-113.5

acf(ret_vola_sampled^2)
# pic_15.png


#######################
# pacf vs acf
#######################

arima(df_prc, order=c(1,1,0))
# arima(x = df_prc, order = c(1, 1, 0))
# Coefficients:
#           ar1
#        0.0640
# s.e.   0.0183
# sigma^2 estimated as 35222:  log likelihood = -19787.96,  aic = 39579.92

arima(df_prc, order=c(1,0,0))
# arima(x = df_prc, order = c(1, 0, 0))
# Coefficients:
#          ar1  intercept
#       0.9984   1397.046
# s.e.  0.0012   1831.071
# sigma^2 estimated as 35335:  log likelihood = -19802.24,  aic = 39610.48

arima(df_prc, order=c(1,1,1))
# arima(x = df_prc, order = c(1, 1, 1))
# Coefficients:
#           ar1     ma1
#       -0.7245  0.7969
# s.e.   0.0544  0.0471
# sigma^2 estimated as 34982:  log likelihood = -19777.79,  aic = 39561.58

fit <- arima(df_prc, order=c(2,1,0))
summary(fit)
# arima(x = df_prc, order = c(2, 1, 0))
# Coefficients:
#          ar1      ar2
#       0.0651  -0.0165
# s.e.  0.0183   0.0183
# sigma^2 estimated as 35212:  log likelihood = -19787.55,  aic = 39581.1
# Training set error measures:
#                    ME     RMSE      MAE      MPE     MAPE     MASE          ACF1
# Training set 2.049947 187.6179 52.55438 0.158755 3.557354 1.002356 -9.230111e-06



####################
####  Forecast
####################

output<-forecast(fit, h=252)
plot(output)
# pic_16.png

names(output)
#  [1] "method"    "model"     "level"     "mean"      "lower"     "upper"     "x"         "series"    "fitted"   
# [10] "residuals"
head(output$lower)
# Time Series:
# Start = 2976 
# End = 2981 
# Frequency = 1 
#           80%      95%
# 2976 6155.160 6028.152
# 2977 6088.554 5902.957
# 2978 6022.230 5793.556
# 2979 5960.046 5694.715
# 2980 5908.062 5614.280
# 2981 5857.081 5537.825
head(output$upper)
# Time Series:
# Start = 2976 
# End = 2981 
# Frequency = 1 
#           80%      95%
# 2976 6635.009 6762.018
# 2977 6789.755 6975.352
# 2978 6886.177 7114.850
# 2979 6962.494 7227.825
# 2980 7017.997 7311.778
# 2981 7063.262 7382.519


fit <- auto.arima(df_prc, test='adf')
summary(fit)
# Series: df_prc 
# ARIMA(1,1,1) 
# 
# Coefficients:
#           ar1     ma1
#       -0.7245  0.7969
# s.e.   0.0544  0.0471
# sigma^2 estimated as 35005:  log likelihood=-19777.79
# AIC=39561.58   AICc=39561.59   BIC=39579.58
# Training set error measures:
#   ME     RMSE      MAE       MPE     MAPE     MASE         ACF1
# Training set 2.062222 187.0023 52.68635 0.1606818 3.575258 1.004873 0.0001974337

fitted.values(fit)
# Time Series:
# Start = 1 
# End = 2975 
# Frequency = 1 
# [1]   0.04946049   0.04970874   0.08812902   0.07862273   0.07604735   0.07846392   0.04904295
# [8]   0.06464899   0.05234061   0.05196102   0.05523394   0.06089959   0.05810403   0.07133012
# ......\
residuals(fit)
# Time Series:
# Start = 1 
# End = 2975 
# Frequency = 1 
# [1]  4.950997e-05  3.613126e-02 -7.329017e-03 -3.882735e-03  3.162652e-03 -2.796392e-02  1.357705e-02
# [8] -1.010899e-02 -1.840614e-03  4.038983e-03  4.766065e-03 -1.999586e-03  1.179597e-02 -8.630119e-03
# ......

# Then doing the forecast

output <- forecast(fit,h=252)
plot(output)
# pic_17.png


############# skip, until nn_model


# predict without fitting a time series model

plot(naive(sp_prc[1:4000],h=200))
# naive is the wrapper of rwf

#############################################
# computing out-of-sample forecast accuracy
#############################################

L <- length(ret_sp500_daily)-5000  # shrink the length to avoid row index error
x <- ret_sp500_daily[1:L]
fit2<-auto.arima(x)
fc <- forecast(fit2, h=200)  # the output would have error if the number of row > 1e4
names(fc)
accuracy(fc, ret_sp500_daily[(L+1):(L+200)])

L    <- length(sp_prc) - 5000
x    <- sp_prc[1:L]

###############
## adf vs kpss
###############

fit.adf <- auto.arima(x,test='adf')
fc.adf  <- forecast(fit.adf, h=200)
accuracy(fc.adf, sp_prc[(L+1):(L+200)])
plot(fc.adf)

fit.kpss <- auto.arima(x)  # using the default kpss test
fc.kpss  <- forecast(fit.kpss, h=200)
accuracy(fc.kpss, sp_prc[(L+1):(L+200)])
plot(fc.kpss)




#################
# nonlinear time series with nnetar, the neural network auto-regressive forecast
# fitting nn-ar time series
#################
# nnetar is from the forecast package
# nnet: a toy of neural network in R, + autoregressive

# run these 2 skipped lines above:
# L    <- length(sp_prc) - 5000
# x    <- sp_prc[1:L]

nn_model <- nnetar(ret_sp500_daily[2:L,1])
nn_model$p     #  34
nn_model$P     #  0
nn_model$size  # 18 = round((34+0+1)/2) neuron size

fitted.nn <- fitted(nn_model)
residuals.nnetar <- residuals(nn_model)
sqrt(mean(residuals.nnetar**2,na.rm=T))  # 0.007333
# [1] 0.007316128
accuracy(nn_model)
#                        ME        RMSE         MAE MPE MAPE      MASE         ACF1
# Training set 1.485031e-05 0.007316128 0.005110433 NaN  Inf 0.6354139 -0.008923458
fc_nn <- forecast(nn_model, h=200)
accuracy(fc_nn, ret_sp500_daily[(L+1):(L+200), 1])
#                        ME        RMSE         MAE      MPE     MAPE      MASE         ACF1
# Training set 1.485031e-05 0.007316128 0.005110433      NaN      Inf 0.6354139 -0.008923458
# Test set     1.453878e-03 0.011427988 0.009165322 92.54326 105.4849 1.1395852           NA

# compared with 

arima_model <- auto.arima(ret_sp500_daily[2:L,1])
accuracy(arima_model)
#                         ME        RMSE         MAE MPE MAPE      MASE          ACF1
# Training set -4.369749e-07 0.008666085 0.005981718 NaN  Inf 0.7437466 -0.0005508739
fc_arima <- forecast(arima_model, h=200)
accuracy(fc_arima, ret_sp500_daily[(L+1):(L+200), 1])
#                         ME        RMSE         MAE      MPE     MAPE      MASE          ACF1
# Training set -4.369749e-07 0.008666085 0.005981718      NaN      Inf 0.7437466 -0.0005508739
# Test set      7.906375e-04 0.011358553 0.009139277 100.1153 100.3107 1.1363468            NA

# neural network model has a worse in-sample performance, also works worsely out-of-sample

