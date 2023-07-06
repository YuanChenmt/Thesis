

library("forecast")

getwd()
setwd()

USA<-read.csv(file = "Divx_fltper_5.csv",header = TRUE, sep = ",")
attach(USA)
library(dplyr)

#split train and test set

usa_train_mx<-filter(USA, Year<=2005, Year>=1966)$mx
usa_test_mx<-filter(USA, Year<=2015, Year>2005)$mx

usa2006mx<-filter(USA, Year==2006)$mx
usa2007mx<-filter(USA, Year==2007)$mx
usa2008mx<-filter(USA, Year==2008)$mx
usa2009mx<-filter(USA, Year==2009)$mx
usa2010mx<-filter(USA, Year==2010)$mx
usa2011mx<-filter(USA, Year==2011)$mx
usa2012mx<-filter(USA, Year==2012)$mx
usa2013mx<-filter(USA, Year==2013)$mx
usa2014mx<-filter(USA, Year==2014)$mx
usa2015mx<-filter(USA, Year==2015)$mx

# compute alpha
logm<-log(usa_train_mx)
M<-matrix(logm,nrow=40,ncol=24,byrow = TRUE)
alpha<-colMeans(M)
alpha

# compute beta and kappa with svd
for (j in 1:24) {
  M[,j]<-M[,j]-alpha[j]
}
d<-svd(M,1,1)
beta<-d$v/sum(d$v)
kappa<-d$u*sum(d$v)*d$d[1]

# predict kappa with arima
library(forecast)
time_series <- auto.arima(kappa, allowdrift= TRUE)
time_series

#plot kappa
pre<-forecast(time_series,10,level=c(80,85,95))
plot(pre, xlab = "Time", ylab = "kappa")

mx_new<-matrix(usa_test_mx)
pre_kappa<- matrix(pre$mean)

#predict mortality rate with kappa
m2006<-matrix(exp(alpha+beta*pre_kappa[1]))
m2007<-matrix(exp(alpha+beta*pre_kappa[2]))
m2008<-matrix(exp(alpha+beta*pre_kappa[3]))
m2009<-matrix(exp(alpha+beta*pre_kappa[4]))
m2010<-matrix(exp(alpha+beta*pre_kappa[5]))
m2011<-matrix(exp(alpha+beta*pre_kappa[6]))
m2012<-matrix(exp(alpha+beta*pre_kappa[7]))
m2013<-matrix(exp(alpha+beta*pre_kappa[8]))
m2014<-matrix(exp(alpha+beta*pre_kappa[9]))
m2015<-matrix(exp(alpha+beta*pre_kappa[10]))
m_total<-rbind(m2006,m2007,m2008,m2009,m2010,m2011,m2012,m2013,m2014,m2015)

# compute mae and rmse
lcmae<-0
for(i in 1:240){
  lcmae<- lcmae+(abs(mx_new[i]-m_total[i]))
  i<-i+1 
}
lcmae<-lcmae*(1/240)


lcrmse<-0
for(i in 1:240){
  lcrmse<- lcrmse+((mx_new[i]-m_total[i])^2)
  i<-i+1
}
lcrmse<-(lcrmse*(1/240))^0.5

#import lstm kappa
pre_kappa_lstm<-matrix(c(
),nrow=10,ncol=1,byrow=TRUE)
#predict mortality rate with lstm kappa
m1_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[1]))
m2_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[2]))
m3_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[3]))
m4_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[4]))
m5_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[5]))
m6_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[6]))
m7_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[7]))
m8_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[8]))
m9_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[9]))
m10_lstm<-matrix(exp(alpha+beta*pre_kappa_lstm[10]))
m_total_lstm<-rbind(m1_lstm,m2_lstm,m3_lstm,m4_lstm,m5_lstm,m6_lstm,m7_lstm,m8_lstm,
                    m9_lstm,m10_lstm)

#import bilstm kappa 
pre_kappa_bilstm<-matrix(c(),nrow=10,ncol=1,byrow=TRUE)
#predict mortality rate with lstm kappa
m1_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[1]))
m2_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[2]))
m3_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[3]))
m4_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[4]))
m5_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[5]))
m6_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[6]))
m7_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[7]))
m8_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[8]))
m9_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[9]))
m10_bilstm<-matrix(exp(alpha+beta*pre_kappa_bilstm[10]))
m_total_bilstm<-rbind(m1_bilstm,m2_bilstm,m3_bilstm,m4_bilstm,m5_bilstm,m6_bilstm,m7_bilstm,m8_bilstm,
                      m9_bilstm,m10_bilstm)

#import GRU kappa
pre_kappa_gru<-matrix(c(),nrow=10,ncol=1,byrow=TRUE)

m1_gru<-matrix(exp(alpha+beta*pre_kappa_gru[1]))
m2_gru<-matrix(exp(alpha+beta*pre_kappa_gru[2]))
m3_gru<-matrix(exp(alpha+beta*pre_kappa_gru[3]))
m4_gru<-matrix(exp(alpha+beta*pre_kappa_gru[4]))
m5_gru<-matrix(exp(alpha+beta*pre_kappa_gru[5]))
m6_gru<-matrix(exp(alpha+beta*pre_kappa_gru[6]))
m7_gru<-matrix(exp(alpha+beta*pre_kappa_gru[7]))
m8_gru<-matrix(exp(alpha+beta*pre_kappa_gru[8]))
m9_gru<-matrix(exp(alpha+beta*pre_kappa_gru[9]))
m10_gru<-matrix(exp(alpha+beta*pre_kappa_gru[10]))
m_total_gru<-rbind(m1_gru,m2_gru,m3_gru,m4_gru,m5_gru,m6_gru,m7_gru,m8_gru,m9_gru,
                   m10_gru)
#import biGRU kappa
pre_kappa_bigru<-matrix(c( ),nrow=10,ncol=1,byrow=TRUE)

m1_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[1]))
m2_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[2]))
m3_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[3]))
m4_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[4]))
m5_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[5]))
m6_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[6]))
m7_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[7]))
m8_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[8]))
m9_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[9]))
m10_bigru<-matrix(exp(alpha+beta*pre_kappa_bigru[10]))
m_total_bigru<-rbind(m1_bigru,m2_bigru,m3_bigru,m4_bigru,m5_bigru,m6_bigru,m7_bigru,m8_bigru,m9_bigru,
                     m10_bigru)

# compute mae and rmse for dl models
lstmmae<-0
for(i in 1:240){
  lstmmae<- lstmmae+(abs(mx_new[i]-m_total_lstm[i]))
  i<-i+1 
}
lstmmae<-lstmmae*(1/240)

lstmrmse<-0
for(i in 1:240){
  lstmrmse<- lstmrmse+((mx_new[i]-m_total_lstm[i])^2)
  i<-i+1
}
lstmrmse<-(lstmrmse*(1/240))^0.5

bilstmmae<-0
for(i in 1:240){
  bilstmmae<- bilstmmae+(abs(mx_new[i]-m_total_bilstm[i]))
  i<-i+1 
}
bilstmmae<-bilstmmae*(1/240)

bilstmrmse<-0
for(i in 1:240){
  bilstmrmse<- bilstmrmse+((mx_new[i]-m_total_bilstm[i])^2)
  i<-i+1
}
bilstmrmse<-(bilstmrmse*(1/240))^0.5

grumae<-0
for(i in 1:240){
  grumae<- grumae+(abs(mx_new[i]-m_total_gru[i]))
  i<-i+1 
}
grumae<-grumae*(1/240)


grurmse<-0
for(i in 1:240){
  grurmse<- grurmse+((mx_new[i]-m_total_gru[i])^2)
  i<-i+1
}
grurmse<-(grurmse*(1/240))^0.5

bigrumae<-0
for(i in 1:240){
  bigrumae<- bigrumae+(abs(mx_new[i]-m_total_bigru[i]))
  i<-i+1 
}
bigrumae<-bigrumae*(1/240)


bigrurmse<-0
for(i in 1:240){
  bigrurmse<- bigrurmse+((mx_new[i]-m_total_bigru[i])^2)
  i<-i+1
}
bigrurmse<-(bigrurmse*(1/240))^0.5
lcmae
lcrmse
lstmmae
lstmrmse
grumae
grurmse
bilstmmae
bilstmrmse
bigrumae
bigrurmse
options(scipen = 200)  