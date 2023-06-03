
library(ggplot2)
library(tidyverse)
library(caret)
library(car)
library(mgcv)
library(Metrics)

data <- read.table("project_data.txt", header = T)
head(data)
# pairs(data)
# pairs(data[c(1:7,16)])
# pairs(data[c(8:15,16)])
# corr <- cor(data)
# abs(cor(data)) > 0.7

data <- data[-5]
data_partition <- createDataPartition(data$Crime, p = 0.8, list = FALSE)
data_train <- data[data_partition, ]
data_valid <- data[-data_partition, ]

# stepwise regression
lm_all <- lm(data = data_train, Crime ~ M+So+Ed+Po1+LF+ 
             MF+Pop+NW+U1+U2+Wealth+Ineq+Prob+Time)
lm_null <- lm(data=data_train, Crime~1)

summary(lm_all)

lm_step <- step(lm_null, scope = list(lower = lm_null, 
                upper = lm_all), direction = "both", trace = 1)
summary(lm_step)

plot(residuals(lm_step), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(lm_step))
qqline(residuals(lm_step))


stud_res_lm <- rstudent(lm_step)
outliers_lm <- which(abs(stud_res_lm) > 2)
data_clean_lm <- data_train[-outliers_lm,]
lm_model_clean <- lm(Crime ~ Po1+Ineq+Ed+M+Prob+U2, data = data_clean_lm)
summary(lm_model_clean)

plot(residuals(lm_model_clean), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(lm_model_clean))
qqline(residuals(lm_model_clean))


# select based simple linear model here
# the residuals show slight heteroscedasticity, variance could be decreasing

# also since it is count data, try poisson model 
# find best fit poisson model here

pois_all1 <- glm(data = data_clean_lm, Crime ~ M+So+Ed+Po1+LF+ 
                 MF+Pop+NW+U1+U2+Wealth+Ineq+Prob+Time, family = poisson(link = "log"))
summary(pois_all1)
plot(residuals(pois_all1), xlab = "Fitted Values", ylab = "Residuals")

pois_step1 <- step(pois_all1, direction = "both", trace = TRUE)
summary(pois_step1)
plot(residuals(pois_step1), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(pois_step1))
qqline(residuals(pois_step1))

anova(pois_step1,test="Chisq")
sigma2 <- sum(residuals(pois_step1,type="pearson")^2)/(pois_step1$df.resid)
1-pchisq(sigma2*pois_step1$df.residual,pois_step1$df.residual)
pois_step1$df.resid*sigma2/qchisq(.95,pois_step1$df.resid)

summary(pois_step1,dispersion=sigma2)

pois_all2 <- glm(data = data_clean_lm, Crime ~ M+So+Ed+Po1+U2+Ineq+Prob, family = quasipoisson(link="log"))
summary(pois_all2)
plot(residuals(pois_all2), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(pois_all2))
qqline(residuals(pois_all2))

anova(pois_all1, pois_all2, test = "Chisq")

gamma_all <- glm(data = data_clean_lm, Crime ~  M+So+Ed+Po1+U2+Ineq+Prob, family = Gamma())
summary(gamma_all)
plot(residuals(gamma_all), xlab = "Fitted Values", ylab = "Residuals")

gamma_all_step <- step(gamma_all, direction = "both", trace = TRUE)
summary(gamma_all_step)
plot(residuals(gamma_all_step), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(gamma_all_step))
qqline(residuals(gamma_all_step))

# gamma_all_identity <- glm(data = data, Crime ~ M+So+Ed+Po1+LF+ 
#                    MF+Pop+NW+U1+U2+Wealth+Ineq+Prob+Time, family = Gamma(link="identity"))
# summary(gamma_all_identity)
# plot(residuals(gamma_all_identity))

quasi1 <- glm(data = data_clean_lm, Crime ~M+So+Ed+Po1+U2+Ineq+Prob, 
              family = quasi(link="log", variance = "mu^2"))
summary(quasi1)
plot(residuals(quasi1), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(quasi1))
qqline(residuals(quasi1))

quasi2 <- glm(data = data_clean_lm, Crime ~M+So+Ed+Po1+U2+Ineq+Prob, 
              family = quasi(link="inverse", variance = "mu^2"))
summary(quasi2)
plot(residuals(quasi2), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(quasi2))
qqline(residuals(quasi2))

## GAM models
# gam_poisson <- gam(Crime ~ s(M,k=5)+So+s(Ed,k=5)+s(Po1,k=5)+s(LF,k=5)+s(MF,k=5)+s(Pop,k=5)
#                    +s(NW,k=5)+s(U1,k=5)+s(U2,k=5)+s(Wealth,k=5)+s(Ineq,k=5)+s(Prob,k=5)+s(Time,k=5),
#                    family = poisson(link = "log"), data = data_clean_lm)

gam_poisson <- gam(Crime ~ s(M,k=5)+So+s(Ed,k=5)+s(Po1,k=5)+s(U2,k=5)+s(Ineq,k=5)+s(Prob,k=5)+s(Time,k=5),
                   family = poisson(link = "log"), data = data_clean_lm)
summary(gam_poisson)
plot(residuals(gam_poisson), xlab = "Fitted Values", ylab = "Residuals")
qqnorm(residuals(gam_poisson))
qqline(residuals(gam_poisson))

# prediction
valid_x <-data_valid[1:14]
valid_y <- data_valid[15]
pois_glm_pred <- predict(pois_all2, newdata = valid_x, type = "response")
pois_gam_pred <- predict(gam_poisson, newdata = valid_x, type = "response")

glm_rmse <- rmse(valid_y$Crime, pois_glm_pred)
gam_rmse <- rmse(valid_y$Crime, pois_gam_pred)
glm_rmse
gam_rmse

glm_mape <- mape(valid_y$Crime, pois_glm_pred)
gam_mape <- mape(valid_y$Crime, pois_gam_pred)
glm_mape
gam_mape





