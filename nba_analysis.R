library(dplyr)
library(MASS)
library(e1071)
library(nnet)
library(class)
library(reshape2)
library(lattice)
library(readr)
library(ggplot2)
library(vtable)
library(heatmaply)
library(insight)
library(tidyverse)
library(broom)
library(car)
library(randomForest)
library(ranger)
library(gbm)


## Read Data 
games <- read.table(file= 'C:\\Users\\nsien\\OneDrive\\Documents\\GT Prep\\ISYE 7406\\nba_project\\nba_game_data_proj_v2.csv', sep = ",", header = TRUE)
head(games)
dim(games)

## Split to training and testing subset 
flag <- sort(sample(dim(games)[1],dim(games)[1]*.25, replace = FALSE))
gamestrain <- games[-flag,]
gamestest <- games[flag,]
nrow(games)
nrow(gamestrain)
## Extra the true response value for training and testing data
y1    <- gamestrain$HOME_TEAM_WINS
y2    <- gamestest$HOME_TEAM_WINS



# determine correlation between data points
#using the full dataset, explore data and extract most correlated values
corr_mat <- round(cor(games),2)
corr_mat
mean_corr <- mean(abs(corr_mat[,-ncol(games)]))

# include mpg01 to ensure high-corr based analysis keeps the predictor
high_corr <- abs(corr_mat[,1]) > mean_corr
high_corr

# and graph
heatmaply_cor(x = cor(games), xlab = "Features", 
              ylab = "Features", k_col = 2, k_row = 2)

dev.off()

# initialize train and test error comparisons
TrainErr <- NULL
TestErr <- NULL


# baseline model- logistic binomial regression

### Method 4: binomial logisitic regression
mod5 <- glm(HOME_TEAM_WINS ~., family = binomial(link="logit"), data=gamestrain);  
summary(mod5)
## Training Error  of binomial logisitic regression
pred_probs <- predict(mod4, gamestrain[, -ncol(gamestrain)], type = 'response')
print(pred_probs)
mod4 <- stepAIC(mod5, direction = 'both', trace = FALSE)
summary(mod4)
# determining the optimal cutoff value
best_cutoff <- NULL
best_err <- 1

# Loop through possible cutoff values, picking the optimal value on the train set
for (cutoff in seq(0.1, 0.9, by = 0.05)) {
  # Convert predicted probabilities to predicted classes using cutoff
  cutoff_pred <- ifelse(pred_probs >= cutoff, 1, 0)
  err <- mean( cutoff_pred != y1)
  # Update best cutoff if err is less than previous best
  if (err < best_err) {
    print(paste('hit at ', cutoff, ", best error: ", err))
    best_err <- err
    best_cutoff <- cutoff
  }
}
pred4 <- if_else(pred_probs >= best_cutoff, 1, 0)

TrainErr <- c(TrainErr, mean( pred4 != gamestrain$HOME_TEAM_WINS))
TrainErr
## Testing Error of binomial logisitic regression
testpred4 <- predict(mod4,gamestest[, -ncol(gamestrain)], type = 'response')
testpred4 <- if_else(testpred4 >= best_cutoff, 1, 0)
TestErr <- c(TestErr, mean( testpred4 != gamestest$HOME_TEAM_WINS) )
TestErr



# assumption of logistic regression
# working with
# selecting only numeric and continuous predictors
games_analysis_home <- gamestrain[, 1:13]
games_analysis_away <- gamestrain[, 14:26]
games_predictors <- colnames(games_analysis)
# home
# associate logit and predictors
games_analysis_home <- games_analysis_home %>%
  mutate(logit = log(pred_probs/(1-pred_probs))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
# graph
ggplot(games_analysis_home, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  facet_wrap(~predictors, scales = "free_y")

# away
# associate logit and predictors
games_analysis_away <- games_analysis_away %>%
  mutate(logit = log(pred_probs/(1-pred_probs))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
# graph
ggplot(games_analysis_away, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  facet_wrap(~predictors, scales = "free_y")

# cooks distance (4/N)
threshold <- 4/dim(gamestrain)[1]
plot(mod5, which = 4)
abline(h=threshold, col='red')
cooks_distances <- cooks.distance(mod4)
exceed_threshold <- which(cooks_distances > threshold*2)
length(exceed_threshold)
nrow(gamestrain)
# VIF multicollinearity calc
vif_tab <- vif(mod4)
names(vif_tab)
df <- data.frame(Variable = names(vif_tab), Value = as.numeric(vif_tab))
ggplot(df, aes(x = Variable, y = Value)) +
  geom_bar(stat = "identity") +
  labs(x = "Variable", y = "Variable Inflation Factor", title = "VIF by Predictor", color = "Variable") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




# baseline method- k- nearest numbers

best_k <- NULL
best_err <- 1

# Loop through possible k values, picking the optimal value on the highly correlated train set
error_mat <- NULL
for (kk in seq(1,41, by = 2)) {
  # Convert predicted probabilities to predicted classes using cutoff
  # new data
  error_row <- NULL
  xnew <- gamestrain[, -ncol(games)]
  xnew2 <- gamestest[, -ncol(games)]
  # predictions
  kpred_train <- knn(gamestrain[, -ncol(games)], xnew, gamestrain[, ncol(games)], k=kk)
  kpred_test <- knn(gamestrain[, -ncol(games)], xnew2, gamestrain[, ncol(games)], k=kk)
  # error
  err_train <- mean( kpred_train != gamestrain$HOME_TEAM_WINS)
  err_test <- mean( kpred_test != gamestest$HOME_TEAM_WINS)
  # combine error
  error_row <- cbind(kk, err_train, err_test)
  error_mat <- rbind(error_mat, error_row)
  # Update best cutoff if err is less than previous best
  if (err_test < best_err) {
    print(paste('hit at', kk, ", best error:", err_test))
    best_err <- err_test
    best_k <- kk
  }
}
error_mat
# k=1 performed best on training data, but will also pick k=3 and k=5 for a quick validation 
# of training error given k=1 may be overfitted 
kpred_train_best <- knn(gamestrain[,-ncol(games)], xnew, gamestrain[, ncol(games)], k=best_k)
TrainErr <- c(TrainErr, mean( kpred_train_best != gamestrain$HOME_TEAM_WINS))
TrainErr
## Testing Error of knn
xnew2 <- gamestest[,-ncol(games)]
kpred_test_best <- knn(gamestrain[,-ncol(games)], xnew2, gamestrain[, ncol(games)], k=best_k)
TestErr <- c(TestErr, mean( kpred_test_best != gamestest$HOME_TEAM_WINS) )
TestErr
# Lets plot errors
error_mat <- data.frame(error_mat)
plot(error_mat$kk, error_mat$err_train, type = "o", col = "blue", xlab = "k Values", ylab = "Error Rate", ylim = c(0, max(error_mat$err_train, error_mat$err_test)), main = "Training and Error Rates vs. k Values")
points(error_mat$kk, error_mat$err_test, type = "o", col = "red")
legend("topright", legend = c("Training Rate", "Error Rate"), col = c("blue", "red"), lty = 1:1, cex = 0.8)




## Build Random Forest with the default parameters
## It can be 'classification', 'regression', or 'unsupervised'

# valid number of mtrys
floor(sqrt(ncol(games)))

rf1 <- randomForest(as.factor(HOME_TEAM_WINS) ~., data=gamestrain, 
                    importance=TRUE)

## Check Important variables
importance(rf1)
## There are two types of importance measure 
##  (1=mean decrease in accuracy, 
##   2= mean decrease in node impurity)
importance(rf1, type=2)
varImpPlot(rf1, main= "Variable Importance for Random Forest")

## Prediction on the testing data set
rf.pred = predict(rf1, gamestest, type='class')
rf_table <- table(rf.pred, y2)
rf_acc <- (rf_table[1] + rf_table[4])/dim(gamestest)[1]

par(mfrow=c(2, 2))
plot(rf1, main='Default Random Forest Error vs Number of Trees')

##In practice, You can fine-tune parameters in Random Forest such as 
#ntree = number of tress to grow, and the default is 500. 
#mtry = number of variables randomly sampled as candidates at each split. 
#          The default is sqrt(p) for classfication and p/3 for regression
#nodesize = minimum size of terminal nodes. 
#           The default value is 1 for classification and 5 for regression

rf2 <- randomForest(as.factor(HOME_TEAM_WINS) ~., data=gamestrain, ntree= 1000, 
                    mtry=sqrt(ncol(gamestrain)), nodesize =5, importance=TRUE)


## The testing error of this new randomForest 
rf.pred2 = predict(rf2, gamestest, type='class')
table(rf.pred2, y2)

plot(rf2, main= 'Median Grid Search Random Forest Error vs Number of Trees')
min(plot(rf2))

rf3 <- randomForest(as.factor(HOME_TEAM_WINS) ~., data=gamestrain, ntree= 250, 
                    mtry=1, nodesize =1, importance=TRUE)

rf4 <- randomForest(as.factor(HOME_TEAM_WINS) ~., data=gamestrain, ntree= 2000, 
                    mtry=11, nodesize =9, importance=TRUE)

par(mfrow=c(2,2))
plot(rf1, main='Default Param RF vs Trees')
plot(rf2, main= 'Median Param RF vs Trees')
plot(rf3, main= 'Low-End Param RF vs Trees')
plot(rf4, main= 'High-End Param RF vs Trees')

grid_search_rf <- expand.grid(
  tree       = c(500, 1000, 2000),
  mtry       = seq(1, 11, by = 2),
  node_size  = seq(1, 9, by = 2),
  sample_size = seq(.55, .85, by = .1),
  error   = 0
)


for (i in 1:nrow(grid_search_rf)) {
  # Convert predicted probabilities to predicted classes using cutoff
  # new data
  model <- ranger(formula = HOME_TEAM_WINS ~., 
                        data=gamestrain, 
                        num.trees= grid_search_rf$tree[i], 
                        mtry=grid_search_rf$mtry[i], 
                        min.node.size =grid_search_rf$node_size[i],
                        sample.fraction = grid_search_rf$sample_size[i])
  
  model.pred = predict(model, gamestest[, -ncol(gamestest)], type='response')
  class.pred <- if_else(model.pred$predictions >= best_cutoff, 1, 0)
  # error
  err_test <- mean( class.pred != gamestest$HOME_TEAM_WINS)
  # combine error
  grid_search_rf$error[i] <- err_test
}
best_rf_ind <- which.min(grid_search_rf$error)
best_rf_row <- grid_search_rf[best_rf_ind, ]
optimal_rf <- ranger(formula = HOME_TEAM_WINS ~., 
                     data=gamestrain, 
                     num.trees= best_rf_row$tree, 
                     mtry=best_rf_row$mtry, 
                     min.node.size = best_rf_row$node_size,
                     sample.fraction = best_rf_row$sample_size)

importance(optimal_rf, type=2)
varImpPlot(optimal_rf, main= "Variable Importance for Random Forest")

## In general, we need to use a loop to try different parameter
## values (of ntree, mtry, etc.) to identify the right parameters 
## that minimize cross-validation errors.



### Method 1: LDA
# fit1 <- lda( y ~ ., data= auto.train, CV= TRUE)
mod1 <- lda(gamestrain[, -ncol(games)], gamestrain[, ncol(games)])
mod1

## training error 
## we provide a detailed code here 
pred1 <- predict(mod1,gamestrain[, -ncol(games)])$class; 
pred1
TrainErr <- c(TrainErr, mean( pred1  != gamestrain[, ncol(games)])); 
TrainErr
## test error
pred1test <- predict(mod1,gamestest[, -ncol(games)])$class; 
TestErr <- c(TestErr,mean(pred1test != gamestest[, ncol(games)]));  
TestErr;

## You can also see the details of Testing Error
##     by the confusion table, which shows how the errors occur
table(pred1test,  gamestest[, ncol(games)]) 


# evaluation of distribution of variables for LDA and QDA

par(mfrow=c(4, 4))
# Iterate through each column of the dataset
for (predictor in names(games)) {
  # Skip non-numeric columns
  if (!is.numeric(games[[predictor]])) {
    next
  }
  if (!endsWith(predictor, "_AWAY")) {
    next
  }
  
  # Generate QQ plot for the current predictor
  qqPlot(games[[predictor]], ylab = predictor, main = predictor)
}

# Monte Carlo and statistical significance

# Cross Validation Code

n = dim(games)[1]; ### total number of observations
n1 = round(n/4); ### number of observations randomly selected for testing data

B= 50 ### number of loops
TEALL = NULL; ### Final TE values
set.seed(7406); ### You might want to set the seed for randomization
for (b in 1:B){
  ### randomly select 10% of observations as testing data in each loop
  flag <- sort(sample(1:n, n1));
  gamestrain <- games[-flag,]
  gamestest <- games[flag,]
  
  
  ### Method 1: LDA
  mod1 <- lda(gamestrain[,-ncol(gamestrain)], gamestrain[,ncol(gamestrain)]) 
  pred1test <- predict(mod1, gamestest[,-ncol(gamestest)])$class
  te1 <- mean(pred1test != gamestest$HOME_TEAM_WINS)
  
  ## Method 2: Random Forest
  mod2 <- ranger(formula = HOME_TEAM_WINS ~., 
                       data=gamestrain, 
                       num.trees= best_rf_row$tree, 
                       mtry=best_rf_row$mtry, 
                       min.node.size = best_rf_row$node_size,
                       sample.fraction = best_rf_row$sample_size)
  model.pred2 = predict(mod2, gamestest[, -ncol(gamestest)], type='response')
  class.pred2 <- if_else(model.pred2$predictions >= best_cutoff, 1, 0)
  # error
  te2 <- mean( class.pred2 != gamestest$HOME_TEAM_WINS)
  
  ### Method 3: binomial logisitic regression
  lr_mod <- glm(as.factor(HOME_TEAM_WINS) ~., family = binomial(link="logit"), data=gamestrain)
  lr_mod <- stepAIC(lr_mod, direction = 'both', trace = FALSE)
  pred_probs <- predict(lr_mod, gamestest[, -ncol(gamestrain)], type = 'response')
  pred <- if_else(pred_probs >= best_cutoff, 1, 0)
  te3 <- mean( pred != gamestest$HOME_TEAM_WINS)
  
  ### Method 4: K-nearest numbers
  xnew2 <- gamestest[,-ncol(games)]
  kpred_test_best <- knn(gamestrain[,-ncol(games)], xnew2, gamestrain[, ncol(games)], k=best_k)
  te4 <- c(TestErr, mean( kpred_test_best != gamestest$HOME_TEAM_WINS) )
  
  # gather all together
  current_test <- cbind(te1, te2, te3, te4)
  TEALL = rbind( TEALL, current_test )
  
}
dim(TEALL)
(TEALL)
colnames(TEALL) <- c("LDA", "Random Forest", 
                     "Logistic Reg", "KNN")
TEALL
sampleError <- apply(TEALL, 2, mean);
sampleVariance <- apply(TEALL, 2, var)
cvTab <- rbind(sampleError, sampleVariance)
cvTab


p <- boxplot(TEALL, main="Classification Testing Error",
             ylab="Model Testing Error",
             names = c("LDA", "RF", 
                       "Log Reg", "KNN"), las=2)
p

anova_result <- aov(sampleError ~ 1)
summary(anova_result)


#t-test loop
model_to_compare <- sampleError[3]
t_table <- data.frame()

for (i in 1:length(sampleError)) {
  if (i != 3) {
    # Create a data frame for t-test
    data_t <- data.frame(
      Model = c("Logistic Regression", paste0("Model", i)),
      Error = c(model_to_compare, sampleError[i])
    )
    print(data_t)
    # Perform t-test
    t_result <- t.test(Error ~ 1, data = data_t)
    # Append result to table
    print(t_result)
  }
}


# Golden State Analysis
# read in data including game and team ID
team_games <- read.table(file= 'C:\\Users\\nsien\\OneDrive\\Documents\\GT Prep\\ISYE 7406\\nba_project\\nba_game_data_proj.csv', sep = ",", header = TRUE)
gsw_id <- 1610612744
gsw_home_games <- team_games[team_games$TEAM_ID_HOME == gsw_id,]
gsw_home_games <- gsw_home_games[, -(1:5)]
gsw_home_games <- gsw_home_games[, -which(names(gsw_home_games) == 'TEAM_ID_AWAY')]
dim(gsw_home_games)
head(gsw_home_games)

## Split to training and testing subset 
flag <- sort(sample(dim(gsw_home_games)[1],dim(gsw_home_games)[1]*.25, replace = FALSE))
gsw_gamestrain <- gsw_home_games[-flag,]
gsw_gamestest <- gsw_home_games[flag,]

### GSW binomial logisitic regression
gsw_mod <- glm(HOME_TEAM_WINS ~., family = binomial(link="logit"), data=gsw_gamestrain);  
summary(gsw_mod)
gsw_mod <- stepAIC(gsw_mod, direction = 'both', trace = FALSE)
summary(gsw_mod)
## Training Error  of binomial logisitic regression
pred_probs <- predict(gsw_mod, gsw_gamestrain[, -ncol(gsw_gamestrain)], type = 'response')
print(pred_probs)
# determining the optimal cutoff value
best_cutoff <- NULL
best_err <- 1

# Loop through possible cutoff values, picking the optimal value on the train set
for (cutoff in seq(0.1, 0.9, by = 0.05)) {
  # Convert predicted probabilities to predicted classes using cutoff
  cutoff_pred <- ifelse(pred_probs >= cutoff, 1, 0)
  err <- mean( cutoff_pred != gsw_gamestrain$HOME_TEAM_WINS)
  # Update best cutoff if err is less than previous best
  if (err < best_err) {
    print(paste('hit at ', cutoff, ", best error: ", err))
    best_err <- err
    best_cutoff <- cutoff
  }
}

# Orlando Analysis
orl_id <- 1610612753
orl_home_games <- team_games[team_games$TEAM_ID_HOME == orl_id,]
orl_home_games <- team_games[team_games$TEAM_ID_HOME == orl_id,]
orl_home_games <- orl_home_games[, -(1:5)]
orl_home_games <- orl_home_games[, -which(names(orl_home_games) == 'TEAM_ID_AWAY')]
dim(orl_home_games)
head(orl_home_games)

## Split to training and testing subset 
flag <- sort(sample(dim(orl_home_games)[1],dim(orl_home_games)[1]*.25, replace = FALSE))
orl_gamestrain <- orl_home_games[-flag,]
orl_gamestest <- orl_home_games[flag,]

### orl binomial logisitic regression
orl_mod <- glm(HOME_TEAM_WINS ~., family = binomial(link="logit"), data=orl_gamestrain);  
summary(orl_mod)
orl_mod <- stepAIC(orl_mod, direction = 'both', trace = FALSE)
summary(orl_mod)
## Training Error  of binomial logisitic regression
pred_probs <- predict(orl_mod, orl_gamestrain[, -ncol(orl_gamestrain)], type = 'response')
print(pred_probs)
# determining the optimal cutoff value
best_cutoff <- NULL
best_err <- 1

# Loop through possible cutoff values, picking the optimal value on the train set
for (cutoff in seq(0.1, 0.9, by = 0.05)) {
  # Convert predicted probabilities to predicted classes using cutoff
  cutoff_pred <- ifelse(pred_probs >= cutoff, 1, 0)
  err <- mean( cutoff_pred != orl_gamestrain$HOME_TEAM_WINS)
  # Update best cutoff if err is less than previous best
  if (err < best_err) {
    print(paste('hit at ', cutoff, ", best error: ", err))
    best_err <- err
    best_cutoff <- cutoff
  }
}



# compare logistic regression coefficients from stepwise selected models
#GSW manually entered coefficient values
gsw_df <- data.frame(
  ID = c('Intercept','fg3a_ra_HOME', 'fta_ra_HOME', 'oreb_ra_HOME', 
         'asts_ra_HOME', 'stls_ra_HOME', 'FG_pct_ra_HOME', 'FG3_pct_ra_HOME', 'pts_ra_AWAY', 'fta_ra_AWAY'),
  Overall = c(-2.13128217,0.08679272,0.10421993,0.22405856, 0.04159217, 0.24201001, 
              45.90135662, 10.18970063,0.11916734, -0.1068676),
  GSW = c(-45.5550314,0.1184245,0.2040231,-0.3927696, -0.7072851, 0.5325578, 
          87.2612325, 34.9049962, -0.1440843,0.1849315)
)

# create multiples of overall coefficient
gsw_df_adj <- gsw_df
gsw_df_adj$'GSW / Overall' <- gsw_df_adj$GSW / gsw_df_adj$Overall
gsw_df_adj$GSW <- NULL
gsw_df_adj$Overall <- 1

# Function to normalize values within columns from -1 to 1
normalize_within_columns <- function(x) {
  print(x)
  scaled <- x / mean(x)
  return(scaled)
}

# Apply the normalization function to all columns except the ID column
gsw_df_scaled <- gsw_df
gsw_df_scaled[, -1] <- apply(gsw_df_scaled[, -1], 1, normalize_within_columns)


# Melt the dataframe to long format
library(reshape2)
gsw_unscaled_long <- melt(gsw_df, id.vars = 'ID')
gsw_df_adj <- melt(gsw_df_adj, id.vars = 'ID')
gsw_df_adj$unscaled <- gsw_unscaled_long$value

# Create scatterplot with large points
library(ggplot2)
ggplot(gsw_df_adj, aes(x = ID, y = value, color = variable, label = round(unscaled, 2))) +
  geom_point(size = 6) +
  labs(x = "Coefficients", y = "Coefficient Value / Overall Coefficient Value", color = "Column") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# orl manually entered coefficient values
orl_df <- data.frame(
  ID = c('Intercept','pts_ra_HOME','fta_ra_HOME','asts_ra_HOME', 'FG3_pct_ra_HOME','FT_pct_ra_HOME',
              'fg3a_ra_AWAY', 'dreb_ra_AWAY','asts_ra_AWAY','FG_pct_ra_AWAY','FG3_pct_ra_AWAY'),
  Overall = c(-2.13128217,-0.13605487,0.10421993,0.04159217,10.18970063,5.52091489,
              -0.0882796,-0.17640928,-0.05227141,-38.01893011,-9.19075739),
  ORL = c(-23.53926784,-0.15829014,0.40632341,0.67852133,15.35067761,20.66953342,
          -0.05471538,-0.28734564,0.18680639,-28.11013152,-18.02491819)
)

orl_df_adj <- orl_df
orl_df_adj$'ORL / Overall' <- orl_df_adj$ORL / orl_df_adj$Overall
orl_df_adj$ORL <- NULL
orl_df_adj$Overall <- 1

# Function to normalize values within columns from -1 to 1
normalize_within_columns <- function(x) {
  print(x)
  scaled <- x / mean(x)
  return(scaled)
}

# Apply the normalization function to all columns except the ID column
orl_df_scaled <- orl_df
orl_df_scaled[, -1] <- apply(orl_df_scaled[, -1], 1, normalize_within_columns)

# Melt the dataframe to long format
library(reshape2)
orl_unscaled_long <- melt(orl_df, id.vars = 'ID')
orl_df_adj <- melt(orl_df_adj, id.vars = 'ID')
orl_df_adj$unscaled <- orl_unscaled_long$value
orl_df_adj


# Create scatterplot with large points
library(ggplot2)
ggplot(orl_df_adj, aes(x = ID, y = value, color = variable, label = round(unscaled, 2))) +
  geom_point(size = 6) +
  labs(x = "Coefficients", y = "Coefficient Value / Overall Coefficient Value", color = "Column") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
