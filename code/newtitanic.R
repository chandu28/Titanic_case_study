# working directory
setwd("C:\\Users\\Chandu\\Desktop\\IMAR DATA\\titanic data")
getwd()
#IMPORTING THE DATA

train_dataset <-read.csv('train.csv',header= TRUE, sep=',')
test_dataset <- read.csv('test.csv',header= TRUE, sep = ',')

summary(train_dataset)
library(psych)
describe(train_dataset)
library(Hmisc)
describe(train_dataset)
sum(is.na(train_dataset$Age))
train_dataset$Age <- ifelse(is.na(train_dataset$Age),
                            median(train_dataset$Age,na.rm = TRUE),
                            train_dataset$Age)
sum(is.na(train_dataset$Age))
table(train_dataset$Age)
#INSERTING THE MEDIAN IN THE MISSING VALUES OF AGE

#checking null values

table(train_dataset$Embarked)

train_dataset$Embarked[train_dataset$Embarked=='']<-'S'
table(train_dataset$Embarked)
names(train_dataset)


library(psych)
library(Hmisc)

class(train_dataset$Pclass)

train_dataset$Pclass <- as.numeric(train_dataset$Pclass)

train_dataset$Pclass <- as.factor(train_dataset$Pclass)

class(train_dataset$Survived)

str(train_dataset)
model <- glm( Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + 
0               Embarked , family = "binomial", 
              data = train_dataset)
summary(model)
train_dataset$predict_model <- predict(model,train_dataset,type = "response")

train_dataset$pred_surv <- ifelse(train_dataset$predict_model>=0.5,1,0)

table(train_dataset$Survived,train_dataset$pred_surv)

library(party)

png(file = "dtree3.png")
set.seed(1234)


list(train_dataset)
outtree <- ctree(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,data = train_dataset)

train_dataset$Age <- as.numeric(train_dataset$Age)
summary(outtree)
outtree
plot(outtree)
dev.off()

train_dataset$preds_tree <- predict(outtree,train_dataset,type="response")

train_dataset$pred_class_tree <- ifelse(train_dataset$preds_tree>=0.5,1,0)

table(train_dataset$Survived,train_dataset$pred_surv)

((489+258)/891)*100

library(randomForest)

model_rf <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, ntree=2000, data = train_dataset)

y

train_dataset$rf_preds <- predict(model_rf,train_dataset)

train_dataset$pred_class_rf <- ifelse(train_dataset$rf_preds>=0.5,1,0)

table(train_dataset$Survived,train_dataset$pred_class_rf)
((531+286)/891)*100

##test

summary(test_dataset)
library(psych)
describe(train_dataset)
library(Hmisc)
describe(train_dataset)
sum(is.na(test_dataset$Age))
test_dataset$Age <- ifelse(is.na(test_dataset$Age),
                           median(test_dataset$Age,na.rm = TRUE),
                           test_dataset$Age)
sum(is.na(test_dataset$Age))
table(test_dataset$Age)
#INSERTING THE MEDIAN IN THE MISSING VALUES OF AGE
test_dataset$Age[test_dataset$Age == ''] <- 'Medium'
boxplot(test_dataset$Fare)
#checking null values

table(test_dataset$Embarked)

test_dataset$Embarked[test_dataset$Embarked=='']<-'S'
table(test_dataset$Embarked)
names(test_dataset)


class(test_dataset$Pclass)

test_dataset$Pclass <- as.numeric(test_dataset$Pclass)

test_dataset$Pclass <- as.factor(test_dataset$Pclass)

class(test_dataset$Survived)

str(train_dataset)


test_dataset$Age <- as.numeric(test_dataset$Age)
class(test_dataset$Age)

library(randomForest)

test_dataset$pred_sur<- predict(model_rf,test_dataset)

test_dataset$survived_pred <- ifelse(test_dataset$pred_sur>=0.5,1,0)
 write.csv(test_dataset,"end.csv")

