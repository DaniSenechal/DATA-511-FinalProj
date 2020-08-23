########################################################
# Project 3
# Danielle Senechal
# DATA 511 Intro to Data Science
# July 14th, 2020
########################################################
set.seed(12345)

library(plyr)
library(caret)
library(rattle)

# read in dataset
proj3 <- read_csv("/Users/daniellesenechal/Documents/CCSU/DATA 511/Project 3/proj3_income")
# View(proj3)


# Delete the variable occupation
(proj3$occupation <- NULL)

# impute missing values from capital gain variable: 

proj3$`capital-gain`[proj3$`capital-gain` == 99999] <- NA # remove code 99999 (missing values)

cgm <- mean(proj3$`capital-gain`, na.rm = TRUE); cgm # mean w/out missing values
cgsd <- sd(proj3$`capital-gain`, na.rm = TRUE); cgsd # standard dev w/out missing values

imputation_model <- preProcess(proj3, method = c("knnImpute")) # model used for predicting
proj3.imp <- predict(imputation_model, proj3) # model to predict standardized values for missing data

proj3$cg.imp <- round(proj3.imp$`capital-gain` * cgsd + cgm, 5) # append the unstandardized precited

summary(proj3$`capital-gain`) # summary of original capital gain variable (with missing)
summary(proj3$cg.imp) # summary of unstandardized predicted missing values (no more missing)
sd(proj3$cg.imp) # standard deviation of unstandardized predicted missing values (no more missing)

(proj3$`capital-gain` <- proj3$cg.imp)
(proj3$cg.imp <- NULL)

#################### Question 2 ####################

########## Part a ##########
# summarize income
summary(proj3) # get summary of all variables in category 3
t.income <- table(proj3$income); t.income # table of income
prop.table(t.income) # proportion table of income

t.inc.edu <- table(proj3$income, proj3$education); t.inc.edu # old category table (16 column)

factor(proj3$education) # get titles of all categories of education

# reclassify education into educ (2 categories: low, high)
proj3$educ <- revalue(proj3$education, 
                                c("Preschool" = "low", 
                                  "1st-4th" = "low",
                                  "5th-6th" = "low",
                                  "7th-8th" = "low",
                                  "9th" = "low",
                                  "10th" = "low", 
                                  "11th" = "low", 
                                  "12th" = "low", 
                                  "HS-grad" = "low", 
                                  "Some-college" = "high", 
                                  "Assoc-acdm" = "high", 
                                  "Assoc-voc" = "high", 
                                  "Bachelors" = "high",
                                  "Prof-school" = "high",
                                  "Masters" = "high",
                                  "Doctorate" = "high"))

# View(proj3) # ensure that the educ column was appended and accurate

# Delete the variable education
(proj3$education <- NULL)

# View(proj3) # ensure that the variable education was deleted

t.inc.edu.new <- table(proj3$income, proj3$educ); t.inc.edu.new # proportion contingency table
p.inc.edu.new <- prop.table(t.inc.edu.new, 2) # column percentages
p.inc.edu.new <- p.inc.edu.new * 100 # multiply by 100 to get percentages
p.inc.edu.new <- round(p.inc.edu.new, digits = 2); p.inc.edu.new # round to 2 decimal places


########## Part b ##########
t.inc.rel <- table(proj3$income, proj3$relationship); t.inc.rel # old category table (6 column)

factor(proj3$relationship) # get titles of all categories of relationship

# reclassify relationship into rel (2 categories: HusWife, Other)
proj3$rel <- revalue(proj3$relationship, 
                      c("Husband" = "HusWife", 
                        "Wife" = "HusWife",
                        "Not-in-family" = "Other",
                        "Other-relative" = "Other",
                        "Own-child" = "Other",
                        "Unmarried" = "Other"))

# View(proj3) # ensure that the rel column was appended and accurate

# Delete the variable education
(proj3$relationship <- NULL)

# View(proj3) # ensure that the variable relationship was deleted

t.inc.rel.new <- table(proj3$income, proj3$rel); t.inc.rel.new # proportion contingency table
p.inc.rel.new <- prop.table(t.inc.rel.new, 2) # column percentages
p.inc.rel.new <- p.inc.rel.new * 100 # multiply by 100 to get percentages
p.inc.rel.new <- round(p.inc.rel.new, digits = 2); p.inc.rel.new # round to 2 decimal places



#################### Question 3 ####################

inTrain <- createDataPartition(y = proj3$income, p = .5, list = FALSE) # 50/50 partition on income

str(inTrain) # structure of inTrain

proj3.tr <- proj3[inTrain,] # select inTrain records for training data
proj3.te  <- proj3[-inTrain,] # select records not in inTrain for testing data

summary(proj3.tr$income)
table(proj3.tr$income)
summary(proj3.te$income)
table(proj3.te$income)

########## Part b ##########
proj3.tr$part <- rep("train", nrow(proj3.tr)) # Append training classifier to all training data
proj3.te$part <- rep("test", nrow(proj3.te)) # Append testing classifier to all testing data

proj3.all <- rbind(proj3.tr, proj3.te) # append a data type (training/testing) column

# View(proj3.all) # View dataset to make sure the part variable was added correctly

# comparison boxplot for capital gains
boxplot(proj3.all$`capital-gain` ~ as.factor(part), data = proj3.all)
# Kruskal-Wallis test on capital gains
kruskal.test(proj3.all$`capital-gain` ~ as.factor(part), data = proj3.all)

# comparison boxplot for capital losses
boxplot(proj3.all$`capital-loss` ~ as.factor(part), data = proj3.all)
# Kruskal-Wallis test on capital losses
kruskal.test(proj3.all$`capital-loss` ~ as.factor(part), data = proj3.all)


########## Part c ##########
# education
t.edu.train <- table(proj3.tr$educ); t.edu.train # training counts
t.edu.test <- table(proj3.te$educ); t.edu.test # testing counts
educ.part.table <- matrix( c(8835, 7446, 8972, 7308), ncol=2) # counts from left to right
prop.test(educ.part.table, correct = FALSE) # chi-squared test


# relationship
t.rel.train <- table(proj3.tr$rel); t.rel.train # training counts
t.rel.test <- table(proj3.te$rel); t.rel.test # testing counts
rel.part.table <- matrix( c(7378, 8903, 7383, 8897), ncol=2) # counts from left to right
prop.test(rel.part.table, correct = FALSE) # chi-squared test


proj3.tr$part <- NULL # delete appended data type classifier column from training data
proj3.te$part <- NULL # delete appended data type classifier column from testing data



#################### Question 4 ####################

t.inc.train <- table(proj3.tr$income); t.inc.train # training counts



#################### Question 5 ####################

TC <- trainControl(method = "CV", number = 10) # 10-fold cross-validation
colnames(proj3.tr) <- make.names(colnames(proj3.tr)) # names for CART model command (. instead of -)
fit <- train(income~., data = proj3.tr[], method = "rpart2", trControl = TC) # CART with 10 fold CV
fancyRpartPlot(fit$finalModel) # plot of decision tree



#################### Question 6 ####################
fit$resample # check for overfitting



#################### Question 7 ####################
colnames(proj3.te) <- make.names(colnames(proj3.te)) # names for predict command (. instead of -)
testsetpreds <- predict(fit, proj3.te)

# Examine a contingency table of the test set predictions
# against the actual test sest value.
table(proj3.te$income, testsetpreds)
