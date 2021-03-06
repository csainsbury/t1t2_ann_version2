## read in csv output from ann
## turn ethnicity levels into readable output for physician interpretation

levelKey <- levels(factorEthnicity)

X_test_RN <- read.csv("~/R/_workingDirectory/t1_t2_ANN/output/X_test_realNumbers.csv", sep=",", header = FALSE, row.names = NULL)
y_test    <- read.csv("~/R/_workingDirectory/t1_t2_ANN/output/y_test.csv", sep=",", header = FALSE, row.names = NULL)
y_pred    <- read.csv("~/R/_workingDirectory/t1_t2_ANN/output/y_pred.csv", sep=",", header = FALSE, row.names = NULL)

library(ROCR)
pred <- prediction(y_pred, y_test)
roc.perf <- performance(pred, measure = 'tpr', x.measure = 'fpr')
plot(roc.perf)

# find sensitvity / specifictiy / optimal cut point
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(roc.perf, pred))

X_test_RN$ethnicity_decoded <- ""
levelKey_dummyVarTrapAdj <- levelKey[2: length(levelKey)]


for (r in seq(1, nrow(X_test_RN), 1)) {
  
  eth_sub <- X_test_RN[r, ][1:24]
  
  if (sum(eth_sub) == 0) {
    X_test_RN$ethnicity_decoded[r] = "nil recorded"
  }
  
  if (sum(eth_sub) == 1) {
    ethnicityString <- levelKey_dummyVarTrapAdj[eth_sub == 1]
  }
  
  X_test_RN$ethnicity_decoded[r] <- ethnicityString
  
}

set.seed(42)

# cut redundant ethnicity keys
X_test_RN_cut <- X_test_RN[25: ncol(X_test_RN)]
colnames(X_test_RN_cut) <- c("age", "male", "hba1c", "sbp", "dbp", "bmi", "ethnicity")

X_test_RN_withTrueType <- cbind(X_test_RN_cut, y_test)
colnames(X_test_RN_withTrueType) <- c("age", "male", "hba1c", "sbp", "dbp", "bmi", "ethnicity", "DMtype")

X_test_RN_withTrueType_withProb <- cbind(X_test_RN_withTrueType, y_pred)
colnames(X_test_RN_withTrueType_withProb) <- c("age", "male", "hba1c", "sbp", "dbp", "bmi", "ethnicity", "DMtype", "DMpred")
X_test_RN_withTrueType_withProbDT <- data.table(X_test_RN_withTrueType_withProb)

t1_table <- X_test_RN_withTrueType_withProbDT[DMtype == 1]
t2_table <- X_test_RN_withTrueType_withProbDT[DMtype == 0]

random_t2_sample <- t2_table[sample(nrow(t2_table), (nrow(t1_table) * 3)), ]

physician_pool <- rbind(t1_table, random_t2_sample)

# first set - for greg
physician_set1 <- physician_pool[sample(nrow(physician_pool), 100), ]

physician_set1_write1 <- subset(physician_set1, select = c(1:7))

write.table(physician_set1, file = "~/R/_workingDirectory/t1_t2_ANN/output/physician_set1_key.csv", sep = ",", row.names = FALSE)
write.table(physician_set1_write1, file = "~/R/_workingDirectory/t1_t2_ANN/output/physician_set1.csv", sep = ",", row.names = FALSE)

# automated sets with proportion randomisation

# n per sample
sample_n <- 100

# number of sets
set_n <- 40

for (ii in seq(1, set_n, 1)) {
  
  random_t2_multiplier <- runif(1, 1, 10)
  
  #  print(random_t2_multiplier)
  
  random_t2_sample <- t2_table[sample(nrow(t2_table), (nrow(t1_table) * random_t2_multiplier)), ]
  
  physician_pool <- rbind(t1_table, random_t2_sample)
  
  physician_set <- physician_pool[sample(nrow(physician_pool), sample_n), ]
  
  physician_set_write <- subset(physician_set, select = c(1:7))
  
  keyName <- paste('~/R/_workingDirectory/t1_t2_ANN/output/samplesForTesting/physician_set_', ii, '_key.csv', sep = '')
  write.table(physician_set, file = keyName, sep = ",", row.names = FALSE)
  
  testSetName <- paste('~/R/_workingDirectory/t1_t2_ANN/output/samplesForTesting/physician_set_', ii, '.csv', sep = '')
  write.table(physician_set_write, file = testSetName, sep = ",", row.names = FALSE)
  
}

## analyse performance
library(caret)
library(ROCR)

performanceAnalysis <- function(physicianPrediction, ANNprediction, key, ANNthreshold) {
  
  ann_pred <- ifelse(ANNprediction > ANNthreshold, 1, 0)
  
  print(confusionMatrix(data = physicianPrediction, reference = key))
  
  cm_phys <- table(key, physicianPrediction); print(cm_phys)
  accuracy_phys <- (cm_phys[1,1] + cm_phys[2,2]) / sum(cm_phys); print(accuracy_phys)
  
  print(confusionMatrix(data = ann_pred, reference = key))
  
  cm_ann <- table(key, ann_pred); print(cm_ann)
  accuracy_ann <- (cm_ann[1,1] + cm_ann[2,2]) / sum(cm_ann); print(accuracy_ann)
  
  pred <- prediction(ANNprediction, key)
  roc.perf <- performance(pred, measure = 'tpr', x.measure = 'fpr')
  plot(roc.perf)
  
  auc.perf = performance(pred, measure = "auc")
  auc.perf@y.values
  
  print(auc.perf)
  
  points((cm_phys[2, 1] / sum(cm_phys[2, ])), (cm_phys[2, 2] / sum(cm_phys[2, ])), col = "red", pch = 16, cex = 2)
  points((cm_ann[2, 1] / sum(cm_ann[2, ])), (cm_ann[2, 2] / sum(cm_ann[2, ])), col = "blue", pch = 16, cex = 2)
  
  
}

perf_1 <- read.csv('~/R/_workingDirectory/t1_t2_ANN/output/samplesForTesting/physician_set_1_done.csv', header = T)
perf_1_key <- read.csv('~/R/_workingDirectory/t1_t2_ANN/output/samplesForTesting/physician_set_1_key.csv', header = T)

performanceAnalysis(perf_1$diagnosis, perf_1_key$DMpred, perf_1_key$DMtype, 0.05)

perf_1 <- read.csv('~/R/_workingDirectory/t1_t2_ANN/output/samplesForTesting/physician_set_38_CS_done.csv', header = T)
perf_1_key <- read.csv('~/R/_workingDirectory/t1_t2_ANN/output/samplesForTesting/physician_set_38_key.csv', header = T)

performanceAnalysis(perf_1$diagnosis, perf_1_key$DMpred, perf_1_key$DMtype, 0.05)


## analyse performance
perf_1 <- read.csv('~/R/_workingDirectory/t1_t2_ANN/output/physician_set1_GJ130617.csv', header = T)
perf_1_key <- read.csv('~/R/_workingDirectory/t1_t2_ANN/output/physician_set1_key.csv', header = T)

performanceAnalysis(perf_1$diagnosis, perf_1_key$DMpred, perf_1_key$DMtype, 0.05)



## reconstruct original ID list
###############################################################################################
## 140717 run from here

diagnostic_test_set_withID <- read.csv("~/R/_workingDirectory/t1_t2_ANN/diagSet_7p_withID.csv")

nameList_physicianResults <- c('physician_set_1_done', 'physician_set_6', 'physician_set_11', 'physician_set_16', 'physician_set_22', 'physician_set_38', 'physician_set_23', 'physician_set_24')
nameList_keys <- c('physician_set_1_key', 'physician_set_6_key', 'physician_set_11_key', 'physician_set_16_key', 'physician_set_22_key', 'physician_set_38_key', 'physician_set_23_key', 'physician_set_24_key')

for (k in seq(1, length(nameList_physicianResults), 1)) {
  
  phys_name <- paste('~/R/_workingDirectory/t1_t2_ANN/output/completedSets/',nameList_physicianResults[k], '.csv', sep = '')
  key_name <- paste('~/R/_workingDirectory/t1_t2_ANN/output/completedSets/',nameList_keys[k], '.csv', sep = '')
  
  physicianSet <- read.table(phys_name, header = T, row.names = NULL, sep = ',')
  colnames(physicianSet) <- c('age', 'male', 'hba1c', 'sbp', 'dbp', 'bmi', 'ethnicity', 'diagnosis')
  keySet <- read.table(key_name, header = T, row.names = NULL, sep = ',')
  
  keySet$physicianDiagnosis <- physicianSet$diagnosis
  
  set_includingIDs <- merge(keySet, as.data.frame(diagnostic_test_set_withID), by.x = c("hba1c", "sbp", "dbp",  "bmi"), by.y = c("hba1c", "sbp", "dbp",  "bmi"))
  set_includingIDs <- unique(set_includingIDs)
  
  # if running individual file, use this statement
  # outputFrame <- set_includingIDs
  
  if (k == 1) { outputFrame <- set_includingIDs }
  if (k > 1)  { outputFrame <- rbind(outputFrame, set_includingIDs) }
  
}

write.table(outputFrame, file = "~/R/_workingDirectory/t1_t2_ANN/outputFrame.csv", sep = ",", row.names = FALSE, col.names = TRUE)

# performanceAnalysis(outputFrame$physicianDiagnosis, outputFrame$DMpred, outputFrame$DMtype, 0.04306091)
performanceAnalysis(outputFrame$physicianDiagnosis, outputFrame$DMpred, outputFrame$DMtype, 0.1)

########### run r_logit_model.R and come back

logitInput <- read.csv("~/R/_workingDirectory/t1_t2_ANN/logit_output.csv")
logitInput <- data.table(logitInput)
logitInput = unique(logitInput)
outputFrame <- data.table(outputFrame)

outputFrame$prob_pred <- (-1)

for (j in seq(1, nrow(outputFrame), 1)) {
  
  idOfInterest <- outputFrame$LinkId[j]
  logitSub <- logitInput[LinkId == idOfInterest]
  
  if(nrow(logitSub == 1)) {
    outputFrame$prob_pred[j] <- logitSub$prob_pred
  }
  
}

# performanceAnalysis(outputFrame$physicianDiagnosis, outputFrame$prob_pred, outputFrame$DMtype, 0.06596188)
performanceAnalysis(outputFrame$physicianDiagnosis, outputFrame$prob_pred, outputFrame$DMtype, 0.1)


outputFrame$ann_binary <- ifelse(outputFrame$DMpred > 0.1, 1, 0)
outputFrame$logit_binary <- ifelse(outputFrame$prob_pred > 0.1, 1, 0)

physicianConcordance_ann <- sum(ifelse((outputFrame$ann_binary == 1 & outputFrame$physicianDiagnosis == 1) | (outputFrame$ann_binary == 0 & outputFrame$physicianDiagnosis == 0), 1, 0)) / nrow(outputFrame)
print(physicianConcordance_ann)
physicianConcordance_logit <- sum(ifelse((outputFrame$logit_binary == 1 & outputFrame$physicianDiagnosis == 1) | (outputFrame$logit_binary == 0 & outputFrame$physicianDiagnosis == 0), 1, 0)) / nrow(outputFrame)
print(physicianConcordance_logit)
ann_concordance_logit <- sum(ifelse((outputFrame$ann_binary == 1 & outputFrame$logit_binary == 1) | (outputFrame$ann_binary == 0 & outputFrame$logit_binary == 0), 1, 0)) / nrow(outputFrame)
print(ann_concordance_logit)

# flag those where ann and logit agree, but diagnosis different
high_prob_misdiagnosis <- outputFrame[(ann_binary == logit_binary) & (ann_binary != DMtype)]

# lm probabilities / residuals
plot(outputFrame$DMpred, outputFrame$prob_pred)
regressionLine <- lm(outputFrame$prob_pred ~ outputFrame$DMpred)
abline(regressionLine, col = 'red')
lm_residuals <- resid(regressionLine)
outputFrame$residuals <- lm_residuals
outputFrame_orderedByResiduals <- outputFrame[order(outputFrame$residuals), ]

# plot relations
plot(outputFrame$bmi, outputFrame$DMpred)
plot(outputFrame$bmi, outputFrame$prob_pred)

plot(outputFrame$age.x, outputFrame$DMpred)

ann_age_boxplot <- boxplot(outputFrame$DMpred ~ cut(outputFrame$age.x, breaks = seq(0, 100, 5)))
logit_age_boxplot <- boxplot(outputFrame$prob_pred ~ cut(outputFrame$age.x, breaks = seq(0, 100, 5)))

plot(ann_age_boxplot$stats[3, ]); lines(ann_age_boxplot$stats[3, ])
points(logit_age_boxplot$stats[3, ], col = 'red'); lines(logit_age_boxplot$stats[3, ], col = 'red')

ann_bmi_boxplot <- boxplot(outputFrame$DMpred ~ cut(outputFrame$bmi, breaks = seq(10, 50, 2)))
logit_bmi_boxplot <- boxplot(outputFrame$prob_pred ~ cut(outputFrame$bmi, breaks = seq(10, 50, 2)))

plot(ann_bmi_boxplot$stats[3, ]); lines(ann_bmi_boxplot$stats[3, ])
points(logit_bmi_boxplot$stats[3, ], col = 'red'); lines(logit_bmi_boxplot$stats[3, ], col = 'red')

ann_hba1c_boxplot <- boxplot(outputFrame$DMpred ~ cut(outputFrame$hba1c, breaks = seq(10, 100, 5)))
logit_hba1c_boxplot <- boxplot(outputFrame$prob_pred ~ cut(outputFrame$hba1c, breaks = seq(10, 100, 5)))

plot(ann_hba1c_boxplot$stats[3, ]); lines(ann_hba1c_boxplot$stats[3, ])
points(logit_hba1c_boxplot$stats[3, ], col = 'red'); lines(logit_hba1c_boxplot$stats[3, ], col = 'red')

ann_sbp_boxplot <- boxplot(outputFrame$DMpred ~ cut(outputFrame$sbp, breaks = seq(80, 200, 10)))
logit_sbp_boxplot <- boxplot(outputFrame$prob_pred ~ cut(outputFrame$sbp, breaks = seq(80, 200, 10)))

plot(ann_sbp_boxplot$stats[3, ]); lines(ann_sbp_boxplot$stats[3, ])
points(logit_sbp_boxplot$stats[3, ], col = 'red'); lines(logit_sbp_boxplot$stats[3, ], col = 'red')



plot(outputFrame$age.x, outputFrame$prob_pred)
plot(outputFrame$age.x, outputFrame$DMpred)
plot(outputFrame$sbp, outputFrame$DMpred)
plot(outputFrame$dbp, outputFrame$DMpred)

boxplot(outputFrame$prob_pred ~ outputFrame$ethnicity.x, las=3, varwidth = T)
boxplot(outputFrame$DMpred ~ outputFrame$ethnicity.x, las=3, varwidth = T)

plot(outputFrame$hba1c, outputFrame$DMpred)
plot(outputFrame$hba1c, outputFrame$prob_pred)
plot(outputFrame$hba1c, outputFrame$DMpred)
points(outputFrame$hba1c, outputFrame$prob_pred, col = 'red')


