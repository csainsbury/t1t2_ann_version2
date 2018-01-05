library(data.table)

returnUnixDateTime<-function(date) {
  returnVal<-as.numeric(as.POSIXct(date, format="%Y-%m-%d", tz="GMT"))
  return(returnVal)
}

findTimeToNearestHbA1c <- function(dmList_LinkId, diagnosisDate_unix) {
  
  hb_id_sub <- cleanHbA1cDataDT[LinkId == dmList_LinkId]
  diffFromDiagnosisDate <- hb_id_sub$dateplustime1 - diagnosisDate_unix
  
  findClosest <- sqrt(diffFromDiagnosisDate ^ 2)
  flagClosest <- ifelse(findClosest == min(findClosest), 1, 0)
  
  outputList <- list(findClosest[flagClosest == 1], hb_id_sub$hba1cNumeric[flagClosest == 1])
  
  return(outputList)
  
}

# set index date
index <- "2018-01-01"
paramFromDiagnosisWindowMonths = 12
paramFromDiagnosisWindowSeconds = paramFromDiagnosisWindowMonths * (60*60*24*(365.25/12))

numberOfYearsData <- 1 # minimum length of time within the dataset to allow correct diagnosis to have been reached

# diagnosisDataset<-read.csv("../GlCoSy/SDsource/diagnosisDateDeathDate.txt")
diagnosisDataset<-read.csv("~/R/_workingDirectory/nEqOneTrial/rawData/demog_all2.txt", quote = "", 
                           row.names = NULL, 
                           stringsAsFactors = FALSE)

diagnosisDatasetDT = data.table(diagnosisDataset)

  # cut down dataset
  cut_diagDT <- data.table(diagnosisDatasetDT$LinkId, diagnosisDatasetDT$DateOfDiagnosisDiabetes_Date, diagnosisDatasetDT$Ethnicity_Mapped, diagnosisDatasetDT$CurrentGender_Mapped, diagnosisDatasetDT$DiabetesMellitusType_Mapped, diagnosisDatasetDT$BirthDate)
  colnames(cut_diagDT) <- c("LinkId", "diagnosisDate", "Ethnicity", "Sex", "diabetesType", "DOB")
  cut_diagDT$LinkId <- as.numeric(cut_diagDT$LinkId)
  cut_diagDT$LinkId <- as.numeric(cut_diagDT$LinkId)
  cut_diagDT$DOB_unix <- returnUnixDateTime(cut_diagDT$DOB)
  
  # remove those without a diagnosis date
  cut_diagDT$diagnosisDate_unix <- returnUnixDateTime(cut_diagDT$diagnosisDate)
  cut_diagDT$diagnosisDate_unix[is.na(cut_diagDT$diagnosisDate_unix)] <- 0
  cut_diagDT <- cut_diagDT[diagnosisDate_unix > 0]
  
  # flag first diagnosis and remove all others from dataset
  cut_diagDT[, c("firstDiagnosis") := ifelse(diagnosisDate_unix == min(diagnosisDate_unix), 1, 0) , by=.(LinkId)]
  cut_diagDT <- cut_diagDT[firstDiagnosis == 1]
  
  # cut to any DM type that might require insulin:
  cut_diagDT <- cut_diagDT[diabetesType == "Type 1 Diabetes Mellitus" | diabetesType == "Type 2 Diabetes Mellitus" | diabetesType == "Latent Autoimmune Diabetes of Adulthood" | diabetesType == "Maternally Inherited Diabetes and Deafness" | diabetesType == "Maturity Onset Diabetes of Youth" | diabetesType == "Other Type of Diabetes" | diabetesType == "Secondary - Disease" | diabetesType == "Secondary - Pancreatic Pathology"]
  
  # cut to ensure at least 1y data for each ID
  cut_diagDT <- cut_diagDT[diagnosisDate_unix < (returnUnixDateTime(index) - ((60*60*24*365.25) * numberOfYearsData))]

# generate node and link files
cleanHbA1cData <- read.csv("~/R/_workingDirectory/nEqOneTrial/cleanedData/hba1cDTclean.csv", sep=",", header = TRUE, row.names = NULL)
cleanHbA1cData$timeSeriesDataPoint <- cleanHbA1cData$hba1cNumeric
  cleanHbA1cDataDT <- data.table(cleanHbA1cData)
  hb_DT_forMerge <- data.table(cleanHbA1cDataDT$LinkId, cleanHbA1cDataDT$dateplustime1, cleanHbA1cDataDT$hba1cNumeric)
  colnames(hb_DT_forMerge) <- c("LinkId", "hb_dateplustime1", "hba1cNumeric")

cleanSBPData <- read.csv("~/R/_workingDirectory/nEqOneTrial/cleanedData/SBPsetDTclean.csv", sep=",", header = TRUE, row.names = NULL)
cleanSBPData$timeSeriesDataPoint <- cleanSBPData$sbpNumeric
  cleanSBPDataDT <- data.table(cleanSBPData)
  sbp_DT_forMerge <- data.table(cleanSBPDataDT$LinkId, cleanSBPDataDT$dateplustime1, cleanSBPDataDT$sbpNumeric)
  colnames(sbp_DT_forMerge) <- c("LinkId", "sbp_dateplustime1", "sbpNumeric")
  
cleanDBPData <- read.csv("~/R/_workingDirectory/nEqOneTrial/cleanedData/DBPsetDTclean.csv", sep=",", header = TRUE, row.names = NULL)
cleanDBPData$timeSeriesDataPoint <- cleanDBPData$dbpNumeric
  cleanDBPDataDT <- data.table(cleanDBPData)
  dbp_DT_forMerge <- data.table(cleanDBPDataDT$LinkId, cleanDBPDataDT$dateplustime1, cleanDBPDataDT$dbpNumeric)
  colnames(dbp_DT_forMerge) <- c("LinkId", "dbp_dateplustime1", "dbpNumeric")

cleanBMIData <- read.csv("~/R/_workingDirectory/nEqOneTrial/cleanedData/BMISetDTclean.csv", sep=",", header = TRUE, row.names = NULL)
  cleanBMIDataDT <- data.table(cleanBMIData)
  bmi_DT_forMerge <- data.table(cleanBMIDataDT$LinkId, cleanBMIDataDT$dateplustime1, cleanBMIDataDT$bmiNumeric)
  colnames(bmi_DT_forMerge) <- c("LinkId", "bmi_dateplustime1", "bmiNumeric")
  
# cleanRenalData <- read.csv("~/R/GlCoSy/SD_workingSource/renalSetDTclean.csv", sep=",", header = TRUE, row.names = NULL)
#   cleanRenalDataDT <- data.table(cleanRenalData)
#   renal_DT_forMerge <- data.table(cleanRenalDataDT$LinkId, cleanRenalDataDT$dateplustime1, cleanRenalDataDT$egfrNumeric)
#   colnames(renal_DT_forMerge) <- c("LinkId", "egfr_dateplustime1", "egfrNumeric")


# find closest value to diagnosis date for each parameter - sequential merge

# hba1c
merge_hb <- merge(hb_DT_forMerge, cut_diagDT, by = "LinkId")
merge_hb[, c("hb_diffFromDiag") := sqrt((hb_dateplustime1 - diagnosisDate_unix) ^ 2) , by=.(LinkId)]
merge_hb[, c("hb_flagClosest") := ifelse(hb_diffFromDiag == min(hb_diffFromDiag), 1, 0) , by=.(LinkId)]

merge_hb <- merge_hb[(hb_diffFromDiag < paramFromDiagnosisWindowSeconds) & hb_flagClosest == 1]
merge_hb <- merge_hb[diff(merge_hb$LinkId) != 0]

# SBP
merge_sbp <- merge(merge_hb, sbp_DT_forMerge, by = "LinkId")
merge_sbp[, c("sbp_diffFromDiag") := sqrt((sbp_dateplustime1 - diagnosisDate_unix) ^ 2) , by=.(LinkId)]
merge_sbp[, c("sbp_flagClosest") := ifelse(sbp_diffFromDiag == min(sbp_diffFromDiag), 1, 0) , by=.(LinkId)]

merge_sbp <- merge_sbp[(sbp_diffFromDiag < paramFromDiagnosisWindowSeconds) & sbp_flagClosest == 1]
merge_sbp <- merge_sbp[diff(merge_sbp$LinkId) != 0]


# DBP
merge_dbp <- merge(merge_sbp, dbp_DT_forMerge, by = "LinkId")
merge_dbp[, c("dbp_diffFromDiag") := sqrt((dbp_dateplustime1 - diagnosisDate_unix) ^ 2) , by=.(LinkId)]
merge_dbp[, c("dbp_flagClosest") := ifelse(dbp_diffFromDiag == min(dbp_diffFromDiag), 1, 0) , by=.(LinkId)]

merge_dbp <- merge_dbp[(dbp_diffFromDiag < paramFromDiagnosisWindowSeconds) & dbp_flagClosest == 1]
merge_dbp <- merge_dbp[diff(merge_dbp$LinkId) != 0]

# BMI
merge_bmi <- merge(merge_dbp, bmi_DT_forMerge, by = "LinkId")
merge_bmi[, c("bmi_diffFromDiag") := sqrt((bmi_dateplustime1 - diagnosisDate_unix) ^ 2) , by=.(LinkId)]
merge_bmi[, c("bmi_flagClosest") := ifelse(bmi_diffFromDiag == min(bmi_diffFromDiag), 1, 0) , by=.(LinkId)]

merge_bmi <- merge_bmi[(bmi_diffFromDiag < paramFromDiagnosisWindowSeconds) & bmi_flagClosest == 1]
merge_bmi <- merge_bmi[diff(merge_bmi$LinkId) != 0]


# renal
# merge_renal <- merge(merge_bmi, renal_DT_forMerge, by = "LinkId")
# merge_renal[, c("egfr_diffFromDiag") := sqrt((egfr_dateplustime1 - diagnosisDate_unix) ^ 2) , by=.(LinkId)]
# merge_renal[, c("egfr_flagClosest") := ifelse(egfr_diffFromDiag == min(egfr_diffFromDiag), 1, 0) , by=.(LinkId)]
# 
# merge_renal <- merge_renal[(egfr_diffFromDiag < paramFromDiagnosisWindowSeconds) & egfr_flagClosest == 1]

# finalset
diagnostic_test_set <- merge_bmi

# remove duplicates
#diagnostic_test_set <- unique(diagnostic_test_set)

diagnostic_test_set$Sex <- ifelse(diagnostic_test_set$Sex == "Male", 1, 0)
diagnostic_test_set$diabetesType <- ifelse(diagnostic_test_set$diabetesType == "Type 1 Diabetes Mellitus", 1, 0)

diagnostic_test_set <- diagnostic_test_set[substr(Ethnicity,1,3) != "ECD"]
# table(diagnostic_test_set$Ethnicity)
# table(subset(diagnostic_test_set, diabetesType == 1)$Ethnicity)
# table(subset(diagnostic_test_set, diabetesType == 0)$Ethnicity)
  factorEthnicity <- factor(diagnostic_test_set$Ethnicity)
  diagnostic_test_set$Ethnicity <- as.numeric(factorEthnicity)

diagnostic_test_set$ageAtDiagnosis <- (diagnostic_test_set$diagnosisDate_unix - diagnostic_test_set$DOB_unix) / (60*60*24*365.25)

diagnostic_test_set_forDrugMatch <- diagnostic_test_set

## 
# classify by insulin prescription


simplifyDrugs <- function(inputFrame) {
  
  # inputFrame <- interestSet
  # inputFrame <- inputFrame[1:100000,]
  
  inputFrame$DrugName <- as.character(inputFrame$DrugName)
  
  inputFrame$DrugName.original <- inputFrame$DrugName
  inputFrame$DrugNameNew <- inputFrame$DrugName
  
  inputFrame <- subset(inputFrame, DrugNameNew != "Disposable")
  
  inputFrame$DrugNameNew[grep("ideglira", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("xultophy", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  # bd mix insulins
  inputFrame$DrugNameNew[grep("Humalog Mix", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Novomix", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Mixtard", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Humulin M4", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("Humulin M3", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Humulin M2", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Humulin M1", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("HUMULIN M2", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("HUMULIN M1", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("Humulin", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("Humalog Mix", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  # basal insulins
  inputFrame$DrugNameNew[grep("Insulin Glargine", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Lantus", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("degludec", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Degludec", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Tresiba", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Insulin Detemir", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Insulatard", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("ULTRATARD", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"  
  inputFrame$DrugNameNew[grep("MONOTARD", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Humulin I", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  # prandial insulins
  inputFrame$DrugNameNew[grep("Actrapid", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Humalog", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Insulin Lispro", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("Novorapid", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("aspart", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("fiasp", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  inputFrame$DrugNameNew[grep("Apidra", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Humulin S", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  # animal insulins
  inputFrame$DrugNameNew[grep("Bovine", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("BOVINE", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  inputFrame$DrugNameNew[grep("Porcine", inputFrame$DrugName, ignore.case = TRUE)] <- "insulin_"
  
  x <- as.data.frame(table(inputFrame$DrugNameNew))
  x = x[order(x$Freq), ]
  
  outputFrame <- inputFrame
  
  outputFrame$DrugName.original <- NULL
  outputFrame$DrugName <- outputFrame$DrugNameNew
  outputFrame$DrugNameNew <- NULL
  
  return(outputFrame)
}

# generate node and link files
drugDataSet <- read.csv("~/R/GlCoSy/SDsource/Export_all_diabetes_drugs.txt",header=TRUE,row.names=NULL)
topUpDrugData <-paste("~/R/_workingDirectory/nEqOneTrial/rawData/diabetesDrugs_nov16-nov17.txt",sep="")
topUpDrugDataSet <- read.csv(topUpDrugData)

concatDrugSet <- rbind(drugDataSet, topUpDrugDataSet)
concatDrugSet <- unique(concatDrugSet)

drugDataSet <- concatDrugSet

# generate list of IDs from the test set above to reduce the drug list to those needed to be analysed to reduce time
IDsForMerge <- data.frame(diagnostic_test_set_forDrugMatch$LinkId, diagnostic_test_set_forDrugMatch$diagnosisDate_unix); colnames(IDsForMerge) <- c("LinkId", "diagnosisDate_unix")
# merge IDs with drug list
drugDataSetForAnalysis <- merge(IDsForMerge, drugDataSet, by.x = "LinkId", by.y = "LinkId")
# run simplify drugs to identify those with an insulin prescription
interestDrugSet <- simplifyDrugs(drugDataSetForAnalysis)

interestDrugSet$PrescriptionDateTime_unix <- returnUnixDateTime(interestDrugSet$PrescriptionDateTime)

# flag whether any insulin prescription in the 12 months post diagnosis
interestDrugSet$prescriptionInterval <- interestDrugSet$PrescriptionDateTime_unix - interestDrugSet$diagnosisDate_unix
interestDrugSet <- subset(interestDrugSet, prescriptionInterval < (60*60*24*365.25) & prescriptionInterval >= 0)
interestDrugSet$insulinFlag <- ifelse(interestDrugSet$DrugName == "insulin_", 1, 0)

# flag if any insulin prescription in the first year
interestDrugSetDT <- data.table(interestDrugSet)
interestDrugSetDT[, c("anyInsulin") := ifelse(sum(insulinFlag > 0), 1, 0) , by=.(LinkId)]
# single row per ID
interestDrugSetDT[, c("prescriptionN") := seq(1, .N, 1) , by=.(LinkId)]
interestDrugSetDTsingleRowPerID <- interestDrugSetDT[prescriptionN == 1]
# set for merging back to diagnostic data:
drugs_mergeBack <- data.frame(interestDrugSetDTsingleRowPerID$LinkId, interestDrugSetDTsingleRowPerID$anyInsulin)
  colnames(drugs_mergeBack) <- c("LinkId", "anyInsulin")
  
## merge back to form final set for saving out
diagnosticSetWithInsulinInfo <- merge(diagnostic_test_set_forDrugMatch, drugs_mergeBack, by.x = "LinkId", by.y = "LinkId", all.x = F)

## save out files for import into python:
diagnosticSetWithInsulinInfo_withID <- data.table(diagnosticSetWithInsulinInfo$LinkId, diagnosticSetWithInsulinInfo$ageAtDiagnosis, diagnosticSetWithInsulinInfo$Ethnicity, diagnosticSetWithInsulinInfo$Sex, diagnosticSetWithInsulinInfo$hba1cNumeric, diagnosticSetWithInsulinInfo$sbpNumeric, diagnosticSetWithInsulinInfo$dbpNumeric,  diagnosticSetWithInsulinInfo$bmiNumeric, diagnosticSetWithInsulinInfo$anyInsulin)

colnames(diagnosticSetWithInsulinInfo_withID) <- c("LinkId", "age", "ethnicity", "sex", "hba1c", "sbp", "dbp", "bmi", "anyInsulin")

diagnosticSetWithInsulinInfo_withoutID <- data.table(diagnosticSetWithInsulinInfo$ageAtDiagnosis, diagnosticSetWithInsulinInfo$Ethnicity, diagnosticSetWithInsulinInfo$Sex, diagnosticSetWithInsulinInfo$hba1cNumeric, diagnosticSetWithInsulinInfo$sbpNumeric, diagnosticSetWithInsulinInfo$dbpNumeric,  diagnosticSetWithInsulinInfo$bmiNumeric, diagnosticSetWithInsulinInfo$anyInsulin)

colnames(diagnosticSetWithInsulinInfo_withoutID) <- c("age", "ethnicity", "sex", "hba1c", "sbp", "dbp", "bmi", "anyInsulin")

# summary(diagnosticSetWithInsulinInfo_withoutID)
# summary(subset(diagnosticSetWithInsulinInfo_withoutID, anyInsulin == 1))
# summary(subset(diagnosticSetWithInsulinInfo_withoutID, anyInsulin == 0))


write.table(diagnosticSetWithInsulinInfo_withoutID, file = "~/R/_workingDirectory/t1t2_ann_version2/diagSet_7p.csv", sep = ",", row.names = FALSE, col.names = TRUE)
write.table(diagnosticSetWithInsulinInfo_withID, file = "~/R/_workingDirectory/t1t2_ann_version2/diagSet_7p_withID.csv", sep = ",", row.names = FALSE, col.names = TRUE)



