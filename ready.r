setwd('C:/Users/nikos/OneDrive/Έγγραφα/MSc/Classes/Bioinformatics for Translational Medicine/bstm-project')

library('caret')
library('gbm')

data<-read.delim('train_call.tsv', header = TRUE, sep = "\t", quote = "\"", dec = ".", fill = TRUE, comment.char = "")

data<-t(data)

clinic<-read.delim('train_clinical.txt', header = TRUE, sep = "\t", quote = "\"", dec = ".", fill = TRUE, comment.char = "",row.names='Sample')

Combo<-merge(clinic, data, by="row.names")
row.names(Combo)<-Combo$Row.names
Combo$Row.names<-NULL

sample <- sample(c(TRUE, FALSE), nrow(Combo), replace=TRUE, prob=c(0.8,0.2))
train  <- Combo[sample,]
test   <- Combo[!sample,]

rocVarImp<-filterVarImp(train[,-1], as.factor(train[,1]), nonpara = FALSE)

final<-apply(rocVarImp, 1, mean)

features<-final[order(final,decreasing = TRUE)]

features<-as.data.frame(features)

index_names <- row.names(features)[1:30]

write.csv(Combo[ , c(index_names)], "n30_combo.csv", row.names=TRUE)

