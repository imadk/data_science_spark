path <- "C:/Users/Imad/Documents/!Code/Pre-processing/output/dataset/";
merged <- paste(path, "merged_tts.csv", sep="");

#Read csv from file to dataframe
filestring = paste("file:///", merged, sep = "");
datasetMerged <- read.csv(filestring, na.strings = c(".", "NA", "", "?"), sep = ",", strip.white = TRUE);

dataset <- datasetMerged;

stats <- NULL;
size <- length(dataset$result);
grayorna <- sum(dataset$result == "GrayZone or NA");
tns <- sum(dataset$result == "TN");
tps <- sum(dataset$result == "TP");
fps <- sum(dataset$result == "FP");
fns <- sum(dataset$result == "FN");

stats$tns <- tns;
stats$fps <- fps;
stats$tps <- tps;
stats$fns <- fns;
stats$grayorna <- grayorna;
stats$accuracy <- (tns + tps) / (tns + tps + fns + fps);
stats$sensitivity <- tps / (tps + fns);
stats$specificity <- tns / (tns + fps);
stats$f1score <-  2*((stats$sensitivity*stats$specificity)/(stats$sensitivity+stats$specificity))
stats$fpRatio <- fps / (tns + tps + fns + fps);
stats$tpRatio <- tps / (tns + tps + fns + fps);
stats$tnRatio <- tns / (tns + tps + fns + fps);
stats$fnRatio <- fns / (tns + tps + fns + fps);
stats$grayornaRatio <-  1 - (tns + tps + fns + fps) / size;
stats$detectionsRatio <- (tns + tps + fns + fps) / size;  
stats$detections <- (tns + tps + fns + fps);
stats$datasize <- size;
stats$errors <- sum(dataset$query_error == 1);
stats$timeouts <- sum(dataset$query_timeout == 1);
stats$totaltime <- sum(dataset$query_time);
stats$averagetime <- mean(dataset$query_time);


stats <- as.data.frame(stats)
stats.T <- t(stats)
stats <- stats.T


value3 = paste("C:/Users/Imad/Documents/!Code/Pre-processing/output/dataset/merged_stats", ".csv", sep="");
value4 = paste("C:/Users/Imad/Documents/!Code/Pre-processing/FrontEnd/data/merged_stats", ".csv", sep="");
con<-file(value3, encoding="utf8");
write.csv(stats, file=con, row.names=TRUE);
con<-file(value4, encoding="utf8");
write.csv(stats, file=con, row.names=TRUE);
