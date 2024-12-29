library(lattice)
library(gplots)
library(dplyr)
library(ggplot2)
library(e1071)

# run runSVM
run_svm_and_plot <- function(data_path, props, run_name, n_runs) {
  # load data
  lda_data <- read.csv(data_path, header = TRUE, stringsAsFactors = FALSE)
  rownames(lda_data) <- lda_data[, 1]
  lda_data <- lda_data[, -1]
  
  result_lda <- runSVM(lda_data[,1:13], lda_data[, 15], props, run_name, n_runs)
  result_lda <- result_lda %>%
    group_by(run_name, train_prop) %>%
    summarise(avg_accu = mean(accu), sd_accu = sd(accu)) %>%
    ungroup()
  
  # plot
  p <- ggplot() +
    # first line
    geom_errorbar(data = result_lda, aes(x = train_prop, y = avg_accu, group = run_name, 
                                         ymin = avg_accu - sd_accu, ymax = avg_accu + sd_accu), width =.02, color = "blue") +
    geom_line(data = result_lda, aes(x = train_prop, y = avg_accu, group = run_name, color = "LDA"), size = 1) +
    geom_point(data = result_lda, aes(x = train_prop, y = avg_accu, group = run_name, color = "LDA"), size = 4) +
    xlab("Traning Data Proportion") +
    ylab("Prediction Accuracy") +
    
    scale_color_manual(name = "Lines", values = c(LDA = "blue")) +
    # 
    theme(
      aspect.ratio = 1/1.2, 
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14)
    )
  
  return(p)
}

# input
# dat: raw data,
# y: label for raw data (cell types)
# props: proportion of training
# n_runs: number of svm repetition
# output
# svm predicted accuracy on current setting
# run once
run.SVM.once <- function(dat, y, train_prop, run_name, seed) {
  set.seed(seed)
  y <- factor(y)
  # split train and test
  tr_idx <- sort(sample(1:length(y), round(train_prop * length(y))))
  te_idx <- 1:length(y)
  te_idx <- te_idx[!(1:length(y) %in% tr_idx)]
  xtr <- dat[tr_idx, ]
  ytr <- y[tr_idx]
  xte <- dat[te_idx, ]
  yte <- y[te_idx]
  # run svm
  model <- svm(x = xtr, y = ytr)
  pred <- predict(model, xte)
  # calculate accuracy
  accu <- sum(pred == yte) / length(yte)
  
  return(data.frame(run_name, train_prop, accu))
}


# @export
runSVM <- function(dat, y, props, run_name, n_runs) {
  res <- data.frame()
  for (train_prop in props) {
    tmp <- lapply(1:n_runs, function(x) run.SVM.once(dat, y, train_prop, run_name,123))
    res <- rbind(res, do.call("rbind", tmp))
  }
  return(res)
}
