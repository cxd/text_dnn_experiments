source("prepare_squad_data.R")
source("read_glove.R")
source("lstm_sequence_learner.R")

squadData <- "data/squad/dev/preprocessed.csv"
squadDf <- read_saved_data(squadData)

text <- squadDf$context[1]

char_indices <- char_index_set()
seq1 <- to_char_index_sequence(char_indices, text)

seq1


windowSize <- 60
windows <- window_char_sequence(seq1, windowSize)

rows <- nrow(windows) - 1

## predicting each end character after each row of windows 
## we have the y characters being the last column of windows
next_chars <- windows[2:nrow(windows),ncol(windows)]
## Set the dimensions of the window back to original size.
windows <- windows[1:rows,1:windowSize]


one_hot_encoding <- to_one_hot_chars(text, windowSize)
## Note it will be important to train inbatches as not all data fits in memory.
## we need to select a batch size.
batchSize <- 100
## run in 10 batches
nchars <- length(char_indices)
model <- make_lstm_sequence(windowSize, nchars)

iterations <- floor(nrow(squadDf)/batchSize)
remainder <- nrow(squadDf) %% batchSize

iterations <- 10
batchStart <- 1
logdir<-"logs/run"
tensorBoardPort=5000
tensorboard(logdir, port=tensorBoardPort, launch_browser = TRUE)
i <- 1
#for(i in 1:iterations) {
  endBatch <- batchStart + batchSize
  all_data <- convert_to_char_indexes(squadDf[batchStart:endBatch,], 60)
  model <- train_on_sequences(model, all_data$contexts, batchNum=i, numEpocs=10)
  batchStart <- endBatch + 1
#}

model %>% save_model_hdf5("test_model1.h5")

(prediction <- predict_sequence_of_length(model, "the Carolingian-based cultures of West Francia", temperature=0.5))

(prediction <- predict_sequence_until(model,  "the Carolingian-based cultures of West Francia", temperature=1))
  