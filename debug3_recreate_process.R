## This recreates the same process described in the 8.1 text generation example.

library(keras)
library(stringr)
source("prepare_squad_data.R")
source("read_glove.R")
source("lstm_sequence_learner.R")

## this script recreates the work done in the example of text processing.

path <- get_file(
  "nietzsche.txt",
  origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
text <- tolower(readChar(path, file.info(path)$size))
cat("Corpus length:", nchar(text), "\n")


char_indices <- char_index_set()
windowSize <- 60
data <- to_one_hot_chars(text, windowSize)

nchars <- length(char_indices)
model <- make_lstm_sequence(windowSize, nchars)

batchSize <- 100
## run in 10 batches
nchars <- length(char_indices)
model1 <- make_lstm_sequence(windowSize, nchars)


iterations <- 1
batchStart <- 1
logdir<-"logs/debug3"
tensorBoardPort=5000

## the example suggests using 60 epochs
numEpocs <- 60

tensorboard(logdir, port=tensorBoardPort, launch_browser = TRUE)

for(i in 1:iterations) {
  model1 <- train_on_sequences(model1, list(data), batchNum=i, 
                               numEpocs=numEpocs,  
                               logdir="logs/debug3",
                               checkpointPath="checkpoints/model3.h5")
}

model1 %>% save_model_hdf5("test_model3.h5")

temp <- strsplit(text," ")[[1]]
temp2 <- paste(temp[1:100], collapse=" ")

(prediction <- predict_sequence_of_length(model1, temp2, temperature=0.5))
