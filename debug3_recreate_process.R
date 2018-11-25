## This recreates the same process described in the 8.1 text generation example.

library(keras)
library(stringr)
source("prepare_squad_data.R")
source("read_glove.R")
source("lstm_sequence_learner.R")

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


iterations <- 10
batchStart <- 1
logdir<-"logs/debug3"
tensorBoardPort=5000
tensorboard(logdir, port=tensorBoardPort, launch_browser = TRUE)
for(i in 1:iterations) {
  model1 <- train_on_sequences(model1, list(data), batchNum=i, numEpocs=10,  logdir="logs/debug3")
}



(prediction <- predict_sequence_of_length(model1, text[1:100], temperature=1))
