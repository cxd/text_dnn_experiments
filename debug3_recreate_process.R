## This recreates the same process described in the 8.1 text generation example.
## Note also this is the same kind of model as described here: 
## https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/


library(keras)
library(stringr)
source("prepare_squad_data.R")
source("read_glove.R")
source("lstm_sequence_learner.R")

## Installing using conda.
## install_keras(method="conda", conda="/usr/local/miniconda3/bin/conda", tensorflow="default")
## After this cd to /usr/local/miniconda3/env
## source activate r-tensorflow
## then install into the environment conda install tensorflow-mkl

## Initialising Keras using conda library.

## Initialising Keras using conda library.
#use_python("/usr/local/miniconda3/envs/r-tensorflow/python")
#use_condaenv("r-tensorflow", conda="/usr/local/miniconda3/bin/conda")



## this script recreates the work done in the example of text processing.

## It confirms that for the use case of generating a sequence, the character input 
## representation is the same as defined in the example.

## Note the shortcoming of this model is that it is a sequence generator purely for
## sequences of text that it has seen before. These are the discrete character sequences 
## that it has trained on. It is not capable of taking a sequence of characters that it has
## not been trained on and stringing togethor the next sequence of possible characters.

## In this case the seed text is generated from a sequence of the previously seen examples.

path <- get_file(
  "nietzsche.txt",
  origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
text <- tolower(readChar(path, file.info(path)$size))

## use a 4th of the text for small dev set.
temp <- strsplit(text,"")[[1]]
len <- floor(length(temp)/4)
temp <- temp[1:len]
text <- paste0(temp,collapse = "")

cat("Corpus length:", nchar(text), "\n")


char_indices <- char_index_set()
windowSize <- 60
data <- to_one_hot_chars(text, windowSize)

nchars <- length(char_indices)

batchSize <- 100
## run in 10 batches
nchars <- length(char_indices)
model1 <- make_lstm_sequence(windowSize, nchars)

if (file.exists("test_model3.h5") == TRUE) {
  model1 <- load_model_hdf5("test_model3.h5", compile=TRUE)
}

iterations <- 1
batchStart <- 1
logdir<-"logs/debug3"
tensorBoardPort=5000

## the example suggests using 60 epochs
numEpocs <- 1

tensorboard(logdir, port=tensorBoardPort, launch_browser = TRUE)

i <- 1
for(i in 1:iterations) {
  model1 <- train_on_sequences(model1, list(data), batchNum=i, 
                               numEpocs=numEpocs,  
                               logdir="logs/debug3",
                               checkpointPath="checkpoints/model3.h5")
}

model1 %>% save_model_hdf5("test_model3.h5")

# Select a text seed at random
maxlen <- 60
start_index <- sample(1:(nchar(text) - maxlen - 1), 1)  
seed_text <- str_sub(text, start_index, start_index + maxlen - 1)

(prediction <- predict_sequence_of_length(model1, seed_text, temperature=0.5))
