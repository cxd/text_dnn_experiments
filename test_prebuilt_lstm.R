
library(keras)
library(stringr)
source("prepare_squad_data.R")
source("read_glove.R")
source("lstm_sequence_learner.R")

prebuilt <- "test/bri-data-01/model3.h5"

modelTest <- load_model_hdf5(prebuilt, compile=TRUE)

path <- get_file(
  "nietzsche.txt",
  origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
text <- tolower(readChar(path, file.info(path)$size))



# Select a text seed at random
## Note the shortcoming of this model is that it is a sequence generator purely for
## sequences of text that it has seen before. These are the discrete character sequences 
## that it has trained on. It is not capable of taking a sequence of characters that it has
## not been trained on and stringing togethor the next sequence of possible characters.
maxlen <- 60
start_index <- sample(1:(nchar(text) - maxlen - 1), 1)  
seed_text <- str_sub(text, start_index, start_index + maxlen - 1)

(prediction <- predict_sequence_of_length(modelTest, seed_text, temperature=0.5))

(prediction2 <- predict_sequence_until(modelTest, seed_text, window=60, temperature=0.6))


