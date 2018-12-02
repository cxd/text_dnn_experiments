
library(keras)
library(stringr)

source("lib/init.R")
source("lib/read_babi_data.R")

# Setup environment
cfg <- init(getwd())

## Recreate the process described in the R example: https://keras.rstudio.com/articles/examples/babi_memnn.html

path <- download_babi_data()

data_set <- read_train_test(path)
train <- data_set$train
test <- data_set$test

vocab_data <- extract_vocab(train, test)

model_data <- vectorize_dataset(train, test, vocab_data$vocab, vocab_data$story_maxlen, vocab_data$query_maxlen) 

model1 <- define_memnet_babi(vocab_data$story_maxlen, vocab_data$query_maxlen, vocab_data$vocab_size)

summary(model1)

## Need to partition training into train and validate.


## Since the questions do not represent classes the data set is just a 70 30 split.
## it is not requiring balancing.

set.seed(42L)

seq <- 1:nrow(model_data$train_vec$stories)
pc <- floor(0.7*length(seq))
idx <- sample(seq, pc, replace=FALSE)

train1 <- list(
  stories=model_data$train_vec$stories[idx,],
  questions=model_data$train_vec$questions[idx,],
  answers=model_data$train_vec$answers[idx,]
)

val1 <- list(
  stories=model_data$train_vec$stories[-idx,],
  questions=model_data$train_vec$questions[-idx,],
  answers=model_data$train_vec$answers[-idx,]
)

history <- train_model(model1, 
            train1, 
            validation_vec=val1,  
            numEpochs=120,
            logdir="logs/babi", 
            checkpointPath="checkpoints/babimodel.h5")

evaluate_model(model1, model_data$test_vec)

plot(history)

## Save the model.
model1 %>% save_model_hdf5("saved_models/test_babi_debug4.h5")

## run testing on new story and new question.
context <- "John travelled to the office."
question <- "Where is John?"

predict_answer(model1, vocab_data$vocab, vocab_data$story_maxlen, vocab_data$query_maxlen, context, question)

predict_answer(model1, vocab_data$vocab, vocab_data$story_maxlen, vocab_data$query_maxlen,
               "Mary travelled to the hallway. John went to the kitchen. Mary went to the kitchen.",
               "Where is Mary?")

predict_answer(model1, vocab_data$vocab, vocab_data$story_maxlen, vocab_data$query_maxlen,
               "Mary travelled to the hallway. John went to the kitchen. Mary went to the kitchen. John went to the hallway.",
               "Where is John?")


