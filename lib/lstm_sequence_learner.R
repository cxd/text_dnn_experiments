require(keras)
require(tensorflow)

source("lib/prepare_squad_data.R")


## LSTM Sequence learner following the example from deep learning with R.
## 8.1-text-generation-with-lstm

## This uses a single layer LSTM model 
## the input data is encoded as  (N , M , K)
## N is the number of rows in the sliding window representation of the quantized characters per sentence.
## M is the window width of the character quantization, it acts as a time series
## K is the size of the alphabet of the one hot encoded character vectors from the position window[i,j]
##
## The input layer of the lstm uses only the N x K dimension.
##
## Output targets are one-hot encoded the loss function used is the categorical_crossentropy function.
##
## The structure of the model is used in the generation of a softmax output for most likely characters
## given the starting character sequence.
## As this is a playful example generating text by first
##
## feeding a 1D convolution of text into the lstm model.
## merging the result into a single dense layer which produces an output
## of vector length corresponding to the number of characters.
## the sequence generation is then used in a softmax prediction of
## the most likely character.
##
make_lstm_sequence <- function(window, shapeKChars, 
                               numUnits=128, 
                               optimizer=optimizer_rmsprop(lr=0.01),
                               loss="categorical_crossentropy") {
  model <- keras_model_sequential() %>%
    layer_lstm(units=numUnits, input_shape = c(window, shapeKChars)) %>%
    layer_dense(units = shapeKChars, activation="softmax")
  model %>% compile(
    loss=loss,
    optimizer=optimizer
  )
  model
}

## Taking the input sequences of 1D convolutions in x_data, y_data
## train the model.
train_on_sequences <- function(model, data, 
                               batchNum=1,
                               batch_size=128, 
                               numEpocs=100, 
                               withTensorBoard=TRUE,
                               logdir="logs/run", 
                               checkpointPath="checkpoints/model.h5") {
  ignore <- sapply(1:length(data), function(i) {
    x <- data[[i]]$data_x
    y <- data[[i]]$data_y
    callbacks <- list(callback_model_checkpoint(checkpointPath))
    if (withTensorBoard == TRUE) {
      callbacks <- list(callback_tensorboard(paste0(logdir,"/",batchNum)),
                        callback_model_checkpoint(checkpointPath))
    }
    model %>% fit(
      x, y, 
      batch_size=batch_size, 
      epochs=numEpocs,
      callbacks=callbacks
    )
  })
  model
}

## The output of the dense layer is a vector of weights
## the same length as the character alphabet.
## in order to predict the next character the weights are normalised
## and the maximum value index is drawn from the set.
softmax_index <- function(vector, temperature=0.5) {
  preds <- as.numeric(vector)
  preds <- log(preds) / temperature
  exp_preds <- exp(preds)
  preds <- exp_preds / sum(exp_preds)
  which.max(t(rmultinom(1, 1, preds)))
}

## Given the vector of characters and the output from the dense layer.
## Return the index of the most heavily weighted character.
next_mostlikely_char <- function(chars, vector, temperature=0.5) {
  idx <- softmax_index(vector, temperature)
  chars[idx]
}

## Predict the sequence given a sequence length.
## In this example a seed text is always required.
predict_sequence_of_length <- function(model, seed_text, window=60, sequence_length=100, temperature=0.5) {
  sliding_text <- seed_text
  chars <- set_of_chars()
  full_text_response <- c()
  for(n in 1:sequence_length) {
    ## Encode the current sliding window input features.
    data <- to_one_hot_chars(sliding_text, window)
    x <- data$data_x
    likelihood_vector <- model %>% predict(x, verbose=0)
    nextChar <- next_mostlikely_char(chars, likelihood_vector[1,], temperature)
    ## append to sliding window
    sliding_text <- paste0(sliding_text, nextChar)
    sliding_text <- substring(sliding_text, 2)
    
    #print(sliding_text)
    
    full_text_response <- c(full_text_response, nextChar)
  }
  # collating all text.
  final_response <- paste0(full_text_response, collapse="")
  list(input_text=seed_text,
       response=final_response)
}

## Predict the next sequence given a termination character.
## In this example a seed text is always required.
predict_sequence_until <- function(model, seed_text, window=60, term_char=c(".","?","!"), temperature=0.5) {
  sliding_text <- seed_text
  chars <- set_of_chars()
  
  flag <- FALSE
  full_text_response <- c()
  while(isFALSE(flag)) {
    data <- to_one_hot_chars(sliding_text, window)
    x <- data$data_x
    likelihood_vector <- model %>% predict(x, verbose=0)
    nextChar <- next_mostlikely_char(chars, likelihood_vector[1,], temperature)
    ## append to sliding window
    sliding_text <- paste0(sliding_text, nextChar)
    sliding_text <- substring(sliding_text, 2)
    
    flag <- which(nextChar %in% term_char)
    if (length(flag) == 0) {
      flag <- FALSE
    } else {
      flag <- TRUE
    }
    
    full_text_response <- c(full_text_response, nextChar)
  }
  # collating all text.
  final_response <- paste0(full_text_response, collapse="")
  list(input_text=seed_text,
       response=final_response)
}
