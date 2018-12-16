source("lib/lstm_sequence_learner.R")
### Define a memnet network used for single word indices input with softmax layer for classification.
## On the base test data set. 933 rows x 7643 cols takes roughly 250s per epoch on CPU.
define_memnet_single <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_dropout(rate = dropout)
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  targets <- sequence_encoded_m %>%
    # RNN layer
    bidirectional(layer_lstm(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


define_embed_single <- function(maxlen, vocab_size, class_label_size, embedding_matrix, embed_dim=50, dropout=0.3) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_dropout(rate = dropout)
  # output: (samples, maxlen, embedding_dim)
  
  ## Set the embedding for the input sequence.
  get_layer(sequence_encoder_m, index=1) %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  targets <- sequence_encoded_m %>%
    # RNN layer
    bidirectional(layer_lstm(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


## Use the cuda dnn library to define the lstm. 
## On NVidia graphics card this is necessary as the regular LSTM is about 4x slower on GPU than CPU 
## whereas the GPU version layer_cudnn_lstm is about 4x faster on GPU than the regular LSTM when run on CPU.
## This model depends on NVidia GPU with CUDA 9 and CuDNN 7 given the default build of tensorflow.
## An quick example of performance difference.
## LSTM on CPU takes around 230s per epoch on 933 samples of dimension 7643 columns.
## cudnn LSTM on GPU takes around 50s per epoch on same data set.
define_memnet_single_gpu <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_dropout(rate = dropout)
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  targets <- sequence_encoded_m %>%
    # RNN layer
    bidirectional(layer_cudnn_lstm(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


### Define a memnet network used for single word indices input with softmax layer for classification.
### Use a gru layer instead of the LSTM layer.
### The GRU may be slightly faster to train than the LSTM.
## On the base test data set. 933 rows x 7643 cols takes roughly 250s per epoch.
define_memnet_gru_single <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_dropout(rate = dropout)
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  targets <- sequence_encoded_m %>%
    # RNN layer
    bidirectional(layer_gru(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}



### Define a memnet network used for single word indices input with softmax layer for classification.
### Use a gru layer instead of the LSTM layer.
### The GRU may be slightly faster to train than the LSTM.
## The 1d Convolutional layers reduce the size of the network somewhat and increases the speed of training.
## On the base test data set. 933 rows x 7643 cols takes roughly 20s per epoch.
## using a convolution in front of an LSTM may be useful for achieving an operable network
## that may not require graphics acceleration.
define_memnet_lstm_conv1d_single <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_dropout(rate = dropout) 
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  # We attempt to reduce the dimensionality using a Convolutional network
  
  targets <- sequence_encoded_m %>%
    layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                  input_shape = list(NULL, embed_dim)) %>% 
    layer_max_pooling_1d(pool_size = embed_dim) %>% 
    layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
    # RNN layer
    bidirectional(layer_lstm(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}



### Define a memnet network used for single word indices input with softmax layer for classification.
### Use a gru layer instead of the LSTM layer.
### The GRU may be slightly faster to train than the LSTM.
## The 1d Convolutional layers reduce the size of the network somewhat and increases the speed of training.
## On the base test data set. 933 rows x 7643 cols takes roughly 20s per epoch. On GPU it takes about 2s per epoch.
## using a convolution in front of an LSTM may be useful for achieving an operable network
## that may not require graphics acceleration.
define_memnet_lstm_conv1d_single_gpu <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_dropout(rate = dropout) 
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  # We attempt to reduce the dimensionality using a Convolutional network
  
  targets <- sequence_encoded_m %>%
    layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                  input_shape = list(NULL, embed_dim)) %>% 
    layer_max_pooling_1d(pool_size = embed_dim) %>% 
    layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
    # RNN layer
    bidirectional(layer_cudnn_lstm(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}





train_model <- function(model, 
                        train_vec, 
                        valid_vec,  
                        train_target_vec,
                        valid_target_vec,
                        numEpochs=120,
                        logdir="logs/single", 
                        checkpointPath="checkpoints/memnet_single.h5") {
  callbacks <- list(callback_model_checkpoint(checkpointPath))
  model %>% fit(
    x = train_vec,
    y = train_target_vec,
    batch_size = 32,
    epochs = numEpochs,
    validation_data = list(valid_vec, valid_target_vec),
    callbacks = callbacks
  )
}

## Test the model
evaluate_model <- function(model, test_vec, test_vec_target) {
  model %>% evaluate(
    x=test_vec,
    y=test_vec_target,
    batch_size=32
  )
}

## Perform a prediction using the softmax to select the maximum value from the ranked
## outputs.
predict_model <- function(model, input_word_indices, target_classes, temperature=0.5) {
  weights <- model %>% predict(input_word_indices,verbose=0)
  max_idx <- softmax_index(weights, temperature)
  class <- target_classes[max_idx]
  list(
    class=class,
    weight=weights[max_idx]
  )
}
