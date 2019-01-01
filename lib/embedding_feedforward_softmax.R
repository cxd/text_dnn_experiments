source("lib/memnet_singleinput_classifier.R")
## A simple feedforward architecture leveraging embeddings
## that feed into the dense layers above to perform softmax classification.

embedding_feedforward_softmax <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3, kernelSize=5, optimizerName="rmsprop") {
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
    layer_conv_1d(filters = 32, kernel_size = kernelSize, activation = "relu",
                  input_shape = list(NULL, embed_dim)) %>% 
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    layer_max_pooling_1d(pool_size = embed_dim) %>%
    layer_flatten() %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}

## Stacked CNN with dense feedforward activation for softmax.
embedding_feedforward_stacked_cnn_softmax <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3, kernelSize=3, optimizerName="rmsprop") {
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim*2) %>%
    layer_dropout(rate = dropout)
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  targets <- sequence_encoded_m %>%
    layer_conv_1d(filters = embed_dim, kernel_size = kernelSize, activation = "relu",
                  input_shape = list(NULL, embed_dim*2)) %>% 
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # second convolutional layer.
    layer_conv_1d(filters = embed_dim/2, kernel_size = kernelSize, activation = "relu",
                  input_shape = list(NULL, embed_dim*2)) %>% 
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # third convolutional layer.
    layer_conv_1d(filters = embed_dim/2, kernel_size = kernelSize, activation = "relu",
                  input_shape = list(NULL, embed_dim)) %>% 
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    layer_max_pooling_1d(pool_size = embed_dim) %>%
    #layer_dense(class_label_size) %>%
    #layer_activation_selu() %>%
    # convert back to flattened output
    layer_flatten() %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}



## Stacked CNN with dense feedforward activation for softmax.
embedding_feedforward_stacked_cnn_softmax_nodense <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3, kernelSize=3, optimizerName="rmsprop") {
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim*2) %>%
    layer_dropout(rate = dropout)
  # output: (samples, maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  targets <- sequence_encoded_m %>%
    layer_conv_1d(filters = embed_dim, kernel_size = kernelSize, activation = "relu",
                  input_shape = list(NULL, embed_dim*2)) %>% 
    layer_max_pooling_1d(pool_size = embed_dim, strides=embed_dim) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    # second convolutional layer.
    layer_conv_1d(filters = embed_dim/2, kernel_size = kernelSize, activation = "relu",
                  input_shape = list(NULL, embed_dim*2)) %>% 
    layer_max_pooling_1d(pool_size = embed_dim, strides=2) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
     # third convolutional layer.
    layer_conv_1d(filters = class_label_size, kernel_size = 1, activation = "relu",
                  input_shape = list(NULL, embed_dim)) %>% 
    layer_max_pooling_1d(pool_size = class_label_size, strides=2) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    layer_flatten() %>%
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


## Same architecture as above but the embedding matrix is supplied and assigned to the first layer.
## This allows preexisting embeddings to be used. Such as glove embeddings.
glove_embedding_feedforward_softmax <- function(maxlen, vocab_size, class_label_size, embedding_matrix, embed_dim=50, dropout=0.3, optimizerName="rmsprop") {
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim)
  # output: (samples, maxlen, embedding_dim)
  
  ## Set the embedding for the input sequence.
  get_layer(sequence_encoder_m, index=1) %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)  %>%
    layer_dropout(rate = dropout)
  
  targets <- sequence_encoded_m %>%
    layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                  input_shape = list(NULL, embed_dim)) %>% 
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
    layer_max_pooling_1d(pool_size = embed_dim) %>%
    layer_flatten() %>%
    # convert back to flattened output
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, targets)
  model %>% compile(
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}