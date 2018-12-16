### Define a memnet network used for single word indices input with softmax layer for classification.

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
