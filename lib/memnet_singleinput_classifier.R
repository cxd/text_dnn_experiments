source("lib/lstm_sequence_learner.R")
### Define a memnet network used for single word indices input with softmax layer for classification.
## On the base test data set. 933 rows x 7643 cols takes roughly 250s per epoch on CPU.
define_memnet_single <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3, optimizerName="rmsprop") {
  
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
    bidirectional(layer_lstm(units=embed_dim,
                             activation="tanh",
                             recurrent_activation="sigmoid",
                             recurrent_dropout=0,
                             unroll=FALSE,
                             use_bias=TRUE)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
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


define_embed_single <- function(maxlen, vocab_size, class_label_size, embedding_matrix, embed_dim=50, dropout=0.3, optimizerName="rmsprop") {
  
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
    optimizer=optimizerName,
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
define_memnet_single_gpu <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3, optimizerName="rmsprop") {
  
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
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


## Embedding matrix on lstm for gpu.
define_embed_single_gpu <- function(maxlen, vocab_size, class_label_size, embedding_matrix, embed_dim=50, dropout=0.3,optimizerName="rmsprop") {
  
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
    bidirectional(layer_cudnn_lstm(units=embed_dim)) %>%
    # Regularization layer.
    layer_dropout(rate=dropout) %>%
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



### Define a memnet network used for single word indices input with softmax layer for classification.
### Use a gru layer instead of the LSTM layer.
### The GRU may be slightly faster to train than the LSTM.
## On the base test data set. 933 rows x 7643 cols takes roughly 250s per epoch.
define_memnet_gru_single <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3,  optimizerName="rmsprop") {
  
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
    optimizer=optimizerName,
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
define_memnet_lstm_conv1d_single <- function(maxlen, vocab_size, class_label_size, embed_dim=64, dropout=0.3, optimizerName="rmsprop") {
  
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
    optimizer=optimizerName,
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
define_memnet_lstm_conv1d_single_gpu <- function(maxlen, vocab_size, class_label_size, 
                                                 embed_dim=64, 
                                                 dropout=0.3, 
                                                 optimizerName="rmsprop",
                                                 kernel_regularizer=regularizer_l1(l=0.01),
                                                 kernel_size=c(3),
                                                 pooling_size=2,
                                                 filter_list=c(32,12),
                                                 lstm_units=12,
                                                 kernel_activation="relu",
                                                 gpu_flag=FALSE,
                                                 embedding_matrix=NULL,
                                                 freeze_weights=FALSE,
                                                 bidirectional=FALSE,
                                                 lstm_activation="tanh",
                                                 recurrent_activation="sigmoid",
                                                 recurrent_dropout=0,
                                                 lstm_bias=TRUE,
                                                 batch_norm=TRUE) {
  
  # Placeholders
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
    layer_spatial_dropout_1d(rate=dropout)
    #layer_dropout(rate = dropout) 
  # output: (samples, maxlen, embedding_dim)
  
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  
  ## Set the embedding for the input sequence.
  if (!is.null(embedding_matrix) & freeze_weights) {
    get_layer(sequence_encoder_m, index=1) %>%
      set_weights(list(embedding_matrix)) %>%
      freeze_weights()
  } else if (!is.null(embedding_matrix)) {
    get_layer(sequence_encoder_m, index=1) %>%
      set_weights(list(embedding_matrix))
  }
 
  if (length(kernel_size) < length(filter_list)) {
    kernel_size <- rep(kernel_size[[1]], length(filter_list))
  }
  # We attempt to reduce the dimensionality using a Convolutional network
  cnn_network_layers <- build_cnn_layers(sequence_encoded_m, embed_dim, filter_list,
                                         kernel_size_list=kernel_size, 
                                         kernel_regularizer=kernel_regularizer,
                                         pooling_size=pooling_size, 
                                         kernel_activation=kernel_activation,
                                         batch_norm=batch_norm) %>%
    layer_dropout(dropout)
  
  lstm_network_layers <- if (bidirectional) {
    cnn_network_layers %>% bidirectional(layer_lstm(units=lstm_units, 
                                                    activation=lstm_activation,
                                                    recurrent_activation=recurrent_activation,
                                                    recurrent_dropout=recurrent_dropout,
                                                    use_bias=lstm_bias))
  } else {
    cnn_network_layers %>% layer_lstm(units=lstm_units,
                                      activation=lstm_activation,
                                      recurrent_activation=recurrent_activation,
                                      recurrent_dropout=recurrent_dropout,
                                      use_bias=lstm_bias)
  }
  # convert back to flattened output
  prediction_layer <- lstm_network_layers %>% 
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, prediction_layer)
  model %>% compile(
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


build_cnn_layers <- function(last_layer, input_size, filter_list, 
                             kernel_size_list=3, 
                             kernel_regularizer=regularizer_l1(l=0.01),
                             pooling_size=2, 
                             kernel_activation="relu",
                             batch_norm=TRUE) {
  
  for (i in 1:length(filter_list)) {
    num_filters <- filter_list[[i]]
    kernel_size <- kernel_size_list[[i]]
    last_layer <- if (i == 1) {
      # two conv layers then max pooling
      last_layer %>% 
        layer_conv_1d(filters = num_filters, 
                      kernel_size = kernel_size,
                      activation=kernel_activation,
                      kernel_regularizer=kernel_regularizer,
                      input_shape = list(NULL, input_size)) %>% 
        layer_conv_1d(filters = num_filters, 
                      activation=kernel_activation,
                      kernel_size = kernel_size,
                      kernel_regularizer=kernel_regularizer) %>% 
        layer_max_pooling_1d(pool_size = pooling_size)
        
      if (batch_norm) {
          last_layer <- last_layer %>%
            layer_batch_normalization()
        }
        
      
    } else {
      last_layer %>% layer_conv_1d(filters = num_filters, 
                                   activation=kernel_activation,
                                   kernel_size = kernel_size,
                                   kernel_regularizer=kernel_regularizer) %>%
        layer_conv_1d(filters = num_filters, 
                      activation=kernel_activation,
                      kernel_size = kernel_size,
                      kernel_regularizer=kernel_regularizer) %>%
        layer_max_pooling_1d(pool_size = pooling_size)
      
      if (batch_norm) {
        last_layer <- last_layer %>%
          layer_batch_normalization()
      }
      
    }
    
  }
  
  last_layer
  
} 


### Define a memnet network used for single word indices input with softmax layer for classification.
### Use a gru layer instead of the LSTM layer.
### The GRU may be slightly faster to train than the LSTM.
## The 1d Convolutional layers reduce the size of the network somewhat and increases the speed of training.
## On the base test data set. 933 rows x 7643 cols takes roughly 20s per epoch. On GPU it takes about 2s per epoch.
## using a convolution in front of an LSTM may be useful for achieving an operable network
## that may not require graphics acceleration.
define_dual_conv1d_lstm <- function(maxlen, vocab_size, class_label_size, 
                                                 embed_dim=64, 
                                                 dropout=0.3, 
                                                 optimizerName="rmsprop",
                                                 kernel_size1=3,
                                                 kernel_size2=5,
                                                 pooling_size=2,
                                                 filter_list1=c(32,12),
                                                 filter_list2=c(32,12),
                                                 lstm_units=12,
                                                 kernel_activation="relu",
                                                 gpu_flag=FALSE,
                                                 embedding_matrix=NULL,
                                                 freeze_weights=FALSE,
                                                 bidirectional=FALSE,
                                                 lstm_activation="tanh",
                                                 recurrent_activation="sigmoid",
                                                 recurrent_dropout=0,
                                                 lstm_bias=TRUE) {
  
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
  
  
  ## Set the embedding for the input sequence.
  if (!is.null(embedding_matrix) & freeze_weights) {
    get_layer(sequence_encoder_m, index=1) %>%
      set_weights(list(embedding_matrix)) %>%
      freeze_weights()
  } else if (!is.null(embedding_matrix)) {
    get_layer(sequence_encoder_m, index=1) %>%
      set_weights(list(embedding_matrix))
  }
  
  
  # We attempt to reduce the dimensionality using a Convolutional network
  if (length(kernel_size1) < length(filter_list1)) {
    kernel_size1 <- rep(kernel_size1[[1]], length(filter_list1))
  }
  if (length(kernel_size2) < length(filter_list2)) {
    kernel_size2 <- rep(kernel_size2[[1]], length(filter_list2))
  }
  cnn_network_layers1 <- build_cnn_layers(sequence_encoded_m, embed_dim, filter_list1,
                                         kernel_size_list=kernel_size1, 
                                         pooling_size=pooling_size, 
                                         kernel_activation=kernel_activation)
  
  cnn_network_layers2 <- build_cnn_layers(sequence_encoded_m, embed_dim, filter_list2,
                                          kernel_size_list=kernel_size2, 
                                         pooling_size=pooling_size, 
                                         kernel_activation=kernel_activation)
  
  ## we need to concatenate the outputs of the 2 layers togethor.
  cnn_network_layers <- layer_concatenate(c(cnn_network_layers1, cnn_network_layers2), axis=-1)
                                         
  
  lstm_network_layers <- if (bidirectional) {
    cnn_network_layers %>% bidirectional(layer_lstm(units=lstm_units, 
                                                    activation=lstm_activation,
                                                    recurrent_activation=recurrent_activation,
                                                    recurrent_dropout=recurrent_dropout,
                                                    use_bias=lstm_bias))
  } else {
    cnn_network_layers %>% layer_lstm(units=lstm_units,
                                      activation=lstm_activation,
                                      recurrent_activation=recurrent_activation,
                                      recurrent_dropout=recurrent_dropout,
                                      use_bias=lstm_bias)
  }
  # convert back to flattened output
  prediction_layer <- lstm_network_layers %>% 
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, prediction_layer)
  model %>% compile(
    optimizer=optimizerName,
    loss="categorical_crossentropy",
    metrics=c("accuracy")
  )
}


## Combine several conv2d operations and flatten outputs before feeding into a 
## dense softmax layer for classification
# Note can supply NULL for regularizer.
define_shared_input_conv2d <- function(maxlen, vocab_size, class_label_size, 
                                                 embed_dim=64, 
                                                 dropout=0.3, 
                                                 optimizerName="rmsprop",
                                                 kernel_size=c(3),
                                                 kernel_regularizer=regularizer_l1(l=0.01),
                                                 pooling_size=2,
                                                 filter_list=c(32,12),
                                                 kernel_activation="relu",
                                                 embedding_matrix=NULL,
                                                 freeze_weights=FALSE,
                                       batch_norm=TRUE,
                                       stacked_filters = c(),
                                       stacked_kernel_size=c()) {
  
  # Placeholders
  width <- maxlen
  sequence <- layer_input(shape = c(maxlen))
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
      layer_embedding(input_dim = vocab_size, output_dim = embed_dim) %>%
      layer_spatial_dropout_1d(rate=dropout) %>%
      layer_reshape(c(width, embed_dim, 1))
  
  #layer_dropout(rate = dropout) 
  # output: (samples, maxlen, embedding_dim)
  # The layer is reshaped so as to enable a 2 dimensional set of convolutional layers to be constructed
  # with their outputs concatenated size is channels last format
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  
  
  ## Set the embedding for the input sequence.
  if (!is.null(embedding_matrix) & freeze_weights) {
    get_layer(sequence_encoder_m, index=1) %>%
      set_weights(list(embedding_matrix)) %>%
      freeze_weights()
  } else if (!is.null(embedding_matrix)) {
    get_layer(sequence_encoder_m, index=1) %>%
      set_weights(list(embedding_matrix))
  }
  
  if (length(kernel_size) < length(filter_list)) {
    kernel_size <- rep(kernel_size[[1]], length(filter_list))
  }
  
  conv_layers <- c()
  for (i in 1:length(filter_list)) {
    filter <- filter_list[[i]]
    kernel <- kernel_size[[i]]
    layer <- sequence_encoded_m %>% 
      layer_conv_2d(filter, kernel_size=kernel,
                    kernel_regularizer=kernel_regularizer,
                   padding="same",
                   activation=kernel_activation,
                   strides=1,
                   input_shape = list(NULL, vocab_size, embed_dim, 1)) %>%
      layer_max_pooling_2d(pool_size=pooling_size)
    if (batch_norm) {
      layer <- layer %>%
        layer_batch_normalization()
    }
    conv_layers <- c(conv_layers, layer)
    
  }
  concat_conv <- layer_concatenate(conv_layers, trainable=TRUE)
  
  # if the stacked depth is bigger than 0 we can add additional conv_2d layers 
  # above the conv2d concatenated layers.
  last_layer <- concat_conv
  stacked_depth <- length(stacked_filters)
  if (stacked_depth > 0) {
    for (i in 1:stacked_depth) {
      
      last_layer <- last_layer %>% layer_conv_2d(stacked_filters[[i]],
                                                 kernel_size=stacked_kernel_size[[i]],
                                                 kernel_regularizer=kernel_regularizer,
                                                 padding="same",
                                                 activation=kernel_activation,
                                                 strides=1) %>%
        layer_max_pooling_2d(pool_size=pooling_size)
      
      if (batch_norm) {
        last_layer <- last_layer %>%
          layer_batch_normalization()
      }
      
    }
  }
  
  wide_layers <- layer_flatten(last_layer)
  
  # convert back to flattened output
  prediction_layer <- wide_layers %>% 
    layer_dense(class_label_size) %>%
    ## Softmax activation
    layer_activation("softmax")
  
  model <- keras_model(inputs=sequence, prediction_layer)
  
  model %>% compile(
    optimizer=optimizerName,
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
                        checkpointPath="checkpoints/memnet_single.h5",
                        stop_mode="min",
                        stop_patience=5,
                        shuffle=TRUE) {
  callbacks <- list(callback_model_checkpoint(checkpointPath, verbose=1, save_best_only=TRUE), 
                    callback_tensorboard(logdir),
                    callback_early_stopping(mode=stop_mode, patience=stop_patience, restore_best_weights=TRUE, verbose=1))
  
  model %>% fit(
    x = train_vec,
    y = train_target_vec,
    batch_size = 32,
    epochs = numEpochs,
    validation_data = list(valid_vec, valid_target_vec),
    callbacks = callbacks,
    shuffle=shuffle
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
