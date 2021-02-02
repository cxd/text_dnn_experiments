
source("lib/init.R")
library(keras)
library(stringr)
library(purrr)

require(lubridate)
source("lib/read_news20.R")
source("lib/memnet_singleinput_classifier.R")
source("lib/read_glove.R")
source("lib/read_classification_text.R")

# Setup environment
cfg <- init(getwd())

newsData <- getPath(type="train", dev=FALSE) %>% 
  read_news_file()

## Get the vocab and the word vector list.
## also encode the classes.
## results in a list with
## class_labels - list of classes in the data set.
## vocab - vocab list object.
## data_set - data frame
##
## The vocab has the following properties
## vocab = list of words in the vocabulary
## vocab_size = the size of the vocabulary
## maxlen = the length of the longest sentence encountered.
##
## the data_set has the following columns
## newsgroup - the text of the newsgroup
## text - the original text
## word_vector - the list of words for the sentence
## class_encoded - a one hot encoded class label, where 1 maps to the current class and 0 the others.
newsDataset <- news_create_data_set(newsData)

# Generate indexed data.
## The list returns
## word_indices - 2d matrix of N sentences x maxlen of index encoded words.
## eg on the dev data set the word indices are 1334 x 7643
## class_encoded - the set of one-hot encoded classes derived from the create_data_set operation.
indexedData <- vectorise_word_indices(newsDataset$data_set, 
                                      newsDataset$vocab$vocab, 
                                      newsDataset$vocab$maxlen)



dropout = 0.1
kernel_size=3

embedding <- c(50,64,128)
filters <- c(32,64,128,256)
lstm_units <- c(32,64,128)

num_samples <- 9

numEpochs <- 10



## break training into train and validation
set.seed(42L)

base_model_name <- "news_conv1d_lstm"
base_checkpoint_dir <- "checkpoints"
base_save_dir <- "saved_models"
base_log_dir <- "logs"

accum_results <- NULL

now_path <- now()

for (i in 1:length(embedding)) {
  # for each embedding we will select a random combination of parameters three times each
  # this avoids a full grid search and allows some random exploration.
  for (j in 1:num_samples) {
    edim <- embedding[[i]]
    f1 <- sample(filters,1)
    f2 <- sample(filters,1)
    l <- sample(lstm_units, 1)
    
    suffix <- paste(c(edim, f1,f2,l), collapse="_")
    
    model_name <- paste(base_model_name, suffix, sep="_")
    checkpoint_file <- paste(model_name, "h5", sep=".")
    save_file <- paste(paste(model_name, suffix, sep="_"), "h5", sep=".")  
    
    logsubdir <- paste(model_name, suffix, sep="_")
    
    checkpoint_file_path <- file.path(base_checkpoint_dir, checkpoint_file)
    save_file_path <- file.path(base_save_dir, save_file)
    
    
    
    model1 <- define_memnet_lstm_conv1d_single_gpu(newsDataset$vocab$maxlen, 
                                                   newsDataset$vocab$vocab_size, 
                                                   length(newsDataset$class_labels), 
                                                   embed_dim=edim, 
                                                   dropout=dropout,
                                                   num_filters = f1,
                                                   num_filters_second=f2,
                                                   lstm_units=l,
                                                   kernel_size = kernel_size,
                                                   gpu_flag=cfg$hasGpu)
    
    
    summary(model1)
    
    
    root_logdir <- file.path(base_log_dir, "news", now_path)
    
    tensorboard(root_logdir)
    
    logdir <- file.path(root_logdir, model_name)
    if (!dir.exists(logdir)) {
      dir.create(logdir, recursive=TRUE)
    }
    
    seq <- 1:nrow(indexedData$word_indices)
    pc <- floor(0.7*length(seq))
    idx <- sample(seq, pc, replace=FALSE)
    
    train1_x <- indexedData$word_indices[idx,]
    train1_x <- as.matrix(train1_x)
    
    train1_y <- indexedData$class_encoded[idx]
    train1_y <- do.call("rbind", train1_y)
    train1_y <- as.matrix(train1_y)
    
    
    val1_x <- indexedData$word_indices[-idx,]
    val1_x <- as.matrix(val1_x)
    
    val1_y <- indexedData$class_encoded[-idx]
    val1_y <- do.call("rbind", val1_y)
    val1_y <- as.matrix(val1_y)
    
    
    history1 <- train_model(model1, 
                            train1_x, 
                            val1_x,  
                            train1_y,
                            val1_y,
                            numEpochs=numEpochs,
                            logdir=logdir, 
                            checkpointPath=checkpoint_file_path)
    
    
    load_model_weights_hdf5(model1, checkpoint_file_path)
    ## Save the model.
    model1 %>% save_model_weights_hdf5(save_file_path)
    
    
    testNewsData <- getPath(type="test", dev=FALSE) %>% 
      read_news_file()
    
    testNewsDataset <- news_create_data_set(testNewsData)
    
    # use the same vocabulary as the training vocab
    testIndexedData <- vectorise_word_indices(testNewsDataset$data_set, 
                                              newsDataset$vocab$vocab, 
                                              newsDataset$vocab$maxlen,
                                              unknownWord="unknown")
    
    
    test1_x <- testIndexedData$word_indices
    test1_x <- as.matrix(test1_x)
    
    test1_y <- testIndexedData$class_encoded
    test1_y <- do.call("rbind", test1_y)
    
    # the model is quite biased if it performs well on the traning data but not well on the validation
    train_eval <- evaluate_model(model1, train1_x, train1_y)
    
    val_eval <- evaluate_model(model1, val1_x, val1_y)
    
    ## being biased it wont perform well in the test set either.
    test_eval <- evaluate_model(model1, test1_x, test1_y)
    
    temp <- data.frame(
      model_name=model_name,
      train_loss=train_eval["loss"],
      train_accuracy=train_eval["accuracy"],
      val_loss=val_eval["loss"],
      val_accuracy=val_eval["accuracy"],
      test_loss=test_eval["loss"],
      test_accuracy=test_eval["accuracy"]
    )
    
    print(train_eval)
    print(val_eval)
    print(test_eval)
    
    if (is.null(accum_results)) {
      accum_results <- temp
    } else {
      accum_results <- rbind(accum_results, temp)
    }
    
  }
  
}

write.csv(accum_results, paste(base_model_name, "csv", sep="."), row.names=FALSE)


eval_data <- read.csv(paste(base_model_name, "csv", sep="."), header=TRUE)

