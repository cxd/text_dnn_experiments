
source("lib/init.R")
library(keras)
library(stringr)
library(purrr)
source("lib/prepare_squad_data.R")
source("lib/read_glove.R")
source("lib/char_text_processing.R")
source("lib/read_news20.R")
source("lib/memnet_singleinput_classifier.R")
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

text <- newsDataset$data_set$text

# use method to extract character sequences of max characters 1024
# this is the same as word indices but does not require a word vocabulary only a character level vocabulary.
# similarly the max width of the character feature space can be increased if required depending on resources.
# smaller widths are suitable for short utterances which is task specific such as reviews, or chat classification, or sentiment analysis.
max_char_width <- 5000
char_data <- convert_to_char_index_sequences(text, max_width=max_char_width)

# now we can prepare the same type of architecture and perform the data split for training.


dropout = 0.1

embedding <- 512
# article suggests increasing filters
filter_list <- c(64, 64, 64, 128, 128, 256, 256, 512, 512)
#filter_list <- c(512,256,128,64)
kernel_size <- c(3, 3, 3, 3)
lstm_units <- 64


kernel_regularizer <- NULL
batch_norm <- TRUE

base_model_name <- "news_char_conv1d_lstm_batch_norm"
base_save_dir <- "saved_models"

# We will first try the 1 dimensional CNN LSTM on the character level.
# However the article suggests increasing the number of filters instead of decreasing them (opposite to pyramid)
model1 <- define_memnet_lstm_conv1d_single_gpu(max_char_width, 
                                               char_data$num_indices, 
                                               length(newsDataset$class_labels), 
                                               embed_dim=embedding, 
                                               dropout=dropout,
                                               filter_list = filter_list,
                                               lstm_units=lstm_units,
                                               kernel_size = kernel_size,
                                               kernel_regularizer = kernel_regularizer,
                                               gpu_flag=cfg$hasGpu,
                                               bidirectional=TRUE,
                                               batch_norm=batch_norm)



load_checkpoints <- FALSE

suffix <- paste(c(embedding, filter_list, lstm_units), collapse="_")

model_name <- paste(base_model_name, suffix, sep="_")
checkpoint_file <- paste(model_name, "h5", sep=".")
save_file <- paste(paste(model_name, suffix, sep="_"), "h5", sep=".")  

save_file_path <- file.path(base_save_dir, save_file)

if (load_checkpoints & file.exists(checkpoint_file)) {
  load_model_weights_hdf5(model1, checkpoint_file)
}


summary(model1)



## break training into train and validation
set.seed(42L)

seq <- 1:nrow(char_data$sequence_df)
pc <- floor(0.7*length(seq))
idx <- sample(seq, pc, replace=FALSE)

train1_x <- char_data$sequence_df[idx,]
train1_x <- as.matrix(train1_x)

train1_y <- newsDataset$data_set$class_encoded[idx]
train1_y <- do.call("rbind", train1_y)
train1_y <- as.matrix(train1_y)


val1_x <- char_data$sequence_df[-idx,]
val1_x <- as.matrix(val1_x)

val1_y <- newsDataset$data_set$class_encoded[-idx]
val1_y <- do.call("rbind", val1_y)
val1_y <- as.matrix(val1_y)




# each epoch tskers around

# 1 epoch is about 300s
# 10 epochs is about 50 minutes.
# 100 epochs is about 8 hours.
# 
numEpochs <- 100


require(lubridate)
path <- now()

logdir <- file.path("logs", "news_char", path)
if (!dir.exists(logdir)) {
  dir.create(logdir, recursive=TRUE)
}

tensorboard(logdir)

history1 <- train_model(model1, 
                        train1_x, 
                        val1_x,  
                        train1_y,
                        val1_y,
                        numEpochs=numEpochs,
                        logdir=logdir, 
                        checkpointPath=checkpoint_file)




load_model_weights_hdf5(model1, checkpoint_file)
## Save the model.
model1 %>% save_model_weights_hdf5(save_file_path)




testNewsData <- getPath(type="test", dev=FALSE) %>% 
  read_news_file()

testNewsDataset <- news_create_data_set(testNewsData)


test_text <- testNewsDataset$data_set$text

test_char_data <- convert_to_char_index_sequences(test_text, max_width=max_char_width)


test1_x <- test_char_data$sequence_df
test1_x <- as.matrix(test1_x)

test1_y <- testNewsDataset$data_set$class_encoded
test1_y <- do.call("rbind", test1_y)

# the model is quite biased if it performs well on the traning data but not well on the validation
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
