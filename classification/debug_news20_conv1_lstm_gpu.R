
source("lib/init.R")
library(keras)
library(stringr)
library(purrr)
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

embedding <- 128
filter_list <- c(512,256,64)
#filter_list <- c(512,256,128,64)
kernel_size <- c(5, 3, 3, 3)
lstm_units <- 64

base_model_name <- "news_conv1d_lstm"
base_save_dir <- "saved_models"

model1 <- define_memnet_lstm_conv1d_single_gpu(newsDataset$vocab$maxlen, 
                                               newsDataset$vocab$vocab_size, 
                                               length(newsDataset$class_labels), 
                                               embed_dim=embedding, 
                                               dropout=dropout,
                                               filter_list = filter_list,
                                               lstm_units=lstm_units,
                                               kernel_size = kernel_size,
                                               gpu_flag=cfg$hasGpu,
                                               bidirectional=TRUE)


load_checkpoints <- TRUE

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

# each epoch tskers around

# 1 epoch is about 300s
# 10 epochs is about 50 minutes.
# 100 epochs is about 8 hours.
# 
numEpochs <- 100


require(lubridate)
path <- now()

logdir <- file.path("logs", "news", path)
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

#png("news_memnet_embed_lstm_gpu.png", width="800", height="600")
#plot(history1)
#dev.off()


load_model_weights_hdf5(model1, checkpoint_file)
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

test_pred <- predict(model1, test1_x)

# convert the test prediction table into a table where the maximum is set to 1.
test_M <- matrix(data=0,nrow=nrow(test_pred), ncol=ncol(test_pred))

rows <- nrow(test_pred)
for (i in 1:rows) {
  idx <- which(test_pred[i,] == max(test_pred[i,]))
  test_M[i,idx] <- 1
}
# TODO: map the newsgroup labels to the predictions and the original test data
# draw confusability matrix.


## what were the class distributions in the training and validation data.
labels1 <- newsData$newsgroup[idx]



labels2 <- newsData$newsgroup[-idx]

labels1Count <- data.frame(labels=labels1) %>% count(labels)
labels1Count

labels2Count <- data.frame(labels=labels2) %>% count(labels)
labels2Count


