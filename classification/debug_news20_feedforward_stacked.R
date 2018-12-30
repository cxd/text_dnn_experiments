
source("lib/init.R")
library(keras)
library(stringr)
library(purrr)
source("lib/read_news20.R")
source("lib/memnet_singleinput_classifier.R")
source("lib/read_glove.R")
source("lib/embedding_feedforward_softmax.R")

# Setup environment
cfg <- init(getwd())

devSet<- FALSE

newsData <- getPath(type="train", dev=devSet) %>% 
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
newsDataset <- list() 
if (file.exists("data/news20/newsDataset_full.rds")) {
  newsDataset <- readRDS("data/news20/newsDataset_full.rds")
} else {
  
  newsDataset <- create_data_set(newsData)
  saveRDS(newsDataset, "data/news20/newsDataset_full.rds")
  
}


# Generate indexed data.
## The list returns
## word_indices - 2d matrix of N sentences x maxlen of index encoded words.
## eg on the dev data set the word indices are 1334 x 7643
## class_encoded - the set of one-hot encoded classes derived from the create_data_set operation.
indexedData <- list()
if (file.exists("data/news20/indexed_data_full.rds")) {
  
  indexedData <- readRDS("data/news20/indexedData_full.rds")
  
} else {
  
  indexedData <- vectorise_word_indices(newsDataset$data_set, 
                                        newsDataset$vocab$vocab, 
                                        newsDataset$vocab$maxlen)
  
  saveRDS(indexedData, "data/news20/indexedData_full.rds")
  
}

dropout = 0.6

model1 <- embedding_feedforward_stacked_cnn_softmax(newsDataset$vocab$maxlen, 
                               newsDataset$vocab$vocab_size, 
                               length(newsDataset$class_labels), 
                               embed_dim=64, 
                               dropout=dropout,
                               optimizerName="nadam")

if (file.exists("saved_models/test_news_feedforward_cnn_weights.h5")) {
  model1 <- load_model_weights_hdf5(model1, "saved_models/test_news_feedforward_cnn_weights.h5")
}

load_checkpoint <- FALSE

if (load_checkpoint && file.exists("checkpoints/news_feedforward_cnn.h5")) {
  model1 <- load_model_hdf5("checkpoints/news_feedforward_cnn.h5")
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
numEpochs <- 500

history1 <- train_model(model1, 
                        train1_x, 
                        val1_x,  
                        train1_y,
                        val1_y,
                        numEpochs=numEpochs,
                        logdir="logs/news/2", 
                        checkpointPath="checkpoints/news_feedforward_cnn.h5")

png("news_feedforward_cnn.png", width="800", height="600")
plot(history1)
dev.off()

## Save the model.
model1 %>% save_model_hdf5("saved_models/test_news_feedforward_cnn.h5")

model1 %>% save_model_weights_hdf5("saved_models/test_news_feedforward_cnn_weights.h5")

## Save the vocal and the maximum length of the data set as well as the class labels.
write.csv(newsDataset$vocab$vocab, "saved_models/news20_full_vocab.csv", row.names=FALSE)


## Save the vocal and the maximum length of the data set as well as the class labels.
write.csv(history1, "saved_models/news20_full_cnn_history.csv", row.names=FALSE)


testNewsData <- getPath(type="test", dev=devSet) %>% 
  read_news_file()

testNewsDataset <- list()
if (file.exists("data/news20/test_news_dataset_full.rds")) {
  testNewsDataset <- readRDS("data/news20/test_news_dataset_full.rds")
} else {
  testNewsDataset <- create_data_set(testNewsData)
  saveRDS(testNewsDataset, "data/news20/test_news_dataset_full.rds")
}

testIndexedData <- list()
if (file.exists("data/news20/test_indexed_data_full.rds")) {
  testIndexedData <- readRDS("data/news20/test_indexed_data_full.rds")
} else {
  # use the same vocabulary as the training vocab
  testIndexedData <- vectorise_word_indices(testNewsDataset$data_set, 
                                            newsDataset$vocab$vocab, 
                                            newsDataset$vocab$maxlen,
                                            unknownWord="unknown")
  saveRDS(testIndexedData, "data/news20/test_indexed_data_full.rds")
  
}


test1_x <- testIndexedData$word_indices
test1_x <- as.matrix(test1_x)

test1_y <- testIndexedData$class_encoded
test1_y <- do.call("rbind", test1_y)

# the model is quite biased if it performs well on the traning data but not well on the validation
evaluate_model(model1, train1_x, train1_y)

evaluate_model(model1, val1_x, val1_y)

## being biased it wont perform well in the test set either.
evaluate_model(model1, test1_x, test1_y)


## what were the class distributions in the training and validation data.
labels1 <- newsData$newsgroup[idx]

labels2 <- newsData$newsgroup[-idx]

temp <- newsDataset$data_set
temp$text_size <- sapply(temp$word_vector, length)
temp <- temp %>%  group_by(newsgroup) %>%
  summarize(maxlen=max(text_size))

names(temp) <- c("label", "text_size")

labels1Count <- data.frame(labels=labels1) %>% count(labels)
names(labels1Count) <- c("label", "trainCount")

labels2Count <- data.frame(labels=labels2) %>% count(labels)
names(labels2Count) <- c("label", "testCount")

labelsAll <- inner_join(inner_join(labels1Count, labels2Count, by="label"), temp, by="label")

write.csv(labelsAll, "saved_models/feedforward_labels.csv", row.names=FALSE)
# the low accuracy could potentially be resolved using more data. 
# or investigate the model architecture.

