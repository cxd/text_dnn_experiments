
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

newsData <- getPath(type="train", dev=TRUE) %>% 
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


dropout = 0.6

model1 <- define_memnet_single(newsDataset$vocab$maxlen, 
                               newsDataset$vocab$vocab_size, 
                               length(newsDataset$class_labels), 
                               embed_dim=50, 
                               dropout=dropout)

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
numEpochs <- 1000

history1 <- train_model(model1, 
                        train1_x, 
                        val1_x,  
                        train1_y,
                        val1_y,
                        numEpochs=numEpochs,
                        logdir="logs/news/2", 
                        checkpointPath="checkpoints/news_memnet_single.h5")

png("news_memnet_embed.png", width="800", height="600")
plot(history1)
dev.off()

## Save the model.
model1 %>% save_model_weights_hdf5("saved_models/test_news_memnet_single_weights.h5")



testNewsData <- getPath(type="test", dev=TRUE) %>% 
  read_news_file()

testNewsDataset <- create_data_set(testNewsData)

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
evaluate_model(model1, train1_x, train1_y)

evaluate_model(model1, val1_x, val1_y)

## being biased it wont perform well in the test set either.
evaluate_model(model1, test1_x, test1_y)


## what were the class distributions in the training and validation data.
labels1 <- newsData$newsgroup[idx]

labels2 <- newsData$newsgroup[-idx]

labels1Count <- data.frame(labels=labels1) %>% count(labels)
labels1Count

labels2Count <- data.frame(labels=labels2) %>% count(labels)
labels2Count

# the low accuracy could potentially be resolved using more data. 
# or investigate the model architecture.

