## Save the vocabulary from the full training set for use with models trained on the full data.

source("lib/init.R")
library(keras)
library(stringr)
library(purrr)
source("lib/read_news20.R")
source("lib/memnet_singleinput_classifier.R")
source("lib/read_glove.R")

# Setup environment
cfg <- init(getwd())


path <- "data/news20/full_vocab.rds"
## test reading it
vocab <- readRDS(path)

testNewsData <- getPath(type="test", dev=FALSE) %>% 
  read_news_file()

testNewsDataset <- create_data_set(testNewsData)

# use the same vocabulary as the training vocab
testIndexedData <- vectorise_word_indices(testNewsDataset$data_set, 
                                          vocab$vocab, 
                                          vocab$maxlen,
                                          unknownWord="unknown")


test1_x <- testIndexedData$word_indices
test1_x <- as.matrix(test1_x)

test1_y <- testIndexedData$class_encoded
test1_y <- do.call("rbind", test1_y)

modelPath <- "saved_models/full_news_memnet_single.h5"
model1 <- load_model_hdf5(modelPath, compile=TRUE)

evaluate_model(model1, test1_x, test1_y)

