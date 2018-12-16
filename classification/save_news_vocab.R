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
newsDataset <- create_data_set(newsData)

## To use the vocab with other phases such as testing we need to save it.

path <- "data/news20/full_vocab.rds"
saveRDS(newsDataset$vocab, path)

## test reading it
vocab <- readRDS(path)
names(vocab)
