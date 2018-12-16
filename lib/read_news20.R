source("lib/read_glove.R")
source("lib/word_text_processing.R")

### get the path to file.
### type: train or test
### dev: TRUE for mini data set, FALSE otherwise
###
getPath <- function(type="train", dev=TRUE) {
  filepath <- if (type == "train" && dev) {
    file.path("data", "news20", "mini20-train.txt")
  } else if (type == "train") {
    file.path("data", "news20", "20ng-train-all-terms.txt")
  } else if (type == "test" && dev) {
    file.path("data", "news20", "mini20-test.txt")
  } else {
    file.path("data", "news20", "20ng-test-all-terms.txt")
  }
  filepath
}

### Read the news file given the supplied path.
read_news_file <- function(path) {
  data <- read.delim(path, header=FALSE, sep="\t", stringsAsFactors=FALSE)
  colnames(data) <- c("newsgroup", "text")
  data
}

## Convert the news data into a list
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
create_data_set <- function(newsData) {
  classLabels <- newsData$newsgroup %>% 
    unique() %>% 
    sort()
  
  encodeClass <- function(newsGroup) {
    sapply(classLabels, function(x) {
      as.integer(x == newsGroup)
    })
  }
  
  outputData <- newsData %>% 
    mutate(word_vector=map(text, ~tokenize_words(.x)),
           class_encoded=map(newsgroup, ~encodeClass(.x))) 
  
 ## We need to extract the vocabularly
  vocab <- extract_vocab(outputData$word_vector)
  
  list(
    class_labels=classLabels,
    vocab=vocab,
    data_set=outputData
  )
}

## Given the resulting data set.
## Vectorise the word indices for their position in the vocab.
## The list returns
## word_indices - 2d matrix of N sentences x maxlen of index encoded words.
## class_encoded - the set of one-hot encoded classes derived from the create_data_set operation.
vectorise_word_indices <- function(data_set, vocabWords, maxlen, unknownWord="<UNK>") {
  words_vecs <- as.list(data_set$word_vector)
  
  indexForWord <- function(word) {
    idx <- which(word == vocabWords)
    if (length(idx) == 0) {
      idx <- which(unknownWord == vocabWords)  
    }
    idx
  }
  
  word_indices <- map(words_vecs, function(x){
    map_int(x, ~indexForWord(.x))
  })
  list(
    word_indices= pad_sequences(word_indices, maxlen = maxlen),
    class_encoded=data_set$class_encoded
  )
}


## Using the glove data convert the word_vectors
## from the supplied data set into word embeddings.
embeddings_for_word_indices <- function(glove, M, data_set, padding=c()) {
  words_vecs <- as.list(data_set$word_vector)
  embeddings <- map(words_vecs, function(x) {
    word_vecs <- words_to_vectors(glove, M, words, padding=c(""))
    word_vecs
  })
  list(
    word_embeddings=embeddings
  )
}

## Convert input text to a data set.
convert_text_to_dataset <- function(input_text) {
  newsData <- data.frame(text=input_text, stringsAsFactors = FALSE)
  
  outputData <- newsData %>% 
    mutate(word_vector=map(text, ~tokenize_words(.x)))
  
  outputData
}
