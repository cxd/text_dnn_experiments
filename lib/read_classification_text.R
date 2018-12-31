require(data.table)
source("lib/read_glove.R")
source("lib/word_text_processing.R")



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

# Read text and classes.
read_text_and_classes <- function(path, class_column="", text_column="", partition_column=NA, sep=",") {
  data <- fread(path, sep=sep)
  data$row <- as.character(1:nrow(data))
  dataset <- list()
  dataSet <- if (!is.na(partition_column)) {
    data.frame(docId=as.character(data$row),
                          text=as.character(data[,..text_column]),
                          class_label=as.character(data[,..class_column]),
                          partition = as.character(data[,..partition_column]),
                          stringsAsFactors = FALSE)
  } else {
    data.frame(docId=as.character(data$row),
                          text=as.character(data[,..text_column]),
                          class_label=as.character(data[,..class_column]),
                          stringsAsFactors = FALSE)
  }
  dataSet
}



## Convert the input data into a list
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
## the data_set has been read by the function read_text_and_classes.
## class_label - the classification label
## text - the original text
## word_vector - the list of words for the sentence
## class_encoded - a one hot encoded class label, where 1 maps to the current class and 0 the others.
create_data_set <- function(dataset) {
  classLabels <- dataset$text %>% 
    unique() %>% 
    sort()
  
  encodeClass <- function(className) {
    sapply(classLabels, function(x) {
      as.integer(x == className)
    })
  }
  
  outputData <- dataset %>% 
    mutate(word_vector=map(text, ~tokenize_words(.x)),
           class_encoded=map(class_label, ~encodeClass(.x))) 
  
  ## We need to extract the vocabularly
  vocab <- extract_vocab(outputData$word_vector)
  
  list(
    class_labels=classLabels,
    vocab=vocab,
    data_set=outputData
  )
}
