require(data.table)
source("lib/read_glove.R")
source("lib/word_text_processing.R")



## Given the resulting data set.
## Vectorise the word indices for their position in the vocab.
## The list returns
## word_indices - 2d matrix of N sentences x maxlen of index encoded words.
## class_encoded - the set of one-hot encoded classes derived from the create_data_set operation.
vectorise_word_indices <- function(data_set, vocabWords, maxlen, unknownWord="<UNK>") {
  
  
  unknown_idx <- which(unknownWord %in% vocabWords)[[1]]
  
  words_vecs <- as.list(data_set$word_vector)
  
  indexForWordVec <- function(word_vec) {
    idx <- which(word_vec %in% vocabWords)
    unknowns <- word_vec[-idx]
    
    copy_vec <-word_vec
    copy_vec[idx] <- word_vec[idx]
    copy_vec[-idx] <- unknownWord
    # copy_vec includes tokens for the unknownWord at the indices having words not in the vocab
    vidx <- which(vocabWords %in% copy_vec)
    vocab_subset <- vocabWords[vidx]
    
    # the word_vec will be shorter than the vocabWords hence we should be able to tabulate the lookup index on the subset of vocabulary words
    inner_index <- function(word) {
      i <- match(word, vocab_subset)
      vi <- vidx[[i]]
      vi
    }
    
    # all word_vec items have a matching index in the vocab, so we return the indices of the vocab.
    match_idx <- map(copy_vec, function(x) {
        map_int(x, ~inner_index(.x))
    })
    match_idx
  }
  
  word_indices <- map(words_vecs, ~indexForWordVec(.x))
  
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
  setDF(data)
  dataSet <- if (!is.na(partition_column)) {
    data.frame(docId=as.character(data$row),
               text=as.character(data[,text_column]),
               class_label=as.character(data[,class_column]),
               partition = as.character(data[,partition_column]),
               stringsAsFactors = FALSE)
  } else {
    data.frame(docId=as.character(data$row),
               text=as.character(data[,text_column]),
               class_label=as.character(data[,class_column]),
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
  classLabels <- dataset$class_label %>% 
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
