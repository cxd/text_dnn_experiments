require(purrr)
require(dplyr)
source("lib/read_glove.R")


## Tokenise text sequence into lists of words including punctuation. 
tokenize_words <- function(x){
  x <- x %>% 
    str_replace_all('([[:punct:]]+)', ' \\1') %>% 
    str_split(' ') %>%
    unlist()
  x[x != ""]
}




## After reading the stories extract the vocabulary
## Supply list of sentences where each is a list of words.
## That is the dataframe contains lists of words for each sentence. 
### Each row in the data frame is a sentence.  
## The vocab is the list of unique words.
## The vocab_size is the number of unique words.
## the maxlen is the maximum length of an input sentence.
extract_vocab <- function(sentences, unknownToken=c("<UNK>")) {
  
  # Extract the vocabulary
  vocab <- c(unlist(sentences)) %>%
    unique() %>%
    sort()
  
  vocab <- c(unknownToken, vocab)
  
  # Reserve 0 for masking via pad_sequences
  vocab_size <- length(vocab) + 1
  maxlen <- map_int(sentences, ~length(.x)) %>% max()
  list(
    vocab=vocab,
    vocab_size=vocab_size,
    maxlen=maxlen
  )
}


# Vectorized versions of training and test sets
##
## The data is a list of train and test.
## Each train and test inputs are in turn data frames with the names (id, question, story, answer)
## "id" - The id is a integer (n observations of unique ids)
## "question" - The question is a list of individual words from the vocabulary that define the actual question for the story.
## "story" - The story is a list of words from the vocabulary that defines the context, it can have multiple lines, but these are all 
## concatenated into one list.
## "answer" - answer is a one word answer for each row of the data frame. N rows for n answers.
##
## The structure of the training and test data is as follows.
##
## "sentences" is a matrix of word indices in the vocab. Nrows = num stories, ncols = max width of vocab.
## values in the vector are an index into the vocab list looking up stories[i,j] will give the index of the word in the vocab.
##
## punctuation is included in the vocab as it represents terminal symbols in the sequence.
vectorize_sentences <- function(sentences, vocab, maxlen) {
  
  wordIndices <- map(sentences, function(x){
    map_int(x, ~which(.x == vocab))
  })
  
  list(
    wordIndices = pad_sequences(wordIndices, maxlen = maxlen)
  )
}

## using the glove embeddings
## generate a data set where
## inputs may be padded by supplying an additional padding vector.
## outputs list of embedding matrices. One matrix per sentence.
sentences_embedding_matrices <- function(sentences, glove, M, padding=c()) {
  embeddings <- map(sentences, function(x) {
    word_vecs <- words_to_vectors(glove, M, words, padding=padding)
    word_vecs
  })
  
  list(
    embeddings = embeddings
  )
}
