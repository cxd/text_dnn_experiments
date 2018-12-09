## mgcv is used to generate random vectors for unknown words.
require(mgcv)


model50 <- "data/glove/glove.6B.50d.txt"
model100 <- "data/glove/glove.6B.100d.txt"
model200 <- "data/glove/glove.6B.200d.txt"
model300 <- "data/glove/glove.6B.300d.txt"

## Read the glove model, return it as a matrix or as an environment.
read_glove_model <- function(model) {
  embeddings_index <- new.env(parent = emptyenv())
  lines <- readLines(model)
  for (line in lines) {
    values <- strsplit(line, ' ', fixed = TRUE)[[1]]
    word <- values[[1]]
    coefs <- as.numeric(values[-1])
    embeddings_index[[word]] <- coefs
  }
  embeddings_index
}
# convert embeddings to matrix.
embeddings_to_matrix <- function(embeddings_index) {
  nrows <- length(embeddings_index)
  labels <- names(embeddings_index)
  label1 <- labels[1]
  ncol <- length(embeddings_index[[label1]])
  M <- matrix(0, nrow=nrows, ncol=ncol)
  for(i in 1:nrows) {
    coefs <- embeddings_index[[labels[i]]]
    M[i,] <- coefs
  }
  rownames(M) <- labels
  M
}

### TODO: generate a vector from the existing glove vectors
### for a word that does not appear in the glove matrix.


## Get the vector for supplied word using the environment glove vector.
get_env_vector_for_word <- function(glove, word) {
  word <- stringr::str_to_lower(word)
  row <- glove[[word]]
  row
}

## Generate a sampled vector from the supplied set of vectors.
## this is used to fill the embeddings for unknown words
generate_sampled_vector <- function(M, mu, C) {
  s1 <- rmvn(1,mu,C)
  s1
}

## get the vector for the word or else generate a new vector by random sampling the glove matrix. 
## random vectors for unknown words will be cached in glove.
get_vector_or_generate <- function(glove, M, word, mu, C) {
  word <- stringr::str_to_lower(word)
  if (is.null(glove[[word]])) {
    rvec <- generate_sampled_vector(M, mu, C)
    glove[[word]] <- rvec
    rvec
  } else {
    glove[[word]] 
  }
}


## given a list of words return a list of word vectors 
words_to_vectors <- function(glove, M, words, padding=c()) {
  C <- var(M)
  mu1 <- colMeans(M)
  paddedWords <- unlist(c(padding,words))
  lapply(paddedWords, function(word) {
    get_vector_or_generate(glove, M, word, mu1, C)
  })
}


## given a list of words return a list of word vectors 
## if pad = TRUE then the first entry is padded for the pad text.
## This is used where vocabularies add additional words not identified
words_to_vectors_list <- function(glove, M, words, padding=c()) {
  items <- list()
  temp <- words_to_vectors(glove, M, words, padding)
  for(i in 1:length(temp)) {
    word <- words[[i]]
    items[[word]] <- temp[[i]]
  }
  items
}

## Given a vocabulary obtain the set of vectors for each
## that defines a matrix of dimension N-words x glove-Size
## eg: N-words x 50
## Unknown words generate a random vector.
words_to_position_matrix <- function(glove, M, vocab_size, vocab) {
  cols <- ncol(M)
  C <- var(M)
  mu1 <- colMeans(M)
  rows <- vocab_size
  embedding_matrix <- array(0, c(rows, cols))
  labels <- rownames(M)
  for(word in vocab) {
    index <- which(vocab %in% word)
    vec <- get_vector_or_generate(glove, M, word, mu1, C)
    # The embedding matrix is offset by 1.
    embedding_matrix[index,] <- vec
  }
  embedding_matrix
}
