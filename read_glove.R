
model50 <- "data/glove/glove.6B.50d.txt"
model100 <- "data/glove/glove.6B.100d.txt"
model200 <- "data/glove/glove.6B.200d.txt"
model300 <- "data/glove/glove.6B.300d.txt"

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

## Get the vector for supplied word using the environment glove vector.
get_env_vector_for_word <- function(glove, word) {
  word <- stringr::str_to_lower(word)
  labels <- names(glove)
  row <- glove[[word]]
  row
}