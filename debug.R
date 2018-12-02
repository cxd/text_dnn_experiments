
source("lib/init.R")
source("lib/prepare_squad_data.R")
source("lib/read_glove.R")

# Setup environment
cfg <- init(getwd())

squadData <- "data/squad/dev/preprocessed.csv"
squadDf <- read_saved_data(squadData)
glove <- read_glove_model(model50)
## in matrix form.
M <- embeddings_to_matrix(glove)

## Playing with the glove vectors for semantic locations.
v1 <- get_env_vector_for_word(glove, "man")
v2 <- get_env_vector_for_word(glove, "woman")

v3 <- get_env_vector_for_word(glove, "king")
v4 <- get_env_vector_for_word(glove, "queen")


unk <- get_vector_or_generate(glove, M, "notinthelist")

words <- list("this", "is" , "a", "test")
word_vecs <- words_to_vectors(glove, M, words)


v5 <- v3 + v2 - v1

xlim <- c(min(v1[1],v2[1],v3[1],v4[1],v5[1]), max(v1[1],v2[1],v3[1],v4[1],v5[1]))
ylim <- c(min(v1[2],v2[2],v3[2],v4[2],v5[2]), max(v1[2],v2[2],v3[2],v4[2],v5[2]))

plot(c(v1[1],v2[1]), c(v1[2],v2[2]), xlim=xlim, ylim=ylim, main="Example ordination in 2d with 'king + woman - man'")
lines(c(v1[1],v2[1]), c(v1[2],v2[2]), col="gray", xlim=xlim, ylim=ylim)
text(c(v1[1]), c(v1[2]), labels="man", col="blue", xlim=xlim, ylim=ylim)
text(c(v2[1]), c(v2[2]), labels="woman", col="blue", xlim=xlim, ylim=ylim)
points(c(v3[1],v4[1]), c(v3[2],v4[2]), xlim=xlim, ylim=ylim)
lines(c(v3[1],v4[1]), c(v3[2],v4[2]), col="gray", xlim=xlim, ylim=ylim)
text(c(v3[1]), c(v3[2]), labels="king", col="blue", xlim=xlim, ylim=ylim)
text(c(v4[1]), c(v4[2]), labels="queen", col="blue", xlim=xlim, ylim=ylim)
points(c(v5[1]), c(v5[2]))
text(c(v5[1]), c(v5[2]), labels="king + woman - man", col="blue")
lines(c(v3[1],v5[1]), c(v3[2],v5[2]), col="gray", xlim=xlim, ylim=ylim)







