## This script reads the babi tasks data.
## This example is based on the keras examples https://keras.rstudio.com/articles/examples/babi_memnn.html
##
## The references there are: 
## References:
##  
##  Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush, “Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks”, http://arxiv.org/abs/1502.05698
##
## Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, “End-To-End Memory Networks”, http://arxiv.org/abs/1503.08895
##
##

library(keras)
library(readr)
library(stringr)
library(purrr)
library(tibble)
library(dplyr)
source("lib/lstm_sequence_learner.R")



challenges <- list(
  # QA1 with 10,000 samples
  single_supporting_fact_10k = "%stasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_%s.txt",
  # QA2 with 10,000 samples
  two_supporting_facts_10k = "%stasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_%s.txt",
  # QA3 Three supporting facts
  three_supporting_facts_10k = "%stasks_1-20_v1-2/en-10k/qa2_three-supporting-facts_%s.txt",
  # QA3 Three supporting facts
  three_supporting_facts_10k = "%stasks_1-20_v1-2/en-10k/qa2_three-supporting-facts_%s.txt",
  # QA4 two arg relations
  two_arg_relations_10k = "%stasks_1-20_v1-2/en-10k/qa4_two-arg-relations_%s.txt",
  # QA5 three arg relations
  three_arg_relations_10k = "%stasks_1-20_v1-2/en-10k/qa5_three-arg-relations_%s.txt",
  # QA6 yes no questions
  yes_no_questions_10k = "%stasks_1-20_v1-2/en-10k/qa6_yes-no-questions_%s.txt",
  # QA7 yes no questions
  counting_questions_10k = "%stasks_1-20_v1-2/en-10k/qa7_counting_%s.txt",
  # QA8 lists sets
  lists_sets_10k = "%stasks_1-20_v1-2/en-10k/qa8_lists-sets_%s.txt",
  # QA9 simple negation
  simple_negation_10k = "%stasks_1-20_v1-2/en-10k/qa9_simple-negation_%s.txt",
  # QA10 Indefinite knowledge
  indefinate_knowledge_10k =  "%stasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_%s.txt",
  # QA11 Basic coreference
  basic_coreference_10k =  "%stasks_1-20_v1-2/en-10k/qa11_basic-coreference_%s.txt"
  ## TODO Add other types for remaining qa12 - qa20
)

default_challenge_type <- "single_supporting_fact_10k"
max_length <- 999999


## Download the data set for training.
## Return the path to the location the data was downloaded.
## The URLs are from the original source code.
## Data can also be accessed via facebook: https://research.fb.com/downloads/babi/
##
## Github project for the babi data set is:
## https://github.com/facebook/bAbI-tasks
##
download_babi_data <- function(baseUrl="https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz") {
  
  # Download data
  path <- get_file(
    fname = "babi-tasks-v1-2.tar.gz",
    origin = baseUrl
  )
  untar(path, exdir = str_replace(path, fixed(".tar.gz"), "/"))
  path <- str_replace(path, fixed(".tar.gz"), "/")
  path
}

## Tokenise text sequence into lists of words including punctuation. 
tokenize_words <- function(x){
  x <- x %>% 
    str_replace_all('([[:punct:]]+)', ' \\1') %>% 
    str_split(' ') %>%
    unlist()
  x[x != ""]
}

## Parse stories, questions and answers.
## The babi test set has the structure where
## There is a line index for text line
## The question s a line that contains a question mark, after which the answer and one or more space delimited line indexes.
## For example:
##
##
##
parse_stories <- function(lines, only_supporting = FALSE){
  lines <- lines %>% 
    str_split(" ", n = 2) %>%
    map_df(~tibble(nid = as.integer(.x[[1]]), line = .x[[2]]))
  
  lines <- lines %>%
    mutate(
      split = map(line, ~str_split(.x, "\t")[[1]]),
      q = map_chr(split, ~.x[1]),
      a = map_chr(split, ~.x[2]),
      supporting = map(split, ~.x[3] %>% str_split(" ") %>% unlist() %>% as.integer()),
      story_id = c(0, cumsum(nid[-nrow(.)] > nid[-1]))
    ) %>%
    select(-split)
  
  stories <- lines %>%
    filter(is.na(a)) %>%
    select(nid_story = nid, story_id, story = q)
  
  questions <- lines %>%
    filter(!is.na(a)) %>%
    select(-line) %>%
    left_join(stories, by = "story_id") %>%
    filter(nid_story < nid)
  
  if(only_supporting){
    questions <- questions %>%
      filter(map2_lgl(nid_story, supporting, ~.x %in% .y))
  }
  
  questions %>%
    group_by(story_id, nid, question = q, answer = a) %>%
    summarise(story = paste(story, collapse = " ")) %>%
    ungroup() %>% 
    mutate(
      question = map(question, ~tokenize_words(.x)),
      story = map(story, ~tokenize_words(.x)),
      id = row_number()
    ) %>%
    select(id, question, answer, story)
}

## Read the train and test data for use in later functions.
## Supply the challenge type.
## Challenge type defaults to: single_supporting_fact_10k
read_train_test <- function(path, challenge_type=default_challenge_type) {
  
  challenge <- challenges[[challenge_type]]
  
  # Reading training and test data
  train <- read_lines(sprintf(challenge, path, "train")) %>%
    parse_stories() %>%
    filter(map_int(story, ~length(.x)) <= max_length)
  
  test <- read_lines(sprintf(challenge, path, "test")) %>%
    parse_stories() %>%
    filter(map_int(story, ~length(.x)) <= max_length)
  
  list(train=train,
    test=test)
}

## After reading the stories extract the vocabulary
## Supply train and test data read via the function parse stories
extract_vocab <- function(train, test) {
  
  # Extract the vocabulary
  all_data <- bind_rows(train, test)
  vocab <- c(unlist(all_data$question), all_data$answer, 
             unlist(all_data$story)) %>%
    unique() %>%
    sort()
  
  # Reserve 0 for masking via pad_sequences
  vocab_size <- length(vocab) + 1
  story_maxlen <- map_int(all_data$story, ~length(.x)) %>% max()
  query_maxlen <- map_int(all_data$question, ~length(.x)) %>% max()
  
  list(
    vocab=vocab,
    vocab_size=vocab_size,
    story_maxlen=story_maxlen,
    query_maxlen=query_maxlen
  )
  
}

## A helper function used to take a new story and question and convert it into a list
## structure suitable for prediction.
process_plain_text <- function(story, question, answer) {
  story_words <- tokenize_words(story)
  story_questions <- tokenize_words(question)
 temp <- list(story=list(story_words),
             question=list(story_questions),
             answer=c(answer))
  as.tibble(temp)
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
## "stories" is a matrix of word indices in the vocab. Nrows = num stories, ncols = max width of vocab.
## values in the vector are an index into the vocab list looking up stories[i,j] will give the index of the word in the vocab.
##
## "questions" are encoded in a fixed width fo max question size.
## rows are num questions, cols are max question size.
## values in each row vector are the index of the word in the vocab.
##
## "answers" is nrow = num records ncol = num words in train data.
## It is a one hot encoded vector for the words that appear in the answer. In this case the answer is a single
## word and the one hot encoding acts as a discrete classification label per story, question pair.
## The network learns a softmax function over the answers, but has no ordering ability.
##
## punctuation is included in the vocab as it represents terminal symbols in the sequence.
vectorize_stories <- function(data, vocab, story_maxlen, query_maxlen){
  
  questions <- map(data$question, function(x){
    map_int(x, ~which(.x == vocab))
  })
  
  stories <- map(data$story, function(x){
    map_int(x, ~which(.x == vocab))
  })
  
  # "" represents padding
  answers <- sapply(c("", vocab), function(x){
    as.integer(x == data$answer)
  })
  
  list(
    questions = pad_sequences(questions, maxlen = query_maxlen),
    stories   = pad_sequences(stories, maxlen = story_maxlen),
    answers   = answers
  )
}

vectorize_dataset <- function(train, test, vocab, story_maxlen, query_maxlen) {
  ## punctuation is included in the vocab as it represents terminal symbols in the sequence.
  train_vec <- vectorize_stories(train, vocab, story_maxlen, query_maxlen)
  test_vec <- vectorize_stories(test, vocab, story_maxlen, query_maxlen)
  
  list(
    train_vec=train_vec,
    test_vec=test_vec
  )
}

## define the memory network model for the input representation 
## used in the example. 
define_memnet_babi <- function(story_maxlen, query_maxlen, vocab_size) {
  # Defining the model ------------------------------------------------------
  
  # Placeholders
  sequence <- layer_input(shape = c(story_maxlen))
  question <- layer_input(shape = c(query_maxlen))
  
  # Encoders
  # Embed the input sequence into a sequence of vectors
  sequence_encoder_m <- keras_model_sequential()
  sequence_encoder_m %>%
    layer_embedding(input_dim = vocab_size, output_dim = 64) %>%
    layer_dropout(rate = 0.3)
  # output: (samples, story_maxlen, embedding_dim)
  
  # Embed the input into a sequence of vectors of size query_maxlen
  sequence_encoder_c <- keras_model_sequential()
  sequence_encoder_c %>%
    layer_embedding(input_dim = vocab_size, output = query_maxlen) %>%
    layer_dropout(rate = 0.3)
  # output: (samples, story_maxlen, query_maxlen)
  
  # Embed the question into a sequence of vectors
  question_encoder <- keras_model_sequential()
  question_encoder %>%
    layer_embedding(input_dim = vocab_size, output_dim = 64, 
                    input_length = query_maxlen) %>%
    layer_dropout(rate = 0.3)
  # output: (samples, query_maxlen, embedding_dim)
  
  # Encode input sequence and questions (which are indices)
  # to sequences of dense vectors
  sequence_encoded_m <- sequence_encoder_m(sequence)
  sequence_encoded_c <- sequence_encoder_c(sequence)
  question_encoded <- question_encoder(question)
  
  # Compute a 'match' between the first input vector sequence
  # and the question vector sequence
  # shape: `(samples, story_maxlen, query_maxlen)`
  match <- list(sequence_encoded_m, question_encoded) %>%
    layer_dot(axes = c(2,2)) %>%
    layer_activation("softmax")
  
  # Add the match matrix with the second input vector sequence
  response <- list(match, sequence_encoded_c) %>%
    layer_add() %>%
    layer_permute(c(2,1))
  
  # Concatenate the match matrix with the question vector sequence
  answer <- list(response, question_encoded) %>%
    layer_concatenate() %>%
    # The original paper uses a matrix multiplication for this reduction step.
    # We choose to use an RNN instead.
    layer_lstm(32) %>%
    # One regularization layer -- more would probably be needed.
    layer_dropout(rate = 0.3) %>%
    layer_dense(vocab_size) %>%
    # We output a probability distribution over the vocabulary
    layer_activation("softmax")
  
  # Build the final model
  model <- keras_model(inputs = list(sequence, question), answer)
  model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  
  model
}


train_model <- function(model, 
                        train_vec, 
                        validation_vec,  
                        numEpochs=120,
                        logdir="logs/babi", 
                        checkpointPath="checkpoints/babimodel.h5") {
  callbacks <- list(callback_model_checkpoint(checkpointPath))
  model %>% fit(
    x = list(train_vec$stories, train_vec$questions),
    y = train_vec$answers,
    batch_size = 32,
    epochs = numEpochs,
    validation_data = list(list(validation_vec$stories, validation_vec$questions), validation_vec$answers),
    callbacks = callbacks
  )
}

evaluate_model <- function(model, test_vec) {
  model %>% evaluate(
    x=list(test_vec$stories, test_vec$questions),
    y=test_vec$answers,
    batch_size=32
  )
}

swap_out_of_vocab <- function(words, vocab, swap=".") {
  words <- unlist(words)
  in_vocab <- which(words %in% vocab)
  
  not_in_vocab <- words[-in_vocab]
  if (length(not_in_vocab) > 0) {
    words[-in_vocab] <- swap
  }
  list(words)
}

## Predict an answer given the context and the question.
## Note both context and question must have words that occur in the known vocabulary.
predict_answer <- function(model, vocab, story_maxlen, query_maxlen, context, question, plausible_ans=".") {
  ## parse both context and question into the representation used by the network input.
  pred_data <- process_plain_text(context, question, plausible_ans)
  
  ## We need to remove any words not in the vocab from the prediction data.
  pred_data$story <- swap_out_of_vocab(pred_data$story, vocab)
  pred_data$question <- swap_out_of_vocab(pred_data$question, vocab)
  
  pred_vec <- vectorize_stories(pred_data, vocab, story_maxlen, query_maxlen)
  
  pred_ans <- model %>% predict(x=list(pred_vec$stories, pred_vec$questions))
  
  pred <- softmax_index_list(pred_ans)
  ans <- "UNKNOWN"
  if (pred$index[1] > 1) {
    idx <- pred$index - 1
    ans <- vocab[idx]
  }
  list(
    answer=ans,
    confidence=pred$confidence)
}
