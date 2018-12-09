
require(jsonlite)
require(dplyr)

source("lib/char_text_processing.R")

## Convert the json data from squad into a data set.
##
prepare_squad_data <- function(path) {
  
  json <- jsonlite::fromJSON(path, simplifyVector=FALSE)
  data1 <- json$data
  
  rows <- lapply(data1, function(row1) {
    title <- row1$title
    paras <- row1$paragraphs
    rows2 <- lapply(paras, function(para) {
      context <- para$context
      qas <- para$qas
      rqrows <- lapply(qas, function(qa) {
        id <- qa$id
        quest <- qa$question
        is_impossible <- qa$is_impossible
        answers <- c(qa$answers, qa$plausible_answers)
        if (is_impossible != TRUE) {
          answerText <- lapply(answers, function(ans) data.frame(plausible_answer=c(ans$text), answer_start=c(ans$answer_start)))
          answersDf <- do.call("rbind", answerText)
          n <- nrow(answersDf)
          answersDf$is_impossible <- rep(is_impossible, n)
          answersDf$id <- rep(id, n)
          answersDf$question <- rep(quest, n)
          answersDf
        } else {
          data.frame(id=c(id), question=c(quest), is_impossible=c(is_impossible), plausible_answer=c(NA), answer_start=c(NA))
        }
      })
      contextDf <- do.call("rbind", rqrows)
      n <- nrow(contextDf)
      contextDf$context <- rep(context, n)
      contextDf
    })
    parasDf <- do.call("rbind", rows2)
    n <- nrow(parasDf)
    parasDf$title <- rep(title,n)
    parasDf
  })
  
  do.call("rbind", rows)
}


prepare_dev_data <- function() {
  path <- "data/squad/dev/dev-v2.0.json"
  df <- prepare_squad_data(path)
  as.data.frame(df)
}

prepare_save_data <- function(output) {
  df <- prepare_dev_data()
  write.csv(df, output, row.names=FALSE)
}

read_saved_data <- function(path) {
  df <- read.csv(path, header=TRUE, stringsAsFactors=FALSE)
  df
}

## Gives the number of questions in the data set.
summary_titles <- function(prepareDf) {
  temp <- data.frame(prepareDf %>% count(title))
  temp
}

questions_to_one_hot_chars <- function(prepareDf, window=60) {
  lapply(prepareDf$questions, function(question) {
    to_one_hot_chars(question, window)
  })
}

context_to_one_hot_chars <- function(prepareDf, window=60) {
  lapply(prepareDf$context, function(context) {
    to_one_hot_chars(context, window)
  })
}

## Convert questions and contexts into one hot encoded character vectors.
## this will result in two separate data frames with a single row for question and context.
convert_to_char_indexes <- function(squadDf, window=60) {
  questions <- questions_to_one_hot_chars(squadDf, window)
  contexts <- context_to_one_hot_chars(squadDf, window)
  list(questions=questions,
       contexts=contexts)
}
