
require(jsonlite)
require(dplyr)

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


to_unique_chars <- function(textset) {
  t <- sapply(textset, function(text) { 
    unique(strsplit(text,"", useBytes=TRUE)[[1]])
    })
  unique(sort(do.call("c",t)))
}

## This generates a sliding window of one hot encoded characters for the text.
## The outputs are two components.
## x_data : the first component x is a tensor of (i , t , k)
## i is the ith row of the sliding window of cols width
## t is the jth index of the columns in the window, it acts as a time series
## k is the index of the one hot encoded character from the position window[i,j]
## the length of k is a vector having the number of supported chars in the alphabet of this model (see set_of_chars)
## Each character in the sentence is treated as one of a set of values.
##
## y_data:
## This is a matrix of dimension (i, k) 
## Each row is a one hot encoded vector of dimension set_of_chars with the kth item matching the next character
## following the sequence i from the window being set to 1.
## all other characters are 0.
##
to_one_hot_chars <- function(text, window=60) {
  
  chars <- set_of_chars()
  char_indices <- char_index_set()
  cols <- length(chars)
  
  ## convert to a index of characters
  sequence1 <- to_char_index_sequence(char_indices, text)
  ## create lagged windowed sequence
  windows <- window_char_sequence(sequence1, window)
  
  rows <- nrow(windows)
 
  x <- array(0L, dim=c(rows, ncol(windows), cols))
  y <- array(0L, dim=c(rows, cols))
  
  ## predicting each end character after each row of windows 
  ## we have the y characters being the last column of windows
  next_chars <- windows[,ncol(windows)]
  
  ## each row of windows is a sentence.
  ## we create a tensor for each row of windows
  ## i,t,k where i is the row, t is the time interval and k is the position of the character index.
  for(i in 1:nrow(windows)) {
    for (t in 1:ncol(windows)) {
      idx <- windows[i,t]
      k <- char_indices[idx]
      x[i,t,k] <- 1
    }
    next_char_idx <- next_chars[i]
    y[i,next_char_idx] <- 1
  }
  list(data_x=x,
       data_x_dim=dim(x),
       data_y=y,
       data_y_dim=dim(y))
}

set_of_chars <- function() c("\n"," ","_","-",",",";",":","!","?",".","'","\"",
                               "(",")","[","]","{","}","*","/","&","#","%","`",
                               "+","<","=",">","|","~","$","0","1","2","3","4",
                               "5","6","7","8","9","a","A","b","B","c","C","d",
                               "D","e","E","f","F","g","G","h","H","i","I","j",
                               "J","k","K","l","L","m","M","n","N","o","O","p",
                               "P","q","Q","r","R","s","S","t","T","u","U","v",
                               "V","w","W","x","X","y","Y","z","Z")

char_index_set <- function() {
  chars <- set_of_chars()
  indices <- 1:length(chars)
  names(indices) <- chars
  indices
}

## Given text convert into a sequence of chars.
## any unknown char is identified as whitespace (ie: ignored)
to_char_index_sequence <- function(char_indices, text) {
  chars <- names(char_indices)
  spaceIdx <- which(chars %in% " ")
  sentence <- strsplit(text,"")[[1]]
  index_seq <- sapply(sentence, function(s) {
    idx <- which(chars %in% s)
    if (length(idx) == 0) {
      spaceIdx
    } else idx
  })
  index_seq
}


slidingMatrix <- function(sequence, p=1) {
  sequence <- sequence[!is.nan(sequence)]
  # note n' = n - p + 1
  n1 <- length(sequence)
  # new number of rows
  n2 <- length(sequence) - p + 1
  if (n2 < p) {
    n2 <- p
  }
  nextSeq <- sapply(1:n2, function(i) {
    sapply(1:p, function(j) {
      k <- i+j-1
      if (k <= length(sequence)) {
        sequence[k]  
      } else 0.0
    })
  })
  nextSeq <- unlist(nextSeq, recursive=F)
  t1 <- floor(length(nextSeq)/60)
  if (length(nextSeq) %% 60 > 0) {
    t1 <- t1 + 1
  }
  if (n2 <= p) {
    n2 <- t1
  }
  m <- matrix(0, nrow=n2, ncol=p, byrow=TRUE)
  m[1:length(nextSeq)] <- nextSeq
  as.matrix(m)
}


window_char_sequence <- function(index_seq, window=60) {
  slidingMatrix(index_seq, window)
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
