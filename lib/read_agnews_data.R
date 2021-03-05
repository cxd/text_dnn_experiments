require(dplyr)
require(tidyr)
require(stringr)
source("lib/read_news20.R")

get_path <- function(type="train") {
  
  basedir <- file.path("data", "crepe", "ag_news_csv")
  
  if (type == "train") {
    file <- file.path(basedir, "train.csv")
  } else {
    file <- file.path(basedir, "test.csv")
  }
  file
}


read_classes <- function() {
  file <- file.path("data", "crepe", "ag_news_csv", "classes.txt")
  data <- read.csv(file, header=FALSE, stringsAsFactors=FALSE)
  data$row <- 1:nrow(data)
  colnames(data) <- c("class", "class_num")
  data <- data.frame(
    class_num=data$class_num,
    class=data$class
  )
  data
}

read_ag_file <- function(file) {
  data <- read.delim(file, header=FALSE, sep=",", stringsAsFactors=FALSE)
  colnames(data) <- c("class_num", "title", "text")
  data$class_num <- as.numeric(data$class_num)
  classes <- read_classes()
  data <- left_join(data, classes, by="class_num")
  data
}

create_ag_dataset <- function(data, mergeTitle=TRUE) {
  newsData <- data
  if (mergeTitle) {
    newsData$text <- paste(newsData$title, newsData$text)
  }
  newsData <- data.frame(
    newsgroup=newsData$class,
    text=newsData$text
  )
  process_data <- news_create_data_set(newsData)
  process_data
}


