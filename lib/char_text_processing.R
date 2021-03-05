require(stringr)


### Convert text to unique characters.
to_unique_chars <- function(textset) {
  t <- sapply(textset, function(text) { 
    unique(strsplit(text,"", useBytes=TRUE)[[1]])
  })
  unique(sort(do.call("c",t)))
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
  
  if (n1 < p) {
    # we create a vector of size p and pad it with 0s
    s1 <- rep(0, p)
    # we want the sequence at the end of the series not the start.
    start <- p-length(sequence)
    end <- p
    s1[start:end] <- sequence
    m <- matrix(s1, nrow=1, ncol=p, byrow=TRUE)
    return(as.matrix(m))
  }
  
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
    m <- matrix(0, nrow=n2, ncol=p, byrow=TRUE)
    m[1:length(nextSeq)] <- nextSeq
  } else {
    m <- matrix(nextSeq, nrow=n2, ncol=p, byrow=TRUE)
  }
  as.matrix(m)
}


window_char_sequence <- function(index_seq, window=60) {
  slidingMatrix(index_seq, window)
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
  
  
  rows <- nrow(windows) - 1
  
  ## predicting each end character after each row of windows 
  ## we have the y characters being the last column of windows
  next_chars <- windows[2:nrow(windows),ncol(windows)]
  ## Set the dimensions of the window back to original size.
  windows <- windows[1:rows,1:window]
  
  x <- array(0L, dim=c(rows, ncol(windows), cols))
  y <- array(0L, dim=c(rows, cols))
  
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

## Convert to text index sequences
convert_to_char_index_sequences <- function(text = list(), max_width=1024) {
  # we will truncate characters to an input dimension of s (as in the Character-level Convolutional Networks for Text classification).
  s <- max_width
  
  char_indices <- char_index_set()
  
  # we need to padd the text but require the maximum length of the text
  sizes <- sapply(text, nchar)
  names(sizes) <- 1:length(sizes)
  
  
  # text windows become padded sequences of 
  # maxwidth x windowSize
  
  text_indices <- sapply(text, function(t) {
    if (nchar(t) > s) {
      t <- str_trunc(t, s)
    }
    # in the case the text is shorter than s it is left padded.
    pad_t <- str_pad(t, s, side="left", pad=" ")
    # importantly based on the article "Text Understanding from Scratch"
    # the text sequence is quantized in backward order. 
    # The latest reading on characters is placed near the beginning of the output.
    pad_t <- pad_t %>% rev()
    sequence <- to_char_index_sequence(char_indices, pad_t)
    sequence
  })
  
  text_df <- t(as.data.frame(text_indices))
  list(
    num_indices=length(char_indices),
    text=text,
    sequence_list=text_indices,
    sequence_df=text_df
  )
}