## Save the vocabulary from the full training set for use with models trained on the full data.

source("lib/init.R")
library(keras)
library(stringr)
library(purrr)
source("lib/read_news20.R")
source("lib/memnet_singleinput_classifier.R")
source("lib/read_glove.R")

# Setup environment
cfg <- init(getwd())


path <- "data/news20/full_vocab.rds"
## test reading it
vocab <- readRDS(path)

testNewsData <- getPath(type="test", dev=FALSE) %>% 
  read_news_file()

testNewsDataset <- create_data_set(testNewsData)

# use the same vocabulary as the training vocab
testIndexedData <- vectorise_word_indices(testNewsDataset$data_set, 
                                          vocab$vocab, 
                                          vocab$maxlen,
                                          unknownWord="unknown")


test1_x <- testIndexedData$word_indices
test1_x <- as.matrix(test1_x)

test1_y <- testIndexedData$class_encoded
test1_y <- do.call("rbind", test1_y)

modelPath <- "saved_models/full_news_memnet_single.h5"
model1 <- load_model_hdf5(modelPath, compile=TRUE)

## Note the evaluation on the current version of the model is around 77% accuracy.
## The model is still training hence this can be improved.
evaluate_model(model1, test1_x, test1_y)

## Some example classifications for unseen text examples.
## Note that these examples themselves are entirely outside of the training or test corpus.
## They do not necessarily reflect the kind of text corpus the model was trained on.
## So in its predictions, it is basically assigning a class label to an input that is entirely novel.
## Does the assignment for these examples result in a categorisation that seems sensible?
## And is there a way to inspect the model to relate activations back to keywords in the utterances 
## so as to get some idea as to how to explain the mapping the model predicts?

input_text <- "NVidia GPU install seems to be stuck as it progresses no further than 'Installing drivers'"

test_sample <- convert_text_to_dataset(input_text)

test_data <- vectorise_word_indices(test_sample, 
                                            vocab$vocab, 
                                            vocab$maxlen,
                                            unknownWord="unknown")

predict_model (model1, test_data$word_indices, testNewsDataset$class_labels)


input_text <- "Already the world is experiencing the symptoms of a looming, global environmental crisis."

test_sample <- convert_text_to_dataset(input_text)

test_data <- vectorise_word_indices(test_sample, 
                                    vocab$vocab, 
                                    vocab$maxlen,
                                    unknownWord="unknown")

predict_model (model1, test_data$word_indices, testNewsDataset$class_labels)



input_text <- "Philosophical questions are usually foundational and abstract in nature. Philosophy is done primarily through reflection and does not tend to rely on experiment, although the methods used to study it may be analogous to those used in the study of the natural sciences."

test_sample <- convert_text_to_dataset(input_text)

test_data <- vectorise_word_indices(test_sample, 
                                    vocab$vocab, 
                                    vocab$maxlen,
                                    unknownWord="unknown")

predict_model (model1, test_data$word_indices, testNewsDataset$class_labels)

