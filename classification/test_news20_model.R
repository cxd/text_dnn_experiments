## Save the vocabulary from the full training set for use with models trained on the full data.

source("lib/init.R")
library(keras)
library(stringr)
library(purrr)
source("lib/read_news20.R")
source("lib/memnet_singleinput_classifier.R")
source("lib/read_glove.R")
source("lib/read_classification_text.R")
# Setup environment
cfg <- init(getwd())


path <- "data/news20/full_vocab.rds"
## test reading it
vocab <- readRDS(path)

newsDataset <- readRDS("data/news20/newsDataset_full.rds")

devSet <- FALSE

testNewsData <- getPath(type="test", dev=devSet) %>% 
  read_news_file()

testNewsDataset <- list()
if (file.exists("data/news20/test_news_dataset_full.rds")) {
  testNewsDataset <- readRDS("data/news20/test_news_dataset_full.rds")
} else {
  testNewsDataset <- news_create_data_set(testNewsData)
  saveRDS(testNewsDataset, "data/news20/test_news_dataset_full.rds")
}

testIndexedData <- list()
if (file.exists("data/news20/test_indexed_data_full.rds")) {
  testIndexedData <- readRDS("data/news20/test_indexed_data_full.rds")
} else {
  # use the same vocabulary as the training vocab
  testIndexedData <- vectorise_word_indices(testNewsDataset$data_set, 
                                            vocab$vocab, 
                                            vocab$maxlen,
                                            unknownWord="unknown")
  saveRDS(testIndexedData, "data/news20/test_indexed_data_full.rds")
  
}


test1_x <- testIndexedData$word_indices
test1_x <- as.matrix(test1_x)

test1_y <- testIndexedData$class_encoded
test1_y <- do.call("rbind", test1_y)

dropout=0.6
model1 <- NULL
if (is.null(cfg$hasGpu) || cfg$hasGpu== FALSE) {
  model1 <- define_memnet_single(vocab$maxlen, 
                                 vocab$vocab_size, 
                                 length(newsDataset$class_labels), 
                                 embed_dim=50, 
                                 dropout=dropout)
  
} else {
  model1 <- define_memnet_single_gpu(vocab$maxlen, 
                                 vocab$vocab_size, 
                                 length(newsDataset$class_labels), 
                                 embed_dim=50, 
                                 dropout=dropout)
  
}
## Load the stored weights, these are smaller in file size than the full model file.
load_model_weights_hdf5(model1, "saved_models/test_news_memnet_single_weights.h5")
summary(model1)

## Note the evaluation on the current version of the model is around 77% accuracy.
## The model is still training hence this can be improved.

## Evaluation using the model trained to validation accuracy of 86% gives roughly 77% accuracy on test set.
## Benchmark models are able to give 84% accuracy so we are aiming at bettering the benchmark.
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


input_text <- "Philosophical questions are usually foundational and abstract in nature. Philosophy is done primarily through reflection and does not tend to rely on experiment, although the methods used to study it may be analogous to those used in the study of the natural sciences."

test_sample <- convert_text_to_dataset(input_text)

test_data <- vectorise_word_indices(test_sample, 
                                    vocab$vocab, 
                                    vocab$maxlen,
                                    unknownWord="unknown")

predict_model (model1, test_data$word_indices, testNewsDataset$class_labels)


## use the model for visualising dimensions.
vis_model <- keras_model(input=model1$input,
                         output=get_layer(model1, name="dense_3")$output)


summary(vis_model)

## we can try and visualise the output

input_text1 <- "NVidia GPU install seems to be stuck as it progresses no further than 'Installing drivers'"

test_sample1 <- convert_text_to_dataset(input_text1)

test_data1 <- vectorise_word_indices(test_sample1, 
                                    vocab$vocab, 
                                    vocab$maxlen,
                                    unknownWord="unknown")


input_text2 <- "Philosophical questions are usually foundational and abstract in nature. Philosophy is done primarily through reflection and does not tend to rely on experiment, although the methods used to study it may be analogous to those used in the study of the natural sciences."

test_sample2 <- convert_text_to_dataset(input_text2)

test_data2 <- vectorise_word_indices(test_sample2, 
                                    vocab$vocab, 
                                    vocab$maxlen,
                                    unknownWord="unknown")

v1 <- vis_model %>% predict(test_data1$word_indices)

v2 <- vis_model %>% predict(test_data2$word_indices)

plot(t(v1), col="red", type="n")
points(t(v2), col="blue", type="n")
text(t(v1), col="red", labels=test_sample1$word_vector[[1]])
text(t(v2), col="blue", labels=test_sample2$word_vector[[1]])
