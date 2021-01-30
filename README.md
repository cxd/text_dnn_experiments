## Exploring Deep Learning Methods for Language Processing and Classification

This project explores the methodology for a variety of deep learning methods for language processing.

It using a subset of the Stanford Question Answering data set (https://rajpurkar.github.io/SQuAD-explorer/).

As well as a subset of the news 20 data set for classification [news 20 notes](classification/news_20_references.md).

The pretrained vector space model is provided by GLOVE. https://nlp.stanford.edu/projects/glove/

The project seeks to recreate a number of different models in the area of text processing, and at this point is in the survey stage of reviewing a number of different papers, tutorials and examples. Links to materials are provided in comments of scripts an in ad-hoc notes in the repo.


Note also for exploring Glove in R see also: https://rpubs.com/ww44ss/glove300d
Additional examples of loading word vectors into a keras layer is shown in:
https://github.com/rstudio/keras/blob/master/vignettes/examples/pretrained_word_embeddings.R

The glove vectors can be downloaded via the download.sh script.

A good discussion on Attention networks in Keras is here:

https://github.com/keras-team/keras/issues/4962

A good discussion on implementation of attention mechanism in R is here: https://blogs.rstudio.com/tensorflow/posts/2018-07-30-attention-layer/

The Babi question answering example is described in the vignette on the keras website here:
https://keras.rstudio.com/articles/examples/babi_memnn.html

## Note on using cudnn lstm weights in cpu lstm weights.
It should be possible to use gpu trained lstm weights in a cpu lstm model.
Comments on keras github have details around this.
https://github.com/keras-team/keras/issues/9463
