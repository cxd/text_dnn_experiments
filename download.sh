#!/bin/bash
DATA_DIR=data
# Download GloVe
GLOVE_DIR=$DATA_DIR/glove
mkdir $GLOVE_DIR

wget https://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR