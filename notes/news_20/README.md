## Convolutional network architecture for Short Utterance Classification.

This project is derived from work done external to Nuance, investigating the use 
of a convolutional neural network for text sequence classification.

The intent in using the CNN is to apply a reasonably quick training and inference architecture on resource constrained hardware while learning a representation of an input sequence that maps the correspondance of words into a lower dimensional feature space.
For example, larger networks using recurrent architectures such as LSTM and RNN have proven highly effective in learning relationships in sequences, however require very long training cycles and are also slow on inference without supporting hardware. 
The current state of the art requires multiple NVidia GPUs to train and perform inference.

Being resource constrained, the aim is to seek a deep network architecture that can be sufficient to train on hardware with limited resources, and be suitable for real time inference. While of course not precluding the use of GPU where available.

## Investigating Network Architectures

The types of network architectures explored to date focus on the following layers.

- An Embedding Layer
- A sequence learning layer
- A Classification Layer - commonly a dense layer to softmax activation.

The embedding layer and classification layers have remained the same structure in each of the architectures investigated, although it will also be useful to investigate the use of prelearnt embeddings (although some investigation has been applied in this area).

The main component of the architecture that has been the focus of change so far is the "sequence learning layer". This term is used loosely as the approach to date has evaluated

- use of LSTM for sequence learning 
- use of GRU for sequence learning
- use of single CNN for feature mapping from sequence input to convolved output.
- use of stacked CNN for multiple convolutions applied from sequence to feature maps.

The LSTM and GRU have proven effective over time, but have limitations in that their training processes are highly inefficient on CPU, and also result in very large models. However it is possible to train on GPU and transfer weights into a CPU only model, post training, while this deserves some investigation is not practical unless GPUs are available in the production environment, especially where training and deployment is sought to occur in production.

The single CNN architecture has proven to be suitable for faster training cycles and shorter epochs on large vocabularies.

The stacked CNN for multiple convolutions also demonstrates some level of suitability for a similarly resource constrained architecture.

The latter two models could feasibly train on CPU where the appropriate libraries are available (possibly the use of intel MKL for accelerated learning), and have shortened training cycles on GPU. 
The initial benchmark data set is the "newsgroup 20 class" data set, which has a large vocabulary and is well balanced in regards to the 20 classes. The original dataset is available at stanford https://nlp.stanford.edu/wiki/Software/Classifier/20_Newsgroups. 
A corresponding commentary on the same dataset is also found here: https://acardocacho.github.io/capstone/. 
Both pages give a set of benchmarks for different types of classifiers, with a benchmark of 84.86% accuracy on the test set given by a linear classifier on the standford page, and a benchmark of 84.7% accuracy given by an SVM classifier on the page by A.Cardocacho. 
The interesting feature of the newsgroup dataset is the reasonably large vocabulary at about 70000 words. The balanced nature of the class assignments also makes the dataset well behaved as a baseline for training and testing. The presence of other benchmarks and its popularity makes it a good tool for baselining classifiers against other work.

The following set of timings give an indication as to the processing speed requirements for different model architectures, these are derived from the news 20 full training set of some 7905 training examples in a single epoch, with a word based vocabulary feeding into a embedding layer as the first layer.

- CPU - 1 Layer LSTM and Feedforward network - Intel(R) Xeon(R) CPU X5650 @ 2.67GHz 6 Cores - 4300s per epoch - 550ms per step - libopenblas 2.20 ubuntu
- CPU - 1 Layer CNN and Feedforward network - Intel i7 2.9Ghz 2 Cores - 50s per epoch - ? per step - Blas Accelerate macOs Framework
- CPU - 1 Layer CNN and Feedforward network - Intel(R) Xeon(R) CPU X5650 @ 2.67GHz 6 Cores - 320s per epoch - 40ms per step - libopenblas 2.20 ubuntu
- GPU - 1 Layer CNN and Feedforward network - NVidia Quadro P4000 - 20s per epoch - ? per step - CUDA 7.0 Windows 10 Pro
- CPU - 3 Layer CNN and Feedforward network - Intel i7 2.9Ghz 2 Cores - 120s per epoch - ? per step - Blas Accelerate macOs Framework
- GPU - 3 Layer CNN and Feedforward network - NVidia Quadro P4000 - 30s per epoch - ? per step - CUDA 7.0 Windows 10 Pro


Note that the timing will be variable on CPU as the libraries for matrix operations will influence the performance a great deal, the use of Intel MKL on cpu should be investigated. 


Should the use of deep network architectures become a much more desirable approach in production, it would be appropriate to deploy GPU based hardware for this purpose both in local lab environments and onsite as inference will be a high performance cost.

The approach where a CNN is applied may be suitable for small utterance inference, however this has not yet been proven, only appropriate load testing will be able to determine this.



