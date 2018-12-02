## Installing using conda.
## install_keras(method="conda", conda="/usr/local/miniconda3/bin/conda", tensorflow="default")
## After this cd to /usr/local/miniconda3/env
## source activate r-tensorflow
## then install into the environment conda install tensorflow-mkl

## Initialising Keras using conda library.

## Initialising Keras using conda library.
#use_python("/usr/local/miniconda3/envs/r-tensorflow/python")
#use_condaenv("r-tensorflow", conda="/usr/local/miniconda3/bin/conda")



### Initialise based on the environment.
init <- function(parentDir) {
 hostname <- Sys.info()["nodename"]
 configPath <- file.path(parentDir, "env", hostname, "config.R")
 defaultPath <- file.path(parentDir, "env", "default", "config.R")
 if (file.exists(configPath)) {
   source(configPath)
 } else {
   source(defaultPath)
 }
 config()
}