
## Environment specific configuration.
config <- function() {
  list()
}

## Plaid does not appear to work with keras.
testingplaid <- function() {
  library(reticulate)
  use_python("/Users/cd/.conda/envs/plaidml/bin/python")
  use_condaenv("plaidml", conda="/usr/local/miniconda3/bin/conda")
  py_config()
  ## Testing with plaidml
  use_backend("plaidml")
  list()
}