
## Environment specific configuration.
config <- function() {
  library(reticulate)
  use_python("~/.conda/envs/r-tensorflow/bin/python3.6")
  use_virtualenv("r-tensorflow")
  py_config()
  list()
}