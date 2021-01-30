require(tensorflow)


init_tf <- function() {
  physical_devices = tf$config$experimental$list_physical_devices('GPU')
  if (length(physical_devices) > 0)
    print('Found GPU Device 0')
  for (i in 1:length(physical_devices))
    tf$config$experimental$set_memory_growth(physical_devices[[i]], TRUE)
  
  # also on this machine favour using gpu 2 quadro p4000.
  if (length(physical_devices) > 1)
    tf$config$experimental$set_visible_devices(physical_devices[[2]], 'GPU')
  
}

# Default environment configuration
## The default is empty.
config <- function() {
  init_tf()
  list(
    hasGpu=TRUE
    )
}