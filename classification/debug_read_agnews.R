source("lib/read_agnews_data.R")


file <- get_path("test")

test_data <- read_ag_file(file)


dataset <- create_ag_dataset(test_data)