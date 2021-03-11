source("lib/read_agnews_data.R")
source("lib/char_text_processing.R")


file <- get_path("train")

test_data <- read_ag_file(file)


newsDataset <- create_ag_dataset(test_data)


text <- newsDataset$data_set$text

# use method to extract character sequences of max characters 1024
# this is the same as word indices but does not require a word vocabulary only a character level vocabulary.
# similarly the max width of the character feature space can be increased if required depending on resources.
# smaller widths are suitable for short utterances which is task specific such as reviews, or chat classification, or sentiment analysis.
max_char_width <- 1024

char_data <- NULL
rds_path <- file.path("data", "crepe", "ag_char_data.rds")
if (file.exists(rds_path)) {
  print("read from RDS")
  char_data <- readRDS(rds_path)
} else {
  char_data <- convert_to_char_index_sequences(text, max_width=max_char_width)
  saveRDS(char_data, rds_path)
}

