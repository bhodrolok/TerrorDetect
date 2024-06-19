import snscrape.modules.twitter as sntwitter
import pandas as pd
import glob

csv_file_extension = 'csv'

# Use list comprehension to get list of filenames that have .csv file extension
all_csv_filenames = [i for i in glob.glob('*.{}'.format(csv_file_extension))]
# Combine these files into a single list
combined_csv_files = pd.concat([pd.read_csv(f) for f in all_csv_filenames])
# Export to a final CSV dataset
combined_csv_files.to_csv("dataset.csv", index=False)

# References:
# https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
