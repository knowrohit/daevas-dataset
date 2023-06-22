import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Define your directory path
dir_path = '/Users/rohittiwari/Downloads/daevas_dataset/12thCBSE/'

# Walk through the directory
for subdir, dirs, files in os.walk(dir_path):
    for filename in files:
        filepath = subdir + os.sep + filename

        # Check if the file is a csv
        if filepath.endswith(".csv"):
            # Load data
            df = pd.read_csv(filepath)

            # Create embeddings for each chunk of text
            embeddings = model.encode(df['Text'].tolist())

            # Add embeddings to the dataframe
            df['embeddings'] = embeddings.tolist()

            # Define parquet file path
            parquet_file_path = subdir + os.sep + filename.split(".")[0] + ".parquet"

            # Save the dataframe as a parquet file
            df.to_parquet(parquet_file_path)
