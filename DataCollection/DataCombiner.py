import os
import pandas as pd
from datetime import datetime

class CSVParser:
    def __init__(self, folder_path, min_word_count = 10):
        self.folder_path = folder_path
        self.min_word_count= min_word_count
        self.df = pd.DataFrame()

    def parse_csv_files(self):
        # Iterate over each file in the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.folder_path, filename)
                
                # Read the CSV file
                data = pd.read_csv(file_path)
                
                # Check if the necessary columns are present in the CSV file
                if all(col in data.columns for col in ['Movie Name', 'Comment', 'Imdb Rating']):
                    # Append data to the existing dataframe
                    self.df = pd.concat([self.df, data[['Movie Name', 'Comment', 'Imdb Rating']]], ignore_index=True)
                else:
                    print(f"Skipping file {filename}: Required columns not found.")

    def save_combined_data(self):
        if not self.df.empty:
            # Create a timestamped folder name
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_folder = os.path.join(self.folder_path, f"CombinedDataset-{timestamp}")
            
            # Create the new folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Define the output file path
            output_file = os.path.join(output_folder, 'combined_comments.csv')

            #Filter comments less than 10 words
            filtered_df = self.df[self.df['Comment'].apply(lambda x: len(x.split()) > self.min_word_count)]

            
            # Save the combined dataframe to CSV
            filtered_df.to_csv(output_file, index=False)
            print(f"Combined data saved to {output_file}")
        else:
            print("No valid data to save.")


