import os
import pandas as pd
from typing import NamedTuple

class ReadFileOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_load_and_convert_to_csv(path: str) -> ReadFileOutputs:
    """Loads a dataset from various formats and ensures it is converted to CSV.

    The detects the file format based on the extension. If the file is 
    not in CSV format (e.g., Excel, JSON, Parquet), it converts the data into 
    a CSV file in the same directory and then loads it into a pandas DataFrame. 
    This ensures consistency for the subsequent steps of the pipeline.

    Args:
        path (str): The relative or absolute path to the input file. 
            Supported formats: .csv, .xlsx, .xls, .json, .parquet.

    Returns:
        ReadFileOutputs: The DataFrame loaded from the input file wrapped in a NamedTuple.
    """
    
    # Split the file path into the base name and the extension
    base_name, extension = os.path.splitext(path)
    extension = extension.lower()

    # Initialize df to None to prevent UnboundLocalError on unsupported formats
    df = None 

    # If the file is already a CSV, load it directly
    if extension == '.csv':
        df = pd.read_csv(path)
        
    else:
        # Map extensions to their corresponding Pandas reader functions
        readers = {
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.json': pd.read_json,
            '.parquet': pd.read_parquet
        }

        if extension in readers:
            print(f"Converting file into CSV")
            
            # Read the original file using the appropriate reader
            temp_df = readers[extension](path)
            
            # Define the new file path with the .csv extension
            new_csv_path = base_name + ".csv"
            
            # Save the DataFrame to a CSV file (index=False avoids creating an extra index column)
            temp_df.to_csv(new_csv_path, index=False)
            
            # Try to assign the resulting data to df_output else error
            try:
                df = pd.read_csv(new_csv_path)
            except ValueError:
                print(f"Unsupported file format: {extension}")
        else:
            print(f"Unsupported file format: {extension}")

    return ReadFileOutputs(df_output=df)