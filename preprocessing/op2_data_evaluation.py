from typing import NamedTuple
import pandas as pd

class EvaluationOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_evaluation(df: pd.DataFrame) -> EvaluationOutputs:
    """Performs a comprehensive exploratory data analysis on a DataFrame.

    This function prints a summary of the dataset, including information about 
    data types, missing values, statistical descriptions, and unique values 
    for categorical columns. It helps identify data quality issues before 
    proceeding to the cleaning phase.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        EvaluationOutputs: The original DataFrame wrapped in a NamedTuple.
    """
    print("DataFrame Information Summary:")
    # Provides a summary of the dataset information
    df.info()
    print("\n" + "="*50 + "\n")

    print("Missing Values Analysis:")
    # Counts how many records are null in each column
    null_values_per_column = df.isnull().sum()
    # Identifies columns with null values
    columns_with_nulls = null_values_per_column[null_values_per_column > 0]
    
    if columns_with_nulls.empty:
        # If the series is empty, it means no nulls were found
        print("No missing values were found in the dataset.")
    else:
        # If not empty, print the count for each column affected
        print("Columns with missing values:")
        print(columns_with_nulls)

    print("\nStatistical Description of the DataFrame:")
    # Prints statistical data for both numerical and categorical columns
    print(df.describe(include='all'))
    print("\n" + "="*50 + "\n")

    # Print a summary of the dataset's structure
    print("Summary:")
    shape = df.shape
    data_types = df.dtypes.unique()
    print(f"The DataFrame contains {shape[1]} columns and {shape[0]} rows.")
    print(f"The possible data types are: {data_types}.")
    print(f"Total null values found: {columns_with_nulls.sum()}.")

    print("\n" + "="*50 + "\n")

    print("First 5 rows preview:")
    # Prints the first rows of the dataframe
    print(df.head())
    print("\n" + "="*50 + "\n")

    # List of columns with dtype 'O' (object/text)
    object_columns = df.select_dtypes(include=['object']).columns

    print("Object-type Columns Analysis:")
    if object_columns.empty:
        print("No object-type (string) columns found in the dataset.")
    else:
        print(f"Found {len(object_columns)} object-type columns: {list(object_columns)}")
        
        df_objects = df[object_columns]
        
        print("\nFirst rows of the Object-only DataFrame:")
        print(df_objects.head())

        # Inspect the actual unique values of these text columns
        print("\nUnique values within Object columns:")
        for col in object_columns:
            unique_values = df_objects[col].dropna().unique()
            print(f"\nColumn: **{col}**")
            # Stampiamo solo i primi 20 valori unici per non intasare lo schermo (utile per i nomi)
            if len(unique_values) > 20:
                print(f"Values (first 20 of {len(unique_values)}): {unique_values[:20]}...")
            else:
                print(f"Values: {unique_values}")

    print("\n" + "="*50 + "\n")

    return EvaluationOutputs(df_output=df)