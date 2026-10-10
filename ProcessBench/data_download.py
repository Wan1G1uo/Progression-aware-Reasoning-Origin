from datasets import load_dataset
import pandas as pd
import ast

def save_split_to_csv(dataset_name, split_name, csv_filename):
    """
    Loads a specific split of a dataset and saves it to a CSV file.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub (e.g., "Qwen/ProcessBench").
        split_name (str): The name of the split to load (e.g., "gsm8k", "math").
        csv_filename (str): The name of the CSV file to save the data to.
    """
    print(f"Loading {split_name} split from {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split_name)
    
    print("Converting to pandas DataFrame...")
    df = pd.DataFrame(dataset)
    
    df.to_csv(csv_filename, index=False)
    print(f"Saved {split_name} split to {csv_filename}")

def create_model_response_column(csv_file_path):
    """
    Reads a CSV file, creates a new 'model_response' column based on the 'steps' column,
    ensuring the result is a pure string. The 'steps' column is expected to contain
    string representations of lists of strings.

    Args:
        csv_file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame with the new 'model_response' column.
    """
    print(f"Reading CSV file: {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    def parse_steps_list(steps_str):
        """
        Safely parses a string representation of a list.
        """
        try:
            if isinstance(steps_str, str) and steps_str.strip().startswith('[') and steps_str.strip().endswith(']'):
                return ast.literal_eval(steps_str)
            else:
                return steps_str
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse steps string: {steps_str}. Returning as is.")
            return steps_str 

    print("Parsing 'steps' column...")
    df['steps'] = df['steps'].apply(parse_steps_list)

    print("Creating 'model_response' column...")
    df['model_response'] = df['steps'].apply(
        lambda x: "\n\n".join(str(item) for item in x) if isinstance(x, list) else (x if isinstance(x, str) else str(x))
    )

    return df

if __name__ == "__main__":
    dataset_name = "Qwen/ProcessBench"
    split_name_to_process = "gsm8k"
    csv_file_name = f"{split_name_to_process}.csv"

    save_split_to_csv(dataset_name, split_name_to_process, csv_file_name)
    
    print(f"\n--- Finished downloading and saving {csv_file_name} ---\n")

    df_with_response = create_model_response_column(csv_file_name)

    print("\nPreview of 'steps' and 'model_response' columns:")
    print(df_with_response[['steps', 'model_response']].head())

    print("\nType of 'model_response' column elements:")
    print(df_with_response['model_response'].apply(type).value_counts())

    print(f"\nSaving processed data back to {csv_file_name}...")
    df_with_response.to_csv(csv_file_name, index=False)
    print(f"Successfully saved processed data to {csv_file_name}")