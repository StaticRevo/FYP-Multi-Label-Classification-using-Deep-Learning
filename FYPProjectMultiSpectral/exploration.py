import ast
import pandas as pd


# Function to clean and parse labels
def clean_and_parse_labels(label_input):
    if isinstance(label_input, list):  # If input is already a list, return it as-is
        return label_input
    elif isinstance(label_input, str): # If input is a string, clean and parse it
        cleaned_labels = label_input.replace(" '", ", '").replace("[", "[").replace("]", "]")
        return ast.literal_eval(cleaned_labels)
    else:
        raise TypeError(f"Expected label_input to be a string or list, got {type(label_input)}")

# Function to process metadata and get per-label counts per split
def get_label_counts_per_split(metadata_df, split_column='split'):
    metadata_df['labels'] = metadata_df['labels'].apply(clean_and_parse_labels)
    label_counts = metadata_df.explode('labels')
    
    split_label_counts = label_counts.groupby([split_column, 'labels']).size().reset_index(name='Number of Images')
    pivot_counts = split_label_counts.pivot(index=split_column, columns='labels', values='Number of Images').fillna(0)
    
    # Add total images per split
    pivot_counts['Total Images'] = pivot_counts.sum(axis=1)
    
    return pivot_counts

metadata_df_10percent = pd.read_csv(r"C:\Users\isaac\Desktop\metadata_10_percent.csv")

# Process the 10% subset
print("Subset: 10%")
subset_10percent_counts = get_label_counts_per_split(metadata_df_10percent)
print(subset_10percent_counts)
