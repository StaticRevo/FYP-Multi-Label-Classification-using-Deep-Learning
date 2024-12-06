import pandas as pd
import ast
import sys



metadata_path: str =  r'C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\metadata_1_percent.csv'
metadata_csv = pd.read_csv(metadata_path)

import pandas as pd
import ast
# Function to clean and parse labels
def clean_and_parse_labels(label_string):
    cleaned_labels = label_string.replace(" '", ", '").replace("[", "[").replace("]", "]")
    return ast.literal_eval(cleaned_labels)

metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

label_counts = metadata_csv['labels'].explode().value_counts()
label_df = label_counts.reset_index()
label_df.columns = ['Label', 'Number of Images']

# Display the table
print(label_df)