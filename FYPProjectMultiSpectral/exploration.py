import pandas as pd
from pathlib import Path

def compare_patch_ids(csv_file1_path, csv_file2_path, column_name='patch_id'):
    """
    Compare patch_id values between two CSV files and report matches and differences.
    
    Parameters:
    - csv_file1_path (str or Path): Path to the first CSV file.
    - csv_file2_path (str or Path): Path to the second CSV file.
    - column_name (str): Name of the column containing patch IDs (default: 'patch_id').
    """
    # Convert paths to Path objects
    csv_file1_path = Path(csv_file1_path)
    csv_file2_path = Path(csv_file2_path)

    # Check if files exist
    if not csv_file1_path.exists():
        print(f"Error: {csv_file1_path} does not exist.")
        return
    if not csv_file2_path.exists():
        print(f"Error: {csv_file2_path} does not exist.")
        return

    # Load the CSV files
    try:
        df1 = pd.read_csv(csv_file1_path)
        df2 = pd.read_csv(csv_file2_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Check if the column exists in both files
    if column_name not in df1.columns:
        print(f"Error: Column '{column_name}' not found in {csv_file1_path}.")
        return
    if column_name not in df2.columns:
        print(f"Error: Column '{column_name}' not found in {csv_file2_path}.")
        return

    # Extract patch_ids as sets for efficient comparison
    patch_ids1 = set(df1[column_name].dropna().unique())
    patch_ids2 = set(df2[column_name].dropna().unique())

    # Find matches and differences
    common_ids = patch_ids1.intersection(patch_ids2)  # IDs present in both
    only_in_file1 = patch_ids1 - patch_ids2  # IDs only in file 1
    only_in_file2 = patch_ids2 - patch_ids1  # IDs only in file 2

    # Report results
    print("\nComparison Results:")
    print(f"Total unique patch_ids in {csv_file1_path.name}: {len(patch_ids1)}")
    print(f"Total unique patch_ids in {csv_file2_path.name}: {len(patch_ids2)}")
    print(f"Number of matching patch_ids: {len(common_ids)}")

    if common_ids:
        print("\nSample of matching patch_ids (up to 5):")
        for pid in list(common_ids)[:5]:
            print(f"  {pid}")
    else:
        print("\nNo matching patch_ids found.")

    if only_in_file1:
        print(f"\nPatch_ids only in {csv_file1_path.name} ({len(only_in_file1)}):")
        for pid in list(only_in_file1)[:5]:
            print(f"  {pid}")
        if len(only_in_file1) > 5:
            print(f"  ... ({len(only_in_file1) - 5} more)")
    else:
        print(f"\nNo patch_ids unique to {csv_file1_path.name}.")

    if only_in_file2:
        print(f"\nPatch_ids only in {csv_file2_path.name} ({len(only_in_file2)}):")
        for pid in list(only_in_file2)[:5]:
            print(f"  {pid}")
        if len(only_in_file2) > 5:
            print(f"  ... ({len(only_in_file2) - 5} more)")
    else:
        print(f"\nNo patch_ids unique to {csv_file2_path.name}.")

    # Summary
    if len(common_ids) == len(patch_ids1) == len(patch_ids2):
        print("\nAll patch_ids match perfectly between the two files!")
    else:
        print("\nMismatch detected. Check the unique patch_ids listed above.")

# Example usage
if __name__ == "__main__":
    csv_file1_path = r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet\metadata_10_percent_old.csv"
    csv_file2_path = r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet\metadata_10_percent.csv"

    compare_patch_ids(csv_file1_path, csv_file2_path, column_name='patch_id')