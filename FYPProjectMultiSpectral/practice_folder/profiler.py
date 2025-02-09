import cProfile, pstats, io
import pandas as pd
from dataloader import BigEarthNetDataLoader
from config.config import DatasetConfig, ModelConfig
import os

def main():
    bands = DatasetConfig.all_bands
    dataset_dir = DatasetConfig.dataset_paths["0.5"]
    metadata_path = DatasetConfig.metadata_paths["0.5"]

    # Read CSV into a DataFrame
    metadata_df = pd.read_csv(metadata_path)

    # Now pass the DataFrame instead of a string path
    data_module = BigEarthNetDataLoader(
        bands=bands,
        dataset_dir=dataset_dir,
        metadata_csv=metadata_df
    )

    data_module.setup(stage='test')

    # Now profile the data loading
    profiler = cProfile.Profile()
    profiler.enable()

    for i, batch in enumerate(data_module.train_dataloader()):
        if i == 10:
            break

    profiler.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

if __name__ == '__main__':
    main()