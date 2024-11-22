import requests
import zstandard as zstd
from tqdm import tqdm

def downloadAndExtractDataset(dataset_dir):
    download_url = 'https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1'
    chunk_size = 1024
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status() 
        total = int(response.headers.get('content-length', 0))
        block_size = chunk_size

        with open(dataset_dir, 'wb') as file, tqdm(total=total, unit='iB', unit_scale=True, desc='Downloading BigEarthNet', unit_divisor=1024) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print('The dataset has been downloaded successfully')
    except requests.exceptions.RequestException as e:
        print(f"Error during dataset download: {e}")
        raise
    
    try:
        with open(dataset_dir, 'rb') as file_in:
            dctx = zstd.ZstdDecompressor()
            with open(dataset_dir[:-4] + '.tar', 'wb') as file_out:
                dctx.copy_stream(file_in, file_out)
        print(f"Decompressed dataset to: {dataset_dir[:-4] + '.tar'}")
    except Exception as e:
        print(f"Error during dataset decompression: {e}")
        raise


if __name__ == "__main__":
    dataset_dir = 'BigEarthNet-S2.tar.zst'
    downloadAndExtractDataset(dataset_dir)