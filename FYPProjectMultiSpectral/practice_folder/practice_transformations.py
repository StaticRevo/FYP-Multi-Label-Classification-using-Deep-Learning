import rasterio

file_path1 = r"C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\web\static\images\multispectral_patch_1743404758.tif"
file_path2 = r"C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\web\static\images\multispectral_patch_1743404926.tif"
file_path3 = r"C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages\S2A_MSIL2A_20170704T112111_N9999_R037_T29SND_53_66.tif"

print("Reading normal multispectral image from sentinel hub")
with rasterio.open(file_path1) as dataset:
    print(f"Width: {dataset.width}, Height: {dataset.height}")
    print(f"Number of Bands: {dataset.count}")
    print(f"Data Type: {dataset.dtypes}")
    print(f"Coordinate Reference System: {dataset.crs}")
    print(f"Transform: {dataset.transform}")
    
    # Read first band stats
    band1 = dataset.read(1)
    print(f"Band 1 Min: {band1.min()}, Max: {band1.max()}")

print()
print("Reading modified multispectral image from sentinel hub")
with rasterio.open(file_path2) as dataset:
    print(f"Width: {dataset.width}, Height: {dataset.height}")
    print(f"Number of Bands: {dataset.count}")
    print(f"Data Type: {dataset.dtypes}")
    print(f"Coordinate Reference System: {dataset.crs}")
    print(f"Transform: {dataset.transform}")
    
    # Read first band stats
    band1 = dataset.read(1)
    print(f"Band 1 Min: {band1.min()}, Max: {band1.max()}")

print()
print("Reading BigEarthNet patch")
with rasterio.open(file_path3) as dataset:
    print(f"Width: {dataset.width}, Height: {dataset.height}")
    print(f"Number of Bands: {dataset.count}")
    print(f"Data Type: {dataset.dtypes}")
    print(f"Coordinate Reference System: {dataset.crs}")
    print(f"Transform: {dataset.transform}")
    
    # Read first band stats
    band1 = dataset.read(1)
    print(f"Band 1 Min: {band1.min()}, Max: {band1.max()}")