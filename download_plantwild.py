from datasets import load_dataset

print("Downloading PlantWild...")

ds = load_dataset("Voxel51/PlantWild")

print("\nDownload complete!")
print(ds)

for split in ds:
    print(split, ":", len(ds[split]))
    