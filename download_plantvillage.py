from datasets import load_dataset

print("Downloading PlantVillage...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)


print("\nDownload complete!")
print(dataset)

print("\nTrain images:", len(dataset["train"]))
print("Test images:", len(dataset["test"]))
