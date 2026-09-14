from datasets import load_dataset
import re

ds = load_dataset("mohanty/PlantVillage", "default")

all_data = list(ds["train"]) + list(ds["test"])

ids = set()

for item in all_data:
    path = item["text"]

    # Only count original color photographs
    if "raw/color/" in path:
        filename = path.split("/")[-1]

        # Extract the UUID before ___
        match = re.search(
            r"([a-f0-9-]{36})___",
            filename
        )

        if match:
            ids.add(match.group(1))

print("Total PlantVillage records:", len(all_data))
print("Unique original color images:", len(ids))