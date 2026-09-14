from huggingface_hub import hf_hub_download
import os

os.makedirs("model", exist_ok=True)

hf_hub_download(
    repo_id="Daksh159/plant-disease-mobilenetv2",
    filename="mobilenetv2_plant.pth",
    local_dir="model"
)

hf_hub_download(
    repo_id="Daksh159/plant-disease-mobilenetv2",
    filename="class_names.json",
    local_dir="model"
)

print("Download finished successfully!")