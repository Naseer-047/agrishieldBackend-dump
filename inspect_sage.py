import requests

url = "https://datasets-server.huggingface.co/filter"

params = {
    "dataset": "tirtho149/SAGE",
    "config": "default",
    "split": "train",
    "where": '"crop"=\'Tomato\'',
    "offset": 0,
    "length": 10
}

r = requests.get(url, params=params, timeout=60)

print("Status:", r.status_code)

data = r.json()

print("Partial:", data.get("partial"))
print("Rows available:", data.get("num_rows_total"))

print("\nReturned samples:")

for row in data.get("rows", []):
    item = row["row"]
    print(item["crop"], "|", item["disease"])