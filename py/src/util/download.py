import os
import requests


def download(url: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return

    print(f'Downloading {url}')
    r = requests.get(url)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(r.content)
