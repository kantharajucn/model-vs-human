import requests
import shutil
import os
from os.path import join
from tqdm import tqdm


dataset_base_url = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{NAME}.tar.gz"

cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = join(cur_dir, "datasets")


def try_download_dataset_from_github(dataset_name):
    download_url = dataset_base_url.format(NAME=dataset_name)
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        total_length = response.headers.get('content-length')
        dataset_file = join(dataset_dir, f'{dataset_name}.tar.gz')
        print(f"Downloading dataset {dataset_name} to {dataset_file}")
        with open(dataset_file, 'wb') as fd:
            if total_length is None:  # no content length header
                fd.write(response.content)
            else:
                for chunk in tqdm(response.iter_content(chunk_size=4096)):
                    fd.write(chunk)
        shutil.unpack_archive(dataset_file, dataset_dir)
        os.remove(dataset_file)
        return True
    else:
        return False


def download_datasets(names=None):
    all_datasets = ["color", "contrast", "cue-conflict", "edge", "eidolonI", "eidolonII", "eidolonIII",
                    "false-color", "high-pass", "low-pass", "phase-scrambling", "power-equalisation",
                    "rotation", "silhoutte", "sketch", "stylized", "uniform-noise"]
    if names is None:
        datasets = all_datasets
    else:
        datasets = names

    if not os.path.exists(dataset_dir):
        os.makedirs("datasets")

    for dataset in datasets:
        if os.path.exists(join(dataset_dir, dataset)):
            print(f"Dataset is already downloaded, skip downloading the dataset {dataset}")
            continue
        try_download_dataset_from_github(dataset)


if __name__ == "__main__":
    download_datasets()