import os
from os.path import join

import pandas as pd
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

root_dir = os.environ.get("MODEL_VS_HUMAN_DIR")


def get_class_to_index(class_name):
    class_to_index = {
        "airplane": 0,
        "bear": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "clock": 9,
        "dog": 10,
        "elephant": 11,
        "keyboard": 12,
        "knife": 13,
        "oven": 14,
        "truck": 15
    }

    return class_to_index[class_name]


def create_dataset(pytorch_dataset):
    dataset_list = []
    for image_path, target in pytorch_dataset:
        dataset_list.append((image_path, target))
    df = pd.DataFrame(dataset_list, columns=["path", "target"])
    x_train, x_test, y_train, y_test = train_test_split(df,
                                                        df.target,
                                                        test_size=.50,
                                                        random_state=42,
                                                        stratify=df.target,
                                                        shuffle=True)

    if not os.path.exists(f"{root_dir}/input"):
        os.mkdir(f"{root_dir}/input")

    x_train.to_csv(os.path.join(root_dir, "input", "train.csv"), index=False)
    x_test.to_csv(os.path.join(root_dir, "input", "test.csv"), index=False)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, *args, **kwargs):
        super(ImageFolderWithPaths, self).__init__(*args, **kwargs)

    def _get_target(self, path):
        if "session-1" in path:
            img_name = path.split("/")[-1]
            category = img_name.split("_")[4]
            return get_class_to_index(category)
        else:
            return get_class_to_index(path.split("/")[-2])

    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        target = self._get_target(path)

        return path, target


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = join(cur_dir, "datasets")
    dataset = ImageFolderWithPaths(dataset_dir)

    create_dataset(dataset)