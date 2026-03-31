# --------------------------------------------------------
# References:
# SatMAE: https://github.com/sustainlab-group/SatMAE
# --------------------------------------------------------

import os
import pandas as pd
import numpy as np
import warnings

from typing import Any, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

try:
    # import rasterio
    # from rasterio import logging
    from tifffile import imread

    # log = logging.getLogger()
    # log.setLevel(logging.ERROR)
except ModuleNotFoundError as err:
    # Error handling
    print(err)
# import rasterio
# from rasterio import logging

# log = logging.getLogger()
# log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
use_antialias_key = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR > 12)


CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    def get_targets(self):
        return np.array([i for i, _ in enumerate(CATEGORIES)])

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                (
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.2, 1.0), interpolation=interpol_mode, antialias=None
                    )
                    if use_antialias_key
                    else transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode)
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode, antialias=None
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform, target_transform=None, custom_targets=None, path_prefix=""):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.target_transform = target_transform

        self.path_prefix = path_prefix

        # custom targets to replace CATEGORIES
        self.custom_targets = custom_targets

    def get_targets(self):
        if self.custom_targets is None:
            return super().get_targets()
        return np.array([i for i, _ in enumerate(self.custom_targets)])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        single_image_name = self.path_prefix + single_image_name
        single_image_name = single_image_name.replace("//", "/")
        img_as_img = Image.open(single_image_name)
        # img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ["value", "one-hot"]
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(
        self,
        csv_path: str,
        transform: Any,
        target_transform=None,
        years: Optional[List[int]] = [*range(2000, 2021)],
        categories: Optional[List[str]] = None,
        label_type: str = "value",
        masked_bands: Optional[List[int]] = None,
        dropped_bands: Optional[List[int]] = None,
        dataset_mean: Optional[List[int]] = None,
        dataset_std: Optional[List[int]] = None,
    ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path).sort_values(["category", "location_id", "timestamp"])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df["year"] = [int(timestamp.split("-")[0]) for timestamp in self.df["timestamp"]]
            self.df = self.df[self.df["year"].isin(years)]

        # self.indices = self.df.index.unique().to_numpy()

        self.transform = transform
        self.target_transform = target_transform

        if label_type not in self.label_types:
            raise ValueError(
                f"FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:",
                ", ".join(self.label_types),
            )
        self.label_type = label_type

        self.masked_bands = np.array(masked_bands, dtype=np.int64) if masked_bands is not None else None
        self.dropped_bands = np.array(dropped_bands, dtype=np.int64) if dropped_bands is not None else None
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

        # Do this do avoid memory leak?
        self.mean = np.array(dataset_mean if dataset_mean else self.mean)
        self.std = np.array(dataset_std if dataset_std else self.std)

        # # Not sure why, but new version of pytorch is running into memory leak
        # self.image_paths = np.asarray(self.df['image_path'].values)
        # self.labels = np.asarray(self.df['category'].values)
        # del self.df

    @staticmethod
    def update_mean_std_with_dropped_bands(dropped_bands):
        mean = np.array(SentinelIndividualImageDataset.mean)
        std = np.array(SentinelIndividualImageDataset.std)
        keep_idxs = [i for i in range(mean.shape[0]) if i not in dropped_bands]
        return mean[keep_idxs].tolist(), std[keep_idxs].tolist()

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):

        # with rasterio.open(img_path) as data:
        #     # img = data.read(
        #     #     out_shape=(data.count, self.resize, self.resize),
        #     #     resampling=Resampling.bilinear
        #     # )
        #     img = data.read()  # (c, h, w)
        # img = img.transpose(1, 2, 0)  # (h, w, c)

        img = imread(img_path)  # (h, w, c)

        return img.astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]
        # selection = {'image_path': self.image_paths[idx], 'category': self.labels[idx]}

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection["image_path"])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = self.mean[self.masked_bands]

        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(images.shape[-1]) if i not in self.dropped_bands]
            images = images[:, :, keep_idxs]

        labels = self.categories.index(selection["category"])

        img_as_tensor = self.transform(images)  # (c, h, w)

        # sample = {
        #     'images': images,
        #     'labels': labels,
        #     'image_ids': selection['image_id'],
        #     'timestamps': selection['timestamp']
        # }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                (
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.2, 1.0), interpolation=interpol_mode, antialias=None
                    )
                    if use_antialias_key
                    else transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode)
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode, antialias=None
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class EuroSat(SatelliteDataset):
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, "r") as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label


def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == "rgb":
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == "temporal":
        dataset = CustomDatasetFromImagesTemporal(csv_path)
    elif args.dataset_type == "sentinel":
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(
            csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands
        )
    elif args.dataset_type == "rgb_temporal_stacked":
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = FMoWTemporalStacked.build_transform(is_train, args.input_size, mean, std)
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == "euro_sat":
        mean, std = EuroSat.mean, EuroSat.std
        transform = EuroSat.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)
    elif args.dataset_type == "naip":
        from util.naip_loader import NAIP_train_dataset, NAIP_test_dataset, NAIP_CLASS_NUM

        dataset = NAIP_train_dataset if is_train else NAIP_test_dataset
        args.nb_classes = NAIP_CLASS_NUM
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
