from functools import partial
from torch.utils.data import Subset
import logging

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator

logger = logging.getLogger("dinov2")


class DINODataLoaderPartitioner:
    def __init__(
        self,
        dataset_path: str,
        global_crops_size: int,
        patch_size: int,
        mask_ratio_min_max: tuple,
        mask_sample_probability: float,
        global_crops_scale: tuple,
        local_crops_scale: tuple,
        local_crops_number: int,
        local_crops_size: int,
        batch_size_per_gpu: int,
        num_workers: int,
        inputs_dtype,
    ):
        self.dataset_path = dataset_path
        self.global_crops_size = global_crops_size
        self.patch_size = patch_size
        self.mask_ratio_min_max = mask_ratio_min_max
        self.mask_sample_probability = mask_sample_probability
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_workers = num_workers
        self.inputs_dtype = inputs_dtype
        self._setup_augmentation_and_masking()
    
    def _setup_augmentation_and_masking(self):
        # Calculate image and token parameters
        self.n_tokens = (self.global_crops_size // self.patch_size) ** 2
        
        # Setup masking generator for iBOT
        self.mask_generator = MaskingGenerator(
            input_size=(self.global_crops_size // self.patch_size, self.global_crops_size // self.patch_size),
            max_num_patches=self.mask_ratio_min_max[-1] * self.global_crops_size // self.patch_size * self.global_crops_size // self.patch_size,
        )
        
        # Setup data augmentation
        self.data_transform = DataAugmentationDINO(
            self.global_crops_scale,
            self.local_crops_scale,
            self.local_crops_number,
            global_crops_size=self.global_crops_size,
            local_crops_size=self.local_crops_size,
        )
        
        # Setup collate function
        self.collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=self.mask_ratio_min_max,
            mask_probability=self.mask_sample_probability,
            n_tokens=self.n_tokens,
            mask_generator=self.mask_generator,
            dtype=self.inputs_dtype,
        )
    
    def get_partitioned_dataloaders(self, n_partitions: int, start_iter: int = 0):
        dataset = make_dataset(
            dataset_str=self.dataset_path,
            transform=self.data_transform,
            target_transform=lambda _: (),
        )
        
        logger.info(f"Creating {n_partitions} partitioned dataloaders from dataset of size {len(dataset)}")
        # Calculate partition boundaries
        total_samples = len(dataset)
        samples_per_partition = total_samples // n_partitions
        
        dataloaders = []
        for partition_idx in range(n_partitions):
            start_idx = partition_idx * samples_per_partition
            # Last partition gets any remaining samples
            end_idx = (partition_idx + 1) * samples_per_partition if partition_idx < n_partitions - 1 else total_samples
            
            partition_indices = list(range(start_idx, end_idx))
            partition_dataset = Subset(dataset, partition_indices)
            
            logger.info(
                f"Partition {partition_idx}: samples {start_idx}-{end_idx} "
                f"({len(partition_indices)} samples)"
            )
            
            # Create dataloader for this partition
            dataloader = make_data_loader(
                dataset=partition_dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=self.num_workers,
                shuffle=True,
                seed=start_iter + partition_idx,  # Different seed per partition
                sampler_type=SamplerType.SHARDED_INFINITE,
                sampler_advance=0,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
            
            dataloaders.append(dataloader)
        
        logger.info(f"Successfully created {len(dataloaders)} partitioned dataloaders")
        return dataloaders
    
    def get_single_dataloader(self):
        dataset = make_dataset(
            dataset_str=self.dataset_path,
            transform=self.data_transform,
            target_transform=lambda _: (),
        )
        
        logger.info(f"Creating single dataloader for dataset of size {len(dataset)}")
        
        dataloader = make_data_loader(
            dataset=dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            shuffle=True,
            seed=0,
            sampler_type=SamplerType.SHARDED_INFINITE,
            sampler_advance=0,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        
        return dataloader
