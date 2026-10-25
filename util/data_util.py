import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from util import misc
from data.CombinedData import CombinedData
from data.LangBind2SecData import LangBind2SecData


def load_data(args):

    # ================================================================
    # OLD DATASET
    # ================================================================
    if args.dataset == 'combined':

        train_segments_metadata = (
            "/home/y_rajeshwar/final_year_project_29/"
            "dataset/2sec/metadata/train.csv"
        )

        val_segments_metadata = (
            "/home/y_rajeshwar/final_year_project_29/"
            "dataset/2sec/metadata/test.csv"
        )

        dataset_tr = CombinedData(
            csv_file=train_segments_metadata,
            mask_ratio=args.mask_ratio,
            num_patches=args.num_patches,
            duration_max=args.duration_max,
            importance_masking_strategy=args.masking_strategy
        )

        dataset_val = CombinedData(
            csv_file=val_segments_metadata,
            mask_ratio=args.mask_ratio,
            num_patches=args.num_patches,
            duration_max=args.duration_max,
            importance_masking_strategy=args.masking_strategy
        )

    # ================================================================
    # NEW 2-SECOND LANGUANGEBIND DATASET
    # ================================================================
    elif args.dataset == 'langbind_2sec':

        dataset_tr = LangBind2SecData(
            root_dir=args.data_path,
            split="train",
            num_patches=args.num_patches,
            mask_ratio=args.mask_ratio,
            masking_strategy=args.masking_strategy
        )

        dataset_val = LangBind2SecData(
            root_dir=args.data_path,
            split="test",
            num_patches=args.num_patches,
            mask_ratio=args.mask_ratio,
            masking_strategy=args.masking_strategy
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}"
        )

    # ================================================================
    # TRAIN DATALOADER
    # ================================================================
    if args.distributed:

        sampler_train = DistributedSampler(
            dataset_tr,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )

        sampler_val = DistributedSampler(
            dataset_val,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )

        data_loader_train = DataLoader(
            dataset_tr,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

        data_loader_val = DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    else:

        data_loader_train = DataLoader(
            dataset_tr,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    print("=" * 70)
    print("DATA LOADER")
    print("=" * 70)
    print(f"Train dataset size : {len(dataset_tr)}")
    print(f"Val dataset size   : {len(dataset_val)}")
    print(f"Train batches      : {len(data_loader_train)}")
    print(f"Val batches        : {len(data_loader_val)}")
    print("=" * 70)

    return data_loader_train, data_loader_val
