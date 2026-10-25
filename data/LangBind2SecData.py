import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LangBind2SecData(Dataset):
    """
    Dataset for 2-second LanguageBind embeddings.

    Expected directory structure:

    root_dir/
        train/
            class_1/
                video_1/
                    0/
                        langbind_output.pt
                    1/
                        langbind_output.pt
                video_2/
                    ...
            class_2/
                ...
        test/
            ...

    Each langbind_output.pt must contain:
        data["embedding"] -> 1024-D embedding
    """

    def __init__(
        self,
        root_dir,
        split="train",
        num_patches=20,
        mask_ratio=0.7,
        masking_strategy="random_masking",
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.num_patches = num_patches
        self.max_len = num_patches
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy

        self.split_dir = os.path.join(root_dir, split)

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(
                f"Dataset split not found: {self.split_dir}"
            )

        # Discover classes
        self.classes = sorted(
            [
                d
                for d in os.listdir(self.split_dir)
                if os.path.isdir(os.path.join(self.split_dir, d))
            ]
        )

        if len(self.classes) == 0:
            raise RuntimeError(
                f"No class directories found in {self.split_dir}"
            )

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # Discover videos
        self.samples = []

        for cls_name in self.classes:
            class_dir = os.path.join(self.split_dir, cls_name)

            video_names = sorted(
                [
                    d
                    for d in os.listdir(class_dir)
                    if os.path.isdir(os.path.join(class_dir, d))
                ]
            )

            for video_name in video_names:
                video_dir = os.path.join(class_dir, video_name)

                segment_dirs = [
                    d
                    for d in os.listdir(video_dir)
                    if os.path.isdir(os.path.join(video_dir, d))
                ]

                # Sort segment directories numerically when possible
                def segment_sort_key(x):
                    try:
                        return (0, int(x))
                    except ValueError:
                        return (1, x)

                segment_dirs = sorted(
                    segment_dirs,
                    key=segment_sort_key
                )

                segment_paths = []

                for segment in segment_dirs:
                    pt_path = os.path.join(
                        video_dir,
                        segment,
                        "langbind_output.pt"
                    )

                    if os.path.isfile(pt_path):
                        segment_paths.append(pt_path)

                if len(segment_paths) > 0:
                    self.samples.append(
                        (
                            segment_paths[:self.num_patches],
                            self.class_to_idx[cls_name],
                            cls_name,
                            video_name,
                        )
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid videos found in {self.split_dir}"
            )

        print("=" * 70)
        print("LangBind2SecData")
        print(f"Split          : {self.split}")
        print(f"Root           : {self.split_dir}")
        print(f"Classes        : {len(self.classes)}")
        print(f"Videos         : {len(self.samples)}")
        print(f"Num patches    : {self.num_patches}")
        print(f"Mask ratio     : {self.mask_ratio}")
        print(f"Mask strategy  : {self.masking_strategy}")
        print("=" * 70)

    def __len__(self):
        return len(self.samples)

    def _load_embeddings(self, segment_paths):
        embeddings = []

        for path in segment_paths:
            data = torch.load(path, map_location="cpu")

            if "embedding" not in data:
                raise KeyError(
                    f"'embedding' not found in {path}"
                )

            embedding = data["embedding"].float().flatten()

            if embedding.numel() != 1024:
                raise ValueError(
                    f"Expected 1024-D embedding, "
                    f"got shape {tuple(embedding.shape)} "
                    f"in {path}"
                )

            embeddings.append(embedding)

        embeddings = torch.stack(embeddings, dim=0)

        # Same normalization used by the existing CombinedData pipeline
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1
        )

        return embeddings

    def _random_masking(self, embeddings):
        """
        Randomly mask valid temporal embeddings.

        Returns:
            masked_embeddings
            ids_restore
            mask
            ids_shuffle
            len_keep
        """

        L = embeddings.shape[0]

        # At least one token must remain
        len_keep = max(
            1,
            int(L * (1 - self.mask_ratio))
        )

        # Random noise for temporal masking
        noise = torch.rand(L)

        # Smallest noise comes first
        ids_shuffle = torch.argsort(noise)

        # Restore original ordering
        ids_restore = torch.argsort(ids_shuffle)

        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:len_keep]

        masked_embeddings = torch.gather(
            embeddings,
            dim=0,
            index=ids_keep.unsqueeze(-1).expand(
                -1,
                embeddings.shape[1]
            )
        )

        # 0 = kept, 1 = masked
        mask = torch.ones(L)

        mask[:len_keep] = 0

        # Convert mask from shuffled order back to original order
        mask = torch.gather(
            mask,
            dim=0,
            index=ids_restore
        )

        return (
            masked_embeddings,
            ids_restore,
            mask,
            ids_shuffle,
            len_keep
        )

    def __getitem__(self, index):
        segment_paths, label, class_name, video_name = self.samples[index]

        # Load normalized embeddings
        embeddings = self._load_embeddings(segment_paths)

        original_length = embeddings.shape[0]

        # ------------------------------------------------------------
        # Apply masking
        # ------------------------------------------------------------
        (
            masked_embeddings,
            ids_restore,
            mask,
            ids_shuffle,
            masked_length
        ) = self._random_masking(embeddings)

        # ------------------------------------------------------------
        # Pad masked embeddings to num_patches
        # ------------------------------------------------------------
        embedding_dim = embeddings.shape[1]

        masked_padded_embeddings = torch.zeros(
            self.max_len,
            embedding_dim,
            dtype=embeddings.dtype
        )

        masked_padded_embeddings[
            :masked_length
        ] = masked_embeddings

        # ------------------------------------------------------------
        # Pad original normalized embeddings
        # ------------------------------------------------------------
        normalized_stacked_embeddings = torch.zeros(
            self.max_len,
            embedding_dim,
            dtype=embeddings.dtype
        )

        normalized_stacked_embeddings[
            :original_length
        ] = embeddings

        # ------------------------------------------------------------
        # Pad ids_restore
        # ------------------------------------------------------------
        ids_restore_padded = torch.zeros(
            self.max_len,
            dtype=torch.long
        )

        ids_restore_padded[
            :original_length
        ] = ids_restore

        # ------------------------------------------------------------
        # Pad ids_shuffle
        # ------------------------------------------------------------
        ids_shuffle_padded = torch.zeros(
            self.max_len,
            dtype=torch.long
        )

        ids_shuffle_padded[
            :original_length
        ] = ids_shuffle

        # ------------------------------------------------------------
        # Pad mask
        # ------------------------------------------------------------
        mask_padded = torch.ones(
            self.max_len,
            dtype=torch.float32
        )

        mask_padded[
            :original_length
        ] = mask

        # ------------------------------------------------------------
        # Attention mask
        #
        # False = valid token
        # True  = padding
        #
        # +1 because LV-MAE has CLS token.
        # ------------------------------------------------------------
        attn_mask = torch.ones(
            self.max_len + 1,
            dtype=torch.bool
        )

        # CLS token is always valid
        attn_mask[0] = False

        # Valid masked embeddings are also visible to encoder
        attn_mask[
            1:1 + masked_length
        ] = False

        return {
            "masked_padded_embeddings": masked_padded_embeddings,
            "ids_restore": ids_restore_padded,
            "ids_shuffle": ids_shuffle_padded,
            "masked_length": torch.tensor(
                masked_length,
                dtype=torch.long
            ),
            "normalized_stacked_embeddings": normalized_stacked_embeddings,
            "length": torch.tensor(
                original_length,
                dtype=torch.long
            ),
            "mask": mask_padded,
            "attn_mask": attn_mask,
        }
