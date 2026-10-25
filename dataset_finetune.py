import os
import torch
from torch.utils.data import Dataset


class LangBindDataset(Dataset):

    def __init__(self, root_dir, num_patches=20):

        self.root_dir = root_dir
        self.num_patches = num_patches
        self.samples = []

        # --------------------------------------------------------
        # FIND CLASSES
        # --------------------------------------------------------

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        # --------------------------------------------------------
        # FIND VIDEOS
        # --------------------------------------------------------

        for cls in self.classes:

            cls_path = os.path.join(
                root_dir,
                cls
            )

            for video in sorted(os.listdir(cls_path)):

                video_path = os.path.join(
                    cls_path,
                    video
                )

                if not os.path.isdir(video_path):
                    continue

                # ------------------------------------------------
                # FIND SEGMENTS
                # ------------------------------------------------

                segments = [
                    s for s in os.listdir(video_path)
                    if os.path.isdir(
                        os.path.join(video_path, s)
                    )
                ]

                # ------------------------------------------------
                # NUMERIC SEGMENT SORTING
                #
                # segment_1
                # segment_2
                # ...
                # segment_10
                #
                # instead of lexical:
                #
                # segment_1
                # segment_10
                # segment_2
                # ------------------------------------------------

                def segment_key(name):

                    last_part = name.split("_")[-1]

                    if last_part.isdigit():
                        return int(last_part)

                    return name

                segments = sorted(
                    segments,
                    key=segment_key
                )

                # ------------------------------------------------
                # FIND LANGUAGEBIND EMBEDDINGS
                # ------------------------------------------------

                segment_paths = []

                for segment in segments:

                    pt_path = os.path.join(
                        video_path,
                        segment,
                        "langbind_output.pt"
                    )

                    if os.path.isfile(pt_path):

                        segment_paths.append(
                            pt_path
                        )

                # ------------------------------------------------
                # KEEP VIDEO IF IT HAS EMBEDDINGS
                # ------------------------------------------------

                if segment_paths:

                    self.samples.append(
                        (
                            segment_paths[
                                :self.num_patches
                            ],
                            self.class_to_idx[cls]
                        )
                    )

        # --------------------------------------------------------
        # DATASET INFORMATION
        # --------------------------------------------------------

        print("=" * 70)
        print("LANGUAGEBIND DATASET")
        print("=" * 70)

        print(
            f"Root directory : {root_dir}"
        )

        print(
            f"Videos         : {len(self.samples)}"
        )

        print(
            f"Classes        : {len(self.classes)}"
        )

        print(
            f"Num patches    : {self.num_patches}"
        )

        print(
            f"Class mapping  : {self.class_to_idx}"
        )

        print("=" * 70)

    # ------------------------------------------------------------
    # LENGTH
    # ------------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # ------------------------------------------------------------
    # GET ITEM
    # ------------------------------------------------------------

    def __getitem__(self, idx):

        segment_paths, label = self.samples[idx]

        embeddings = []

        # --------------------------------------------------------
        # LOAD EACH 2-SECOND LANGUAGEBIND EMBEDDING
        # --------------------------------------------------------

        for path in segment_paths:

            data = torch.load(
                path,
                map_location="cpu"
            )

            embedding = data["embedding"].float().flatten()

            # ----------------------------------------------------
            # CHECK EMBEDDING DIMENSION
            # ----------------------------------------------------

            if embedding.numel() != 1024:

                raise ValueError(
                    f"Expected 1024-D embedding, "
                    f"but got {embedding.numel()} "
                    f"in {path}"
                )

            embeddings.append(
                embedding
            )

        # --------------------------------------------------------
        # STACK SEGMENTS
        # --------------------------------------------------------

        features = torch.stack(
            embeddings
        )

        # Actual number of valid segments
        length = features.shape[0]

        # --------------------------------------------------------
        # PAD TO num_patches = 20
        # --------------------------------------------------------

        if length < self.num_patches:

            padding = torch.zeros(
                self.num_patches - length,
                features.shape[1],
                dtype=features.dtype
            )

            features = torch.cat(
                [
                    features,
                    padding
                ],
                dim=0
            )

        # --------------------------------------------------------
        # RETURN
        #
        # features : (20, 1024)
        # length   : actual number of segments
        # label    : class index
        # --------------------------------------------------------

        return (
            features,
            length,
            label
        )
