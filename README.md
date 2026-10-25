# Masked Spatio-Temporal Learning for Multi-Class Crime Detection

## 📰 Updates

- We extend the LV-MAE framework toward multi-class crime detection using short video segments and spatio-temporal representation learning.
- This repository includes the modified preprocessing, LanguageBind embedding extraction, model fine-tuning, and evaluation pipelines.

## 🎯 Overview

This project explores a deep learning framework for multi-class crime detection in videos, building upon **LV-MAE (Learning Long Video Representations through Masked-Embedding Autoencoders)**.

The pipeline processes videos into short segments, extracts multimodal video representations using LanguageBind, and uses learned spatio-temporal features for downstream crime classification.

The project explores the following components:

- **Video segmentation:** Divides videos into 2-second segments for finer-grained temporal analysis.
- **LanguageBind feature extraction:** Converts video segments into embedding representations.
- **LV-MAE representation learning:** Learns high-level temporal relationships between segment embeddings.
- **Summary Token:** Provides a compact representation intended to aggregate information across the video sequence.
- **Temporal attention:** Helps the model focus on informative segments and temporal relationships.
- **Crime classification:** Fine-tunes the learned representations to predict crime categories.

The objective is to investigate whether combining masked-embedding learning with temporal modeling can improve video-based crime recognition.

## 🏗️ Methodology

The overall pipeline consists of the following stages:

1. **Input video:** A crime-related or non-crime video is provided as input.
2. **Video segmentation:** The video is divided into consecutive 2-second segments.
3. **Preprocessing:** Segment metadata and input paths are prepared for feature extraction.
4. **LanguageBind encoder:** Each segment is processed to obtain an embedding representation.
5. **Spatio-temporal representation learning:** The LV-MAE framework models relationships between segment embeddings.
6. **Summary Token and temporal attention:** The modified architecture aggregates temporal information and emphasizes informative segments.
7. **Fine-tuning:** The learned representations are used to train a classifier for multiple crime categories.
8. **Prediction:** The model produces a predicted class for the input video.

## 🛠️ Setup

Clone the repository:

```bash
git clone https://github.com/BommadiHemanthKumarreddy/Masked-Spatio-Temporal-Learning-for-Multi-Class-Crime-Detection.git
cd Masked-Spatio-Temporal-Learning-for-Multi-Class-Crime-Detection
```

Create the Conda environment using the environment configuration provided in the repository:

```bash
conda env create -f environment.yml
```

Activate the environment using the environment name defined in `environment.yml`:

```bash
conda env list
conda activate <environment_name>
```

Install any additional dependencies required by your environment and verify that the appropriate PyTorch and CUDA versions are available.

## 📊 Dataset and Preprocessing

The project uses video data for multi-class crime detection.

**Dataset:** [Add the exact dataset name and its official source.]

The dataset must be prepared before model training. The preprocessing pipeline includes:

1. **Video segmentation:** Split the source videos into 2-second clips.
2. **Metadata generation:** Create metadata containing segment paths and the information required for processing.
3. **Feature extraction:** Use LanguageBind to extract embeddings for the generated segments.
4. **Dataset preparation:** Organize the embeddings and labels into the format expected by the fine-tuning dataset loader.
5. **Train/test preparation:** Generate the appropriate dataset splits while avoiding overlap or leakage between related videos.

### Preprocessing scripts

The repository contains scripts for segment metadata creation, preprocessing, and LanguageBind feature extraction. Review the command-line arguments in each script before running it.

Relevant scripts include:

- `data_preprocess/create_2sec_metadata.py`
- `data_preprocess/create_prepro_json.py`
- `data_preprocess/langbind_preprocessing_2sec.py`
- `data_preprocess/langbind_preprocessing_2sec_v2.py`
- `data_preprocess/segment_video_library_movie_clip_2sec.py`
- `data_preprocess/segment_dataframe_movie_clip.py`
- `data_preprocess/prepare_paired_paths.py`

The exact scripts required depend on the dataset and the preprocessing stage.

**Note:** Dataset files, generated embeddings, and model checkpoints are not included in this repository unless explicitly provided.

## 🧠 Model Architecture

### LV-MAE Backbone

The project builds upon LV-MAE, which learns video representations by masking and reconstructing segment-level embeddings. Instead of directly operating on every raw video frame throughout the full sequence, the embedding-based approach models higher-level video information.

### Summary Token

The modified architecture incorporates a Summary Token intended to aggregate information from the video sequence into a compact representation. This representation can be used by the downstream classification pipeline.

### Temporal Attention

Temporal attention is used to model the relative importance of video segments and their relationships over time. It is intended to help the model emphasize informative temporal patterns associated with different classes.

### Multi-Class Crime Classifier

The fine-tuning stage maps the learned video representation to the target crime categories. The predicted class depends on the labels and category definitions used in the training dataset.

## 🔥 Pre-training

LV-MAE pre-training is performed using `main_pretrain.py`.

Example command structure:

```bash
python main_pretrain.py \
    --batch_size 16 \
    --dataset combined \
    --decoder_depth 8 \
    --decoder_embed_dim 512 \
    --decoder_num_heads 16 \
    --depth 32 \
    --epochs 151 \
    --input_size 1024 \
    --mask_ratio 0.5 \
    --num_heads 16 \
    --num_patches 256
```

These arguments are an example based on the upstream LV-MAE configuration. Confirm the available arguments and required dataset settings in your version of `main_pretrain.py` before running the command.

## 🎯 Fine-tuning

Fine-tune the model for multi-class crime classification using `main_finetune.py`.

```bash
python main_finetune.py --help
```

Configure the dataset paths, number of classes, model checkpoint, training parameters, and other required arguments according to the script.

The fine-tuning pipeline uses the prepared segment embeddings and corresponding labels to train the crime classification model.

## 📈 Evaluation

Evaluate the fine-tuned model using the evaluation scripts included in the repository.

```bash
python evaluate_finetune.py --help
```

Additional scripts are available for testing the model, inspecting embeddings, and extracting attention information, including:

- `test.py`
- `test_classifier.py`
- `test_dataset.py`
- `test_extract.py`
- `test_langbind_clip.py`
- `test_load.py`
- `extract_attention.py`

Report evaluation metrics such as accuracy, precision, recall, and F1-score using the actual test-set results. Specify the dataset split and evaluation protocol so that results can be interpreted consistently.

## 📁 Repository Structure

```text
.
├── assets/
├── data/
├── data_preprocess/
├── languagebind/
├── loggers/
├── util/
├── classifier.py
├── dataset_finetune.py
├── engine_finetune.py
├── evaluate_finetune.py
├── main_finetune.py
├── main_pretrain.py
├── models_mae.py
├── test.py
├── environment.yml
└── README.md
```

## ⚠️ Notes

- The dataset must be downloaded and prepared separately.
- Generated embeddings and trained model checkpoints may require substantial storage.
- The training commands must match the arguments implemented in the current scripts.
- Performance claims should be based on reproducible evaluation results.
- This repository is intended for research and educational purposes.

## 🙏 Acknowledgments

This project builds upon the LV-MAE framework:

- **LV-MAE:** [Learning Long Video Representations through Masked-Embedding Autoencoders](https://arxiv.org/abs/2504.03501)
- **Original LV-MAE repository:** [amazon-science/lv-mae](https://github.com/amazon-science/lv-mae)
- **MAE:** [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **LanguageBind:** [LanguageBind GitHub repository](https://github.com/PKU-YuanGroup/LanguageBind)

Please refer to the original papers and repositories for their respective licenses and citation requirements.

## 📚 BibTeX

```bibtex
@article{naiman2025lvmae,
  title={{LV-MAE:} Learning Long Video Representations through Masked-Embedding Autoencoders},
  author={Naiman, Ilan and Ben Baruch, Emanuel and Anschel, Oron and
          Shoshan, Alon and Kviatkovsky, Igor and Aggarwal, Manoj and
          Medioni, G{\'e}rard G.},
  journal={CoRR},
  volume={abs/2504.03501},
  year={2025}
}
```

If you use this repository in academic work, please cite the original LV-MAE paper and any additional datasets or models that your work builds upon.
