# LungNet-Hybrid3D
3D hybrid lung nodule segmentation + lesion-level classification pipeline.
Features:
- 3D self-supervised pretraining hook (Models-Genesis style)
- Hybrid encoder (3D CNN) + lightweight 3D Transformer bottleneck
- Multi-task heads: segmentation + malignancy classification + size regression
- MC-dropout for uncertainty + simple radiomics FP reduction

Follow `setup_env.sh`, prepare dataset in `data/`, then run training: `bash scripts/run_train.sh`.
