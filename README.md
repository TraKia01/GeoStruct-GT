# GeoStruct-GT

A method for fine-grained document structure analysis that improves the parsing accuracy of ancient books and dictionary documents. The Fine-grained Document Structure Dataset (FDSD) is constructed based on the Great Chinese Dictionary.

## Overview

- [Dataset Download](https://drive.google.com/file/d/1WCjt7CiypWIUIJsfj5iOTPhUkDwknHXk/view?usp=sharing)

**Model Architecture**

![Model Diagram](<./GeoStruct-GT 模型结构图片 (1).png>)

**Dataset Example**

![Dataset Example](./label_list.png)

## Files

- `geostruct_gt_train.py`: training
- `geostruct_gt_inference.py`: inference without graph post-processing
- `geostruct_gt_model.py`: model
- `geostruct_gt_dataset.py`: dataset loader
- `geostruct_gt_evaluator.py`: evaluation
- `geostruct_gt_constraints.py`: constraint utilities

## Environment

- Python 3.10 or 3.11
- PyTorch
- `ultralytics`

Install:

```bash
pip install torch torchvision
pip install ultralytics opencv-python numpy tqdm matplotlib networkx
```

## Data Layout

```text
dataset_gdsa/
  images/
    train/
    val/
  labels/
    train/
    val/
  classes.txt

dict_spatial_rels/
dict_logic_rels/
```

## Training

```bash
python geostruct_gt_train.py --yolo yolo11n.pt --epochs 100
```

```bash
python geostruct_gt_train.py --yolo yolo11n.pt --epochs 100 --rel-train-mode pred
```

## Inference

```bash
python geostruct_gt_inference.py --checkpoint runs/geostruct_gt_yolo11n_cw0.1/best_mAP_g_05.pth --image test.jpg
```

```bash
python geostruct_gt_inference.py --checkpoint runs/geostruct_gt_yolo11n_cw0.1/best_mAP_g_05.pth --image test.jpg --use-original-size --max-size 1280
```
