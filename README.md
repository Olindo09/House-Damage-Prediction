# House Monitoring – TensorFlow Object Detection (TF1)

End-to-end pipeline to train and validate an SSD MobileNet V2 object detection model using the TensorFlow Object Detection API on custom, PASCAL‑VOC–style annotated data. Includes dataset preparation (XML→CSV→TFRecord), training with TensorBoard, exporting a SavedModel, and running inference/visualization on test images.

> **Colab-first**: The notebooks are written for Google Colab and mount Google Drive for data/model I/O.
> You can also run locally with TF 1.15 and the TensorFlow Object Detection API (legacy).

---

## Repository structure

```
.
├── House_monitoring.ipynb                 # training & export
├── vaildation_HM.ipynb                    # quick inference
├── comment_vaildation_HM_(1)_txt.ipynb    # commented inference walkthrough
└── (data/, training/, fine_tuned_model/, ...)  # created at runtime on Drive/locally
```

## What the notebooks do

- `House_monitoring.ipynb` — Main training pipeline: installs deps, converts VOC XML to CSV, builds TFRecords, configures SSD MobileNet V2, runs training with TensorBoard & exports a SavedModel.
- `vaildation_HM.ipynb` — Loads the exported SavedModel and runs inference on sample/test images with visualizations.
- `comment_vaildation_HM_(1)_txt.ipynb` — Commented validation walkthrough with the same inference flow as above, plus additional explanations.

---

## Setup (Colab)

The notebooks install exact versions inside the runtime. Key packages observed in the code:

- `tensorflow==1.15.0` (with `%tensorflow_version 1.x` in Colab)
- `tf_slim`
- `pycocotools`, `lxml`, `Pillow`, `matplotlib`, `Cython`
- (occasionally) `lvis`
- TensorFlow Object Detection API (legacy, compiled with `protoc`)

**Steps you will see in the notebooks:**

1. **Mount Drive**
   ```python
   from google.colab import drive
   drive.mount('/drive')
   ```

2. **Install dependencies** (done in-notebook using `pip/apt-get`).  
   Compile TF OD API protos:
   ```bash
   %cd /drive/MyDrive/Housedamage/models/research/
   protoc object_detection/protos/*.proto --python_out=.
   ```

3. **(Optional) TensorBoard** via `ngrok` tunnel for remote viewing while training.

---

## Data preparation

The training notebook expects **PASCAL VOC XML** annotations and converts them to CSV, then to TFRecords.

### 1) XML → CSV

A helper like `xml_to_csv()` scans `train_label/` and `test_label/` folders and produces CSVs with columns:
`filename, width, height, class, xmin, ymin, xmax, ymax`

> Place your images + XMLs in:
>
> - `train_label/` (training split)
> - `test_label/` (evaluation split)

### 2) Label map

Create `data/label_map.pbtxt` mapping your classes to integer IDs, e.g.:
```protobuf
item { id: 1  name: "class_a" }
item { id: 2  name: "class_b" }
# add more as needed
```

### 3) CSV → TFRecord

The notebook builds `train.record` and `test.record` from those CSVs and images.

> **Tip:** Keep image file names consistent with the `filename` field in the XMLs.

---

## Model & training

The pipeline uses **SSD MobileNet V2** (COCO) configuration adapted for your dataset. In the notebook you will see a block defining or referencing the `.config` with fields like:

- `num_classes` (match to your label map)
- `fine_tune_checkpoint` (COCO checkpoint)
- `train_input_reader` / `eval_input_reader` with `*.record` paths and `label_map_path`
- `image_resizer` (e.g., 300×300)

Start training via:
```bash
python3 object_detection/model_main.py   --pipeline_config_path=/drive/MyDrive/Housedamage/models/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config   --model_dir=training/
```

### Monitoring

TensorBoard is launched in-notebook and (optionally) exposed via `ngrok`. Loss curves and eval metrics appear under `training/`.

### Exporting the model

After training, the notebook picks the latest `model.ckpt-XXXX` and exports an inference graph / SavedModel:

```bash
python object_detection/export_inference_graph.py   --input_type=image_tensor   --pipeline_config_path=.../ssd_mobilenet_v2_coco.config   --output_directory=./fine_tuned_model   --trained_checkpoint_prefix=training/model.ckpt-<STEP>
```

> In the commented validation, a SavedModel is loaded with:
> ```python
> tf.saved_model.load("/drive/MyDrive/Housedamage/models/research/fine_tuned_model/saved_model")
> ```

---

## Inference & visualization

Both validation notebooks load the SavedModel, then run detection + draw boxes/labels:

```python
def run_inference_for_single_image(model, image_np):
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
    outputs = model(input_tensor)
    # post-process: num_detections, classes (int), boxes, scores, masks (if any)
    ...

from object_detection.utils import label_map_util, visualization_utils as vis_util

# Label map & test images
PATH_TO_LABELS = '/drive/MyDrive/Housedamage/data/label_map.pbtxt'
TEST_IMAGE_PATHS = list(Path('/drive/MyDrive/Housedamage/data/test').glob('*.jpg'))

# Visualize
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    line_thickness=2
)
```

Save or display the annotated images directly in Colab.

---

## How to reproduce (quick start)

1. Put your dataset on Drive, e.g. under `/drive/MyDrive/Housedamage/` with subfolders:
   ```
   data/
     label_map.pbtxt
     train_label/  # images + XML
     test_label/   # images + XML
   ```
2. Open **`House_monitoring.ipynb`** in Colab and **run all cells** (edit paths/classes as needed).
3. After training finishes and the model is exported, open **`vaildation_HM.ipynb`** (or the commented version) and run inference on sample images in `data/test/`.
4. (Optional) Commit exported artifacts (config, label map, a few sample images) to this repo for reproducibility.

---

## Results

_Add sample detections, mAP/precision, confusion matrix, and training curves here._  
E.g., include a few before/after images and report **precision @ IoU=0.5** on your test split.

---

## Notes

- **TensorFlow 1.x** is EOL; consider porting to TF2 OD API or [Ultralytics YOLOv5/8] for simpler modern pipelines.
- If you change image size or aspect ratio, ensure your training config and preprocessing match.
- Keep the label map IDs stable between training and inference.

---

## Citation / Acknowledgements

- TensorFlow Object Detection API (legacy): https://github.com/tensorflow/models/tree/master/research/object_detection
- SSD MobileNet V2 (COCO) checkpoint & sample configs from TensorFlow Models.

---



