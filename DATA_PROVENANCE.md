# Data and model provenance

The local verification corpus was imported from the user-supplied `data(1).zip`, which contains
500 COCO validation images and legacy YOLO metadata. Only a bounded 30-image sample was extracted
into ignored `runtime/` storage. No supplied images, legacy absolute paths, or model weights are
committed to this repository.

Reviewers may index their own JPG, JPEG, PNG, or WebP files. The import command deliberately
flattens archive paths to filenames and applies an explicit image-count limit. Users remain
responsible for the licenses and privacy rights of images they ingest.

The verified run downloaded Ultralytics `yolo11n.pt` and OpenCLIP ViT-B/32 LAION2B weights from
their respective official model distribution mechanisms. Model caches and `.pt` files are
excluded from Git.

