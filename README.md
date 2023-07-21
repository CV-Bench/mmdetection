install with

```bash
docker build -t mmdetection -f docker/Dockerfile .
```

run with

```bash
docker run --gpus all --shm-size=8g -it -v "$(pwd)/data:/mmdetection/data" -v "$(pwd)/configs:/mmdetection/configs" mmdetection
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest /mmdetection/configs/yolo
python tools/train.py configs/_user_/yolo_v3.py
```