/usr/local/python-3.6.5/bin/python3 ./train_faster_rcnn.py \
    --epochs 20 \
    --gpus="4,5,6,7" \
    --batch_size 4 \
    --learning_rate 0.001 \
    --save_path="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/light-head-rcnn-vgg16-9anchors" \
    --save_interval 10000 \
    --model="vgg16" \
    --feature_name="vgg0_conv12_fwd_output" \
