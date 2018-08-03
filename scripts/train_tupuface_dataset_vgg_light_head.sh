/usr/local/python-3.6.5/bin/python3 ./train_faster_rcnn.py \
    --epochs 20 \
    --gpus="0,1,2,3" \
    --batch_size 4 \
    --learning_rate 0.0005 \
    --save_path="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/light-head-rcnn-vgg16-9anchors" \
    --save_interval 10000 \
    --model="vgg16" \
    --feature_name="vgg0_conv12_fwd_output" \
    --pretrained_model="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/light-head-rcnn-vgg16-9anchors/light-head-rcnn-vgg16-9anchors-10000.gluonmodel"
