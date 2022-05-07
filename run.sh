#python usage_examples/swinT_example_test.py \
python cam_test.py \
    --use-cuda \
    --method gradcam \
    --model_name resnext50_32x4d \
    --image-path /data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/test/none_adenoma/ \
    --pth_dir /data3/qilei_chen/DATA/polyp_xinzi/preprocessed_4_classification/work_dir/resnext50_32x4d-224/model_best.pth.tar \
    --save_dir /data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/cam/resnext50_32x4d/none_adenoma