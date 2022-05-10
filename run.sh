python usage_examples/swinT_example_test.py \
    --use-cuda \
    --method gradcam \
    --model_name swin_base_patch4_window7_224-224 \
    --image-path /data3/qilei_chen/DATA/polyp_xinzi/D1_D2/ \
    --pth_dir /data3/qilei_chen/DATA/polyp_xinzi/D1_D2/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar \
    --save_dir /data3/qilei_chen/DATA/polyp_xinzi/D1_D2/cam/swin_base_patch4_window7_224/