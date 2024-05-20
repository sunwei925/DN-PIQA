CUDA_VISIBLE_DEVICES=0,1 python -u models/train.py \
--num_epochs 10 \
--batch_size 6 \
--resize 448 \
--crop_size 384 \
--lr 0.00001 \
--decay_ratio 0.9 \
--decay_interval 2 \
--snapshot /data/sunwei_data/ModelFolder/StairIQA/PIQ/ \
--database_dir /data/sunwei_data/PIQ2023/Dataset/Overall/ \
--feature_dir /data/sunwei_data/PIQ2023/LIQE_feature/Overall/ \
--face_dir /data/sunwei_data/PIQ2023/Dataset/face/ \
--model DN_PIQA \
--pretrained_path /home/sunwei/code/IQA/StairIQA-main/ckpts/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth \
--pretrained_path_face /home/sunwei/code/IQA/StairIQA-main/ckpts/Swin_b_384_in22k_SlowFast_Fast_pretrained_GFIQA_L2_NR_v0_epoch_8_SRCC_0.973102.pth \
--multi_gpu \
--with_face \
--print_samples 200 \
--database PIQ \
--test_method five \
--num_patch 0 \
--loss_type fidelity \
>> logs/train.log

CUDA_VISIBLE_DEVICES=0 python -u models/LIQE.py \
--csv_path csvfiles/ntire24_overall_scene_train.csv \
--data_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--feature_save_folder /data/sunwei_data/PIQ2023/LIQE_feature/pip_ft2/ \
>> logs/extract_LIQE_features_train.log

CUDA_VISIBLE_DEVICES=0 python -u models/LIQE.py \
--csv_path csvfiles/ntire24_overall_scene_test.csv \
--data_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--feature_save_folder /data/sunwei_data/PIQ2023/LIQE_feature/pip_ft2/ \
>> logs/extract_LIQE_features_test.log

CUDA_VISIBLE_DEVICES=0 python -u models/extract_face_images.py \
--csv_path csvfiles/ntire24_overall_scene_train.csv \
--image_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--face_save_dir /data/sunwei_data/PIQ2023/Dataset/face2/ \
>> logs/extract_face_images_train.log

CUDA_VISIBLE_DEVICES=0 python -u models/extract_face_images.py \
--csv_path csvfiles/ntire24_overall_scene_test.csv \
--image_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--face_save_dir /data/sunwei_data/PIQ2023/Dataset/face2/ \
>> logs/extract_face_images_test.log