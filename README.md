# DN-PIQA
This is a repository for the models proposed in the paper "Dual-Branch Network for Portrait Image Quality Assessment".

## Usage
### Install the environment 
Note that the version of pytorch should be 1.10.0 (for face detection) and the version of timm should be 0.6.7
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
pip install timm==0.6.7
pip install ipython matplotlib opencv-python pandas PyYAML scipy seaborn tqdm requests thop
```

### Download the pre-trained model
Download the models: [LIQE.pt](https://drive.google.com/file/d/1jEG9IwyqJqplUtCrR49MjVEswPXA9V6S/view?usp=sharing), [preweight.pt](https://drive.google.com/file/d/1MvB55XoWpI6iQHTv8K4kzxuq5gtrbYNw/view?usp=sharing), [PIQ_model.pth](https://drive.google.com/file/d/1y3T8DCwhYZtLN_QrN70qE8yZ1r8Bi-dG/view?usp=sharing), and put them into the folder of weights


### Test the model
```
CUDA_VISIBLE_DEVICES=0 python -u models/test_images.py --image_path samples/ --image_name 1000_Indoor_Scene_10.jpg
```

### Train DN-PIQA model on PIQ dataset
- Download the [PIQ](https://corp.dxomark.com/data-base-piq23/) dataset.
- Download the pre-trained models on [LSVQ](https://drive.google.com/file/d/1jgzVV0sil0kGhhHIV0RLr6YoDZNp7LNi/view?usp=sharing) and [GFIQA](https://drive.google.com/file/d/18KNadKCEOyQ31xqtGm_uvhMoCVM5NVss/view?usp=sharing).
- Extract the LIQE features:
```
CUDA_VISIBLE_DEVICES=0 python -u models/LIQE.py \
--csv_path csvfiles/ntire24_overall_scene_train.csv \
--data_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--feature_save_folder /data/sunwei_data/PIQ2023/LIQE_feature/Overall/ \
>> logs/extract_LIQE_features_train.log

CUDA_VISIBLE_DEVICES=0 python -u models/LIQE.py \
--csv_path csvfiles/ntire24_overall_scene_test.csv \
--data_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--feature_save_folder /data/sunwei_data/PIQ2023/LIQE_feature/Overall/ \
>> logs/extract_LIQE_features_test.log
```

- Extract the face images:
```
CUDA_VISIBLE_DEVICES=0 python -u models/extract_face_images.py \
--csv_path csvfiles/ntire24_overall_scene_train.csv \
--image_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--face_save_dir /data/sunwei_data/PIQ2023/face/ \
>> logs/extract_face_images_train.log

CUDA_VISIBLE_DEVICES=0 python -u models/extract_face_images.py \
--csv_path csvfiles/ntire24_overall_scene_test.csv \
--image_dir /data/sunwei_data/PIQ2023/Dataset/Overall \
--face_save_dir /data/sunwei_data/PIQ2023/face/ \
>> logs/extract_face_images_test.log
```






- Train the model
```
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
--pretrained_path weights/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth \
--pretrained_path_face weights/Swin_b_384_in22k_SlowFast_Fast_GFIQA \
--multi_gpu \
--with_face \
--print_samples 200 \
--database PIQ \
--test_method five \
--num_patch 0 \
--loss_type fidelity \
>> logs/train.log
```


## Acknowledgement

1. <https://github.com/zwx8981/LIQE>
2. <https://github.com/deepcam-cn/yolov5-face>
3. <https://github.com/zwx8981/UNIQUE>