# Two-stream-View-Invariance-SSL-on-Assembly101
We propose a pretraining method that uses two-streams of SSL to learn View-Invariance representation on Assembly101. Note that this method requires synchronized videos, and is concurrent with more advanced work Viewpoint Rosetta Stone(CVPR 2025) which uses unsynchronized videos. So do take a look at their work if you want to work on this.

Download Assembly101 dataset and annotation at: https://assembly-101.github.io/

The project report is here: [/materials](https://github.com/XuZhangNYU/Two-stream-View-Invariance-SSL-on-Assembly101/blob/main/materials/DS_GA_3001_Project_Report_Team_12_Xu_Sihang.pdf)

More technical parts:

Before running, make sure .pkl which records statistics of videos such as max frames is ready (our experiment pkl is already included here). If not, consider run data_stat.py. 

The annotation file and environment file are included in the folder. 

To run pretraining, run ssl_pretraining.py:

python3 ssl_pretraining.py \
--action train \
--feature_path /home/xz4863/scratch/scratch_folder/TSM_features/e4 \
--teacher_feature_path /home/xz4863/scratch/scratch_folder/TSM_features/C10118_rgb \
--ssl_method our_method \ 
--split train \

Among the arguments, change ssl_method can choose the ssl settings: "sync_kd" for synchronization knowledge distillation baseline, "ablation_simple_3d" for the ablation setting with simpler pseudo-3d stream architecture, "our_method" for our ssl setup.

'feature_path' and 'teacher_feature_path' are the synchronized video feature paths of different views used for ssl. It expects a mdb file with video frames already extracted as tensors and being stored in the mdb file. For example, for synchronization kd baseline, 'teacher_feature_path' will be egocentric videos' feature path, and 'feature_path' will be the exocentric video feature path. We included a dummy feature_path with trivial files to help test whether the code runs. 

The mdb feature path files works like a dictionary where its values are frame-wise feature, and keys are in the format :
nusar-2021_action_both_9016-c10b_9016_user_id_2021-02-17_143201/C10118_rgb/C10118_rgb_0000000081.jpg (i.e. video_name/view_name/view_name_frame_num.jpg)

To run training, choose the file that corresponds to the desired setting that you want to run: main_train_[baseline/our_method/sync_kd]. Essentially those files are pretty similar except the part that does forward of the model. Also change the 'views' to be the specific view that you want to use as train data. Since our experiment is always on labled exocentric data, we never twist this.

python3 main_train_specific_setting_name.py \
    --action train \
    --feature_path /scratch/xz4863/TSM_features \
    --split train \

To run evaluation, run the corresponding evaluation file of desired setting: testset_eval_[setting].py. You can choose the view point and weights in the code. Go to function 'predict' to change weight.
python3 testset_eval_baseline.py \
    --action predict \
    --feature_path /scratch/xz4863/TSM_features \
    --test_aug 0 \


