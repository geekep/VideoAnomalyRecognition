#!/bin/sh

#BSUB -q normal

#BSUB -o %J.out

#BSUB -e %J.err

#BSUB -n 1

#BSUB -J JOBNAME

#BSUB  -R span[ptile=1]

#BSUB -m "user-g4a60"

#BSUB  -gpu  num=4

python TrainingAnomalyDetector_public.py --features_path c3d_out --annotation_path Anomaly_Detection_splits/Train_Annotation.txt --annotation_path_test Anomaly_Detection_splits/Test_Annotation.txt

#python feature_extractor.py --dataset_path --annotation_path Train_Annotation.txt --annotation_path_test Test_Annotation.txt --pretrained_3d ./network/c3d.pickle

#python generate_ROC.py --features_path c3d_out --annotation_path Test_Annotation.txt