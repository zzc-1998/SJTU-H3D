# CUDA_VISIBLE_DEVICES=0 python -u ./quality_measure_utils/semantic_affinity_quality_measure.py \
#  --database SJTU-H3D \
#  --model_name  ViT-B-32 \
#  --pretrained laion2b_s34b_b79k \
#  --info_path ./datainfo/H3D_datainfo.csv \
#  --data_path /DATA/digital_human/alloy_humans/dis_6projections \
#  --output_csv ./output_csvs/H3D_SAQM.csv \
#  >> logs/SJTU-H3D.log  

#  CUDA_VISIBLE_DEVICES=0 python -u ./quality_measure_utils/geometry_loss_quality_measure.py \
#  --database SJTU-H3D \
#  --info_path ./datainfo/H3D_datainfo.csv \
#  --data_path /DATA/digital_human/alloy_humans/dis_objs/ \
#  --output_csv ./output_csvs/H3D_GLQM.csv \
#  >> logs/SJTU-H3D.log    


 CUDA_VISIBLE_DEVICES=0 python -u ./quality_measure_utils/spatial_naturalness_quality_measure.py \
 --database SJTU-H3D \
 --info_path ./datainfo/H3D_datainfo.csv \
 --data_path /DATA/digital_human/alloy_humans/dis_6projections \
 --output_csv ./output_csvs/H3D_SNQM.csv \
 >> logs/SJTU-H3D.log    