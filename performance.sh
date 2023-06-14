python -u performance.py \
    --database H3D \
    --saqm_csv_path ./output_csvs/H3D_SAQM.csv \
    --snqm_csv_path ./output_csvs/H3D_SNQM.csv \
    --glqm_csv_path ./output_csvs/H3D_GLQM.csv \
    --info_path ./datainfo/H3D_datainfo.csv  \
    --selected_projections 1,1,1,1,1,1 \
    --num_projection 6 \
    --total_num 1120 \
    ## enable SVR supervised training
    #--supervised True \


## DHHQA database contains only front and left projections
python -u performance.py \
    --database DHH \
    --saqm_csv_path ./output_csvs/DHH_SAQM.csv \
    --snqm_csv_path ./output_csvs/DHH_SNQM.csv \
    --glqm_csv_path ./output_csvs/DHH_GLQM.csv \
    --selected_projections 1,1 \
    --num_projection 2 \
    --info_path ./datainfo/DHH_datainfo.csv \
    --total_num 1540 \
    #--mos_scale 100 \
    #--supervised True \