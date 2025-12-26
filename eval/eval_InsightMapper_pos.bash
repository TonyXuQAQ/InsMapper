python eval/main.py \
    -model InsightMapper_pos_b \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_pos_B_ins_self_attn_r50_24e/Fri_Aug_18_08_51_18_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_pos_c \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_pos_C_ins_self_attn_r50_24e/Fri_Aug_18_22_48_54_2023/pts_bbox/nuscmap_results.json
