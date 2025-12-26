python eval/main.py \
    -model InsightMapper_24_no_attn \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_no_ins_self_attn_r50_24e/Sun_Aug_27_02_45_37_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_14_nornal_attn \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_normal_self_attn_ins_self_attn_r50_24e/Fri_Aug_25_23_22_19_2023/pts_bbox/nuscmap_results.json