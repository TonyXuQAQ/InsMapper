python eval/main.py \
    -model InsightMapper_naive \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_naive_query_generation_r50_24e/Tue_Aug_15_23_47_01_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_instance \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_hierarchy_query_generation_r50_24e/Tue_Aug_15_02_49_29_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_dynamic \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_dynamic_r50_24e/Mon_Aug_21_22_58_04_2023/pts_bbox/nuscmap_results.json