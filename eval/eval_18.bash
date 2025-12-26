# python eval/main.py \
#     -model InsightMapper_18 \
#     -mode naive \
#     -threshold 0.4 \
#     -merge_threshold 1  \
#     -matching_threshold 0.00005 \
#     -interval 0.0001 \
#     -result_dir ./val/work_dirs/IVTR_r18_24e/Mon_Sep_11_11_55_44_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model Maptr_18 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/maptr_baseline_r18_24e/Tue_Sep_12_02_03_57_2023/pts_bbox/nuscmap_results.json