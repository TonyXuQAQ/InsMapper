# python eval/main.py \
#     -model maptr_v2-99_24 \
#     -mode naive \
#     -threshold 0.4 \
#     -merge_threshold 1  \
#     -matching_threshold 0.00005 \
#     -interval 0.0001 \
#     -result_dir ./val/work_dirs/maptr_baseline_vovnet_24e/Fri_Sep_15_12_08_11_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model insight_v2-99_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_swin_24e/Thu_Sep_14_22_57_29_2023/pts_bbox/nuscmap_results.json
