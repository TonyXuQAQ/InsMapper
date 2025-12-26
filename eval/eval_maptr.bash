python eval/main.py \
    -model maptr_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/maptr_baseline_r50_24e/Mon_Aug_14_13_08_31_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model maptr_110 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./work_dirs/nuscmap_results_maptr.json