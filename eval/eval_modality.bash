python eval/main.py \
    -model InsightMapper_modality_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_multi_modal_r50_24e/Tue_Sep_12_18_48_25_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model maptr_modality_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/maptr_baseline_multi_modal_r50_24e/Wed_Sep_13_10_05_30_2023/pts_bbox/nuscmap_results.json