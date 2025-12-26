python eval/main.py \
    -model InsightMapper_mask_0 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_mask_0_r50_24e/Sun_Aug_20_16_20_34_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_mask_50 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_mask_50_r50_24e/Sun_Aug_20_02_26_25_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_mask_80 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_mask_80_r50_24e/Sat_Aug_19_12_37_21_2023/pts_bbox/nuscmap_results.json