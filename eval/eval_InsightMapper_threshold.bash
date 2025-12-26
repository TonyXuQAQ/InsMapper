python eval/main.py \
    -model InsightMapper_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_r50_24e/Sun_Aug_13_22_38_09_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 0  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_r50_24e/Sun_Aug_13_22_38_09_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 0.2  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_r50_24e/Sun_Aug_13_22_38_09_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_24 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 0.05  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_r50_24e/Sun_Aug_13_22_38_09_2023/pts_bbox/nuscmap_results.json