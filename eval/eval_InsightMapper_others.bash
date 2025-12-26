python eval/main.py \
    -model InsightMapper_24_topo \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_topo_r50_24e/Sun_Aug_27_16_20_34_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_24_mean_class \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_point_level_class_r50_24e/Wed_Aug_23_18_46_53_2023/pts_bbox/nuscmap_results.json
