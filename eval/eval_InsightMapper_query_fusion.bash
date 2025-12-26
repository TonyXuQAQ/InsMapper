python eval/main.py \
    -model InsightMapper_no_query_fusion \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_no_fusion_r50_24e/Wed_Aug__9_16_25_45_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_mean_fusion \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_mean_query_fusion_r50_24e/Thu_Aug_17_18_58_12_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_ffn_fusion \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_ffn_query_fusion_r50_24e/Thu_Aug_17_03_37_19_2023/pts_bbox/nuscmap_results.json