# python eval/main.py \
#     -model InsightMapper_v2_24 \
#     -mode naive \
#     -threshold 0.4 \
#     -merge_threshold 1  \
#     -matching_threshold 0.00005 \
#     -interval 0.0001 \
#     -result_dir ../InsMap_V2/val/work_dirs/insmap_r50_24ep/Mon_Sep__4_00_24_57_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model InsightMapper_v2_110 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir  ../InsMap_V2/val/work_dirs/insmap_r110_24ep/Mon_Sep__4_22_48_04_2023/pts_bbox/nuscmap_results.json