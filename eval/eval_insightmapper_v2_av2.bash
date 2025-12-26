python eval/main.py \
    -model maptr_v2_av2 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ../InsMap_V2/val/work_dirs/maptr_r50_24ep_av2/Wed_Sep_27_09_59_43_2023/pts_bbox/av2map_results.json

python eval/main.py \
    -model insight_v2_av2 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ../InsMap_V2/val/work_dirs/insmap_r50_24ep_av2/Mon_Sep_25_22_10_09_2023/pts_bbox/av2map_results.json
