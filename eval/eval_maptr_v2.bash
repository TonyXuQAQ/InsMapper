# python eval/main.py \
#     -model maptr_24 \
#     -mode naive \
#     -threshold 0.4 \
#     -merge_threshold 1  \
#     -matching_threshold 0.00005 \
#     -interval 0.0001 \
#     -result_dir ../InsMap_V2/val/work_dirs/maptr_plus_plus_nusc_r50_24ep/Thu_Aug_31_19_31_49_2023/pts_bbox/nuscmap_results.json

python eval/main.py \
    -model maptr_v2_110 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ../InsMap_V2/val/work_dirs/maptr_plus_plus_nusc_r110_24ep/Mon_Sep__4_18_29_50_2023/pts_bbox/nuscmap_results.json