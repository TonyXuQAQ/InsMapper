# python eval/main.py \
#     -model maptr_av2 \
#     -mode naive \
#     -threshold 0.4 \
#     -merge_threshold 1  \
#     -matching_threshold 0.00005 \
#     -interval 0.0001 \
#     -result_dir ./val/work_dirs/maptr_baseline_r50_24e_av2/Sun_Sep_24_08_45_07_2023/pts_bbox/av2map_results.json

python eval/main.py \
    -model insight_av2 \
    -mode naive \
    -threshold 0.4 \
    -merge_threshold 1  \
    -matching_threshold 0.00005 \
    -interval 0.0001 \
    -result_dir ./val/work_dirs/IVTR_r50_24e_av2/Sun_Sep_24_00_01_26_2023/pts_bbox/av2map_results.json
