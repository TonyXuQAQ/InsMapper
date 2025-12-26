# [ECCV2024] InsMapper: 
This is the official implementation of paper **InsMapper: Exploring Inner-instance Information for Vectorized HD Mapping** by [Zhenhua Xu](https://tonyxuqaq.github.io/), [Kwan-Yee. K. Wong](https://i.cs.hku.hk/~kykwong/), [Hengshuang Zhao](https://hszhao.github.io/) from the Univerisity of Hong Kong.

[Project page](https://tonyxuqaq.github.io/InsMapper/)

## Update
- Jun/2024: Accepted by ECCV 2024.

- Sep/28/2023: Update the manuscript. More experiments are added, including the comparison with [MapTR-V2](https://github.com/hustvl/MapTR/tree/maptrv2).

- Aug/24/2023: Release more demos and the supplementary document at our [new project page](https://tonyxuqaq.github.io/InsMapper/). Code will be released in a later stage. 

- Aug/17/2023: Release paper on arxiv at [paper](https://arxiv.org/abs/2308.08543).

## Training
```
# find the config script in ./projects/configs/eccv
./train.sh
```

## Testing
```
./test.sh
```

## Demo
We visualize live demos of InsMapper under various scenarios, with different whether and time. Some scenes may have complicated road intersection areas. Short demos are visualized with 2Hz, while long demos have 4Hz FPS. 

For more demos, please visit our new [project page](https://tonyxuqaq.github.io/InsMapper/).

<!-- ### Demo 1. Short, 2Hz, day time
![Alt Text](demos/gif/0.gif)
### Demo 2. Short, 2Hz, day time
![Alt Text](demos/gif/1.gif) -->
### Demo 3. Short, 2Hz, day time, complicated road intersections
![Alt Text](demos/gif/4.gif)
<!-- ### Demo 4. Short, 2Hz, night
![Alt Text](demos/gif/7.gif)
### Demo 5. Long, 4Hz, day time, complicated road intersections
![Alt Text](demos/gif/2.gif)
### Demo 6. Long, 4Hz, day time, complicated road intersections
![Alt Text](demos/gif/3.gif) -->
### Demo 7. Long, 4Hz, day time, rainy, complicated road intersections
![Alt Text](demos/gif/5.gif)
### Demo 8. Long, 4Hz, night
![Alt Text](demos/gif/6.gif)

## Acknowledgement
We thank these high-quality open-sourced projects 
[MapTR](https://github.com/hustvl/MapTR),
[VectorMapNet](https://tsinghua-mars-lab.github.io/vectormapnet/),
[HDMapNet](https://tsinghua-mars-lab.github.io/HDMapNet/),
[STSU](https://github.com/ybarancan/STSU),
[Sat2Graph](https://github.com/songtaohe/Sat2Graph),
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

## Note
This repo is not maintained. The code should work fine but without further testing.

## Citation
```
@inproceedings{xu2024insmapper,
  title={Insmapper: Exploring inner-instance information for vectorized hd mapping},
  author={Xu, Zhenhua and K. Wong, Kwan-Yee and Zhao, Hengshuang},
  booktitle={European Conference on Computer Vision},
  pages={296--312},
  year={2024},
  organization={Springer}
}
```
