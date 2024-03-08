---
layout: project_page
permalink: /

projectname: InsMapper
title: Exploring Inner-instance Information for Vectorized HD Mapping
author1: Zhenhua Xu
author1link: https://tonyxuqaq.github.io/
author2: Kwan-Yee. K. Wong
author2link: https://i.cs.hku.hk/~kykwong/
author3: Hengshuang Zhao
author3link: https://hszhao.github.io/
affiliations:
    the University of Hong Kong
paper: https://arxiv.org/abs/2308.08543
code: https://github.com/TonyXuQAQ/InsMapper/tree/main
---

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Vectorized high-definition (HD) maps contain detailed information about surrounding road elements, which are crucial for various downstream tasks in modern autonomous vehicles, such as motion planning and vehicle control. Recent works attempt to directly detect the vectorized HD map as a point set prediction task, achieving notable detection performance improvements. However, these methods usually overlook and fail to analyze the important inner-instance correlations between predicted points, impeding further advancements. To address this issue, we investigate the utilization of inner-instance information for vectorized high-definition mapping through transformers, and propose a powerful system named <strong>InsMapper</strong>, which effectively harnesses inner-instance information with three exquisite designs, including hybrid query generation, inner-instance query fusion, and inner-instance feature aggregation. The first two modules can better initialize queries for line detection, while the last one refines predicted line instances.  InsMapper is highly adaptable and can be seamlessly modified to align with the most recent HD map detection frameworks. Extensive experimental evaluations are conducted on the challenging NuScenes and Argoverse 2 datasets, where InsMapper surpasses the previous state-of-the-art method, demonstrating its effectiveness and generality.
        </div>
    </div>
</div>

---

## Demo Videos


## Demo Figures
We visualize live demos of InsMapper under various scenarios, with different whether and time. Some scenes may have complicated road intersection areas. Short demos are visualized with 2Hz, while long demos have 4Hz FPS.
#### Demo 1. Short, 2Hz, day time
![Alt Text](img/0.gif)
#### Demo 2. Short, 2Hz, day time
![Alt Text](img/1.gif)
#### Demo 3. Short, 2Hz, day time, complicated road intersections
![Alt Text](img/4.gif)
#### Demo 4. Short, 2Hz, night
![Alt Text](img/7.gif)
#### Demo 5. Long, 4Hz, day time, complicated road intersections
![Alt Text](img/2.gif)
#### Demo 6. Long, 4Hz, day time, complicated road intersections
![Alt Text](img/3.gif)
#### Demo 7. Long, 4Hz, day time, rainy, complicated road intersections
![Alt Text](img/5.gif)
#### Demo 8. Long, 4Hz, night
![Alt Text](img/6.gif)


## Code and data
The code will be released in a later stage. For latest update, please refer to our [github repo](https://github.com/TonyXuQAQ/InsMapper/tree/main). 

## Contact
For any questions, please send email to zhxuv at hku dot hk or open an issue in the [github repo](https://github.com/TonyXuQAQ/InsMapper/tree/main).

## Acknowledgement
We thank these high-quality open-sourced projects 
[MapTR](https://github.com/hustvl/MapTR),
[VectorMapNet](https://tsinghua-mars-lab.github.io/vectormapnet/),
[HDMapNet](https://tsinghua-mars-lab.github.io/HDMapNet/),
[STSU](https://github.com/ybarancan/STSU),
[Sat2Graph](https://github.com/songtaohe/Sat2Graph),
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

## Citation
```
@article{xu2023InsMapper,
  title={InsMapper: A Closer Look at Inner-instance Information for Vectorized High-Definition Mapping},
  author={Xu, Zhenhua and Wong, Kenneth KY and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2308.08543},
  year={2023}
}
```
