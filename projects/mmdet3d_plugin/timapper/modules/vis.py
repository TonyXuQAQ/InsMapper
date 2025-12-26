def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts


def vis_cross_attn(reference_pts,sampling_pts,attn_weight,index=0):
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.patches import Rectangle
    from os import path as osp
    import numpy as np
    import os

    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    
    def denormalize_2d_pts(pts):
        if not len(pts):
            return pts
        new_pts = pts
        new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                                pc_range[0]) + pc_range[0])
        new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                                pc_range[1]) + pc_range[1])
        return new_pts
    
    # import pdb;pdb.set_trace()
    car_img = Image.open('./figs/lidar_car.png')
    colors_plt = ['r', 'b', 'g','pink']
    cls = ['div','ped','bound','center']
    
    if not os.path.isdir(f'./work_dirs/temp/train/MAP_test'):
        os.makedirs(f'./work_dirs/temp/train/MAP_test',exist_ok=True)
    show_dir = osp.join(f'./work_dirs/temp/train/MAP_test')
        # =========== vis gt labels
        
    

    instance_num = 100
    for instance_id in range(instance_num):
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        pred_seg_1 = reference_pts.reshape(-1,100,20,2)[0,instance_id].cpu().detach().numpy()        
        pred_seg_1 = denormalize_2d_pts(pred_seg_1)
        plt.plot(pred_seg_1[:,0],pred_seg_1[:,1],'-',linewidth=1,
            color=colors_plt[1], zorder=1
        )
        import torch
        pts = sampling_pts.reshape(1,100,20,8,1,4,2)[0,instance_id,:,:,0,:].reshape(-1,2).cpu().detach().numpy()
        weight = attn_weight.reshape(1,100,20,8,1,4)[0,instance_id,:,:,0,:].flatten(-2,-1)
        weight = (weight / torch.max(weight,dim=-1)[0][:,None])
        weight = weight.reshape(-1).cpu().detach().numpy()
        pts = denormalize_2d_pts(pts)
        sorted_list = [(w,p) for w, p in sorted(zip(weight,pts),key=lambda x:x[0])]
        pts = np.array([list(x[1]) for x in sorted_list])
        weight = np.array([x[0] for x in sorted_list])
        
        sample_pts_color = [[x,0,(1-x)] for x in weight]

        size = [2 for x in weight]
        
        plt.scatter(pts[:,0],pts[:,1],
            color=sample_pts_color, s=size,
        )

        # save
        # plt.imshow(car_img, extent=[-1.2, 1.2+pc_range[0], -1.5, 1.5],zorder=4)
        map_path = osp.join(show_dir, f'COMPARE_MAP_test_{index}_{instance_id}.jpg')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()
    print(f'vis saved')

def vis_anchor(reference_pts,index=0):
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.patches import Rectangle
    from os import path as osp
    import numpy as np
    import os
    import json

    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    
    def denormalize_2d_pts(pts):
        if not len(pts):
            return pts
        new_pts = pts
        new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                                pc_range[0]) + pc_range[0])
        new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                                pc_range[1]) + pc_range[1])
        return new_pts
    
    # import pdb;pdb.set_trace()
    car_img = Image.open('./figs/lidar_car.png')
    colors_plt = ['r', 'b', 'g','pink']
    cls = ['div','ped','bound','center']
    
    if not os.path.isdir(f'./work_dirs/temp/train/MAP_test'):
        os.makedirs(f'./work_dirs/temp/train/MAP_test',exist_ok=True)
    show_dir = osp.join(f'./work_dirs/temp/train/MAP_test')
        # =========== vis gt labels
        
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')

    
    import torch
    output_save = []
    for layer_idx in range(100):
        pred_seg_1 = reference_pts.reshape(-1,100,20,2)[0,layer_idx].cpu().detach().numpy()
        
        
        pred_seg_1 = denormalize_2d_pts(pred_seg_1)

        output_save.append(pred_seg_1.tolist())
        
        plt.plot(pred_seg_1[:,0],pred_seg_1[:,1],'-',linewidth=1,
         zorder=1
        )

    # save
    map_path = osp.join(show_dir, f'COMPARE_MAP_anchor_{index}.jpg')
    plt.savefig(map_path, bbox_inches='tight', dpi=400)
    plt.close()
    with open(f'./anchor_{index}.json', 'w') as f:
        json.dump(output_save, f)