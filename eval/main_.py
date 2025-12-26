import numpy as np
import math
import sys
import pickle

from numpy import tile
import graph as splfy
import topo as topo
import json
import os
#import TOPORender
import shutil
import argparse
from tqdm import tqdm
import mmcv
from scipy.spatial import distance_matrix

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir)
parser = argparse.ArgumentParser()

parser.add_argument('-graph_gt', default='',action='store', dest='graph_gt', type=str,
                    help='ground truth graph (in xy coordinate)')

parser.add_argument('-graph_prop', default='',action='store', dest='graph_prop', type=str,
                    help='proposed graph (in xy coordinate)')

parser.add_argument('-output', default='', action='store', dest='output', type=str,
                    help="outputfile with '.txt' as suffix")                  

parser.add_argument('-matching_threshold', action='store', dest='matching_threshold', type=float,
                    help='topo marble-hole matching distance ', required =False, default=0.00005)

parser.add_argument('-interval', action='store', dest='topo_interval', type=float,
                    help='topo marble-hole interval ', required =False, default=0.0001)

parser.add_argument('-model', type=str)

parser.add_argument('-threshold', type=float, default=0.5)
parser.add_argument('-merge_threshold', type=float, default=1)
parser.add_argument('-result_dir', type=str)
parser.add_argument('-mode', type=str, default='naive')
parser.add_argument('-intersection_threshold',type=float,default=0.9)

args = parser.parse_args()
print(args)

class Vertex():
    def __init__(self,v,id):
        self.x = v[0]
        self.y = v[1]
        self.id = id
        self.neighbors = []

class Edge():
    def __init__(self,src,dst,id):
        self.src = src
        self.dst = dst
        self.id = id

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertex_num = 0
        self.edge_num = 0
        self.endpoints_num = 0
        self.endpoints = []

    def find_v(self,v_coord):
        if f'{v_coord[0]}_{v_coord[1]}' in self.vertices.keys():
            return self.vertices[f'{v_coord[0]}_{v_coord[1]}']
        return 

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.find_v(v1_coord)
        if v1 is None:
            v1 = Vertex(v1_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v1.x}_{v1.y}'] = v1
        
        v2 = self.find_v(v2_coord)
        if v2 is None:
            v2 = Vertex(v2_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v2.x}_{v2.y}'] = v2

        if v1 not in v2.neighbors:
            v2.neighbors.append(v1)
        if v2 not in v1.neighbors:
            v1.neighbors.append(v2)
        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1



def naive_find_line_connections(lines):
    new_edges = []
    endpoints = []
    for line in lines:
        endpoints.append(line[0])
        endpoints.append(line[-1])
    endpoints = np.array(endpoints)
    dis_matrix = distance_matrix(endpoints, endpoints)
    for i in range(len(endpoints)):
        dis_vec = dis_matrix[i,:].copy()
        dis_vec[i] = np.inf
        if i%2:
            dis_vec[i-1] = np.inf
        else:
            dis_vec[i+1] = np.inf
        min_dis = np.min(dis_vec)
        for j in range(len(endpoints)):
            if dis_vec[j] < args.merge_threshold and i<j:
                new_edges.append(scale_up([endpoints[i],endpoints[j]]))
    return np.array(new_edges)

def distance(p1,p2):
    return np.linalg.norm(p1-p2)

def topo_find_line_connections(lines,line_idxs,intersection_gen_results, intersection_idxs, intersection_cls_idxs):
    
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.patches import Rectangle
    from os import path as osp
    import os
    import numpy as np

    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    
    print(intersection_cls_idxs,len(lines),len(intersection_gen_results))
    # import pdb;pdb.set_trace()
    # car_img = Image.open('./figs/lidar_car.png')
    colors_plt = ['r', 'b', 'g','pink']
    colors_connection = ['orange', 'cyan', 'purple']

    new_edges = []
    total_num_lines = 50
    num_lines = len(lines)
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')

    for line in lines:
        plt.plot(line[:,0],line[:,1],'-',linewidth=1,
                    color=colors_plt[0], zorder=1
                    )
    for ii in range(num_lines-1):
        for jj in range(ii+1,num_lines):
            line1, line2 = lines[ii], lines[jj]
            idx1, idx2 = line_idxs[ii], line_idxs[jj]
            index = (2*total_num_lines-idx1-3)*idx1//2+idx2
            inter_pts = intersection_gen_results[index*2:(index+1)*2]
            segment_idxs = intersection_idxs[index*2:(index+1)*2]
            valid_inter_indexs = [i for i, x in enumerate(inter_pts) if inter_pts[i][-1]>args.intersection_threshold]
            if len(valid_inter_indexs):
                for valid_inter_index in valid_inter_indexs:
                    inter_pt = inter_pts[valid_inter_index,:-1]
                    line1_pt = line1[segment_idxs[valid_inter_index,0]]
                    line2_pt = line2[segment_idxs[valid_inter_index,1]]
                    if distance(inter_pt,line1_pt) < 5:
                        new_edges.append([inter_pt,line1_pt])
                    if distance(inter_pt,line2_pt) < 5:
                        new_edges.append([inter_pt,line2_pt])
                    
                    

                    plt.plot(line1[:,0],line1[:,1],'-',linewidth=1,
                    color=colors_plt[0], zorder=1
                    )
                    plt.plot(line2[:,0],line2[:,1],'-',linewidth=1,
                    color=colors_plt[0], zorder=1
                    )
                    plt.scatter([inter_pt[0]],[inter_pt[1]],color=colors_connection[1],s=2,zorder=2)
                    plt.scatter([line1_pt[0]],[line1_pt[1]],color=colors_connection[-1],s=2,zorder=2)
                    plt.scatter([line2_pt[0]],[line2_pt[1]],color=colors_connection[-1],s=2,zorder=2)
    map_path = osp.join('work_dirs/val', f'COMPARE_MAP_.jpg')
    plt.savefig(map_path, bbox_inches='tight', dpi=400)
    plt.close()
    while 1:pass
    return np.array(new_edges)

def scale_up(line):
    line = np.array(line)
    assert len(line.shape)==2
    new_line = np.zeros_like(line)
    if args.model == 'vectormapnet':
        new_line[:,0] = (line[:,0] + 30) * 400/60
        new_line[:,1] = (line[:,1] + 15)* 400/30
    else:
        new_line[:,0] = (line[:,0] + 15) * 400/30
        new_line[:,1] = (line[:,1] + 30)* 400/60
    return new_line

def gt_linestrings_to_graph(lines):
    graph = Graph()
    lines = np.array(lines)
    if args.model!='hdmapnet':
        lines = lines.reshape(lines.shape[0],-1,2)
    for line in lines:
        new_line = scale_up(line)
        for i in range(len(line)-1):
            segment = [new_line[i],new_line[i+1]]
            graph.add(segment)

    output_graph = {}
    for k, v in graph.vertices.items():
        output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]
    return output_graph

def pred_linestrings_to_graph(lines,segment_idxs,intersection_gen_results, intersection_idxs, intersection_cls_idxs):
    
    graph = Graph()
    for line in lines:
        new_line = scale_up(line)
        for i in range(len(line)-1):
            segment = [new_line[i],new_line[i+1]]
            graph.add(segment)
    
    if args.mode == 'naive':
        new_edges = naive_find_line_connections(lines)
    elif args.mode == 'none':
        new_edges = []
    elif args.mode == 'topo':
        new_edges = topo_find_line_connections(lines,segment_idxs,intersection_gen_results, intersection_idxs, intersection_cls_idxs)
    else:
        raise Exception

    if len(new_edges):
        for edge in new_edges:
            graph.add(edge)

    output_graph = {}
    for k, v in graph.vertices.items():
        output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]
    return output_graph

def vis(gen_results, annotations, connection_gen_results, i):
    # visualization
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.patches import Rectangle
    import os.path as osp
    
    # import pdb;pdb.set_trace()
    # car_img = Image.open('./figs/lidar_car.png')
    colors_plt = ['r', 'b', 'g','pink']
    show_dir = f'eval/vis/{args.model}'
    if not osp.isdir(osp.join(show_dir)):
        os.makedirs(osp.join(show_dir),exist_ok=True)
    pc_range=[-15, -30, -5.0, 15, 30, 3.0]
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0]*2, pc_range[3]*2)
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')

    for line in gen_results[i]:
        if line[-1]<args.threshold:
            continue
        line = line[:-1]
        l = line.reshape(-1,2)
        if args.model == 'vectormapnet':
            plt.plot(l[:,1]+pc_range[0],l[:,0],'-',linewidth=1,
                color=colors_plt[-1], zorder=1
            )
        else:
            plt.plot(l[:,0]+pc_range[0],l[:,1],'-',linewidth=1,
                color=colors_plt[-1], zorder=1
            )

    for line in annotations[i]:
        l = line.reshape(-1,2)
        if args.model == 'vectormapnet':
            plt.plot(l[:,1]+pc_range[3],l[:,0],'-',linewidth=1,
                color=colors_plt[-1], zorder=1
                )
        else:
            plt.plot(l[:,0]+pc_range[3],l[:,1],'-',linewidth=1,
                color=colors_plt[-1], zorder=1
                )

    points = [x[:-1] for x in connection_gen_results[i] if x[-1]>args.intersection_threshold]
    xx = [p[0]+pc_range[0] for p in points]
    yy = [p[1] for p in points]
    plt.scatter(xx,yy,color='cyan',s=2,zorder=2)

    map_path = osp.join(show_dir, f'{i}.jpg')
    plt.savefig(map_path, bbox_inches='tight', dpi=400)
    plt.close()

def process_HDMapNet(pred_vecs,gt_vecs):
    gen_results, annotations = {'centerline':[]}, {'centerline':[]}
    pred_vecs, gt_vecs = pred_vecs['results'], gt_vecs['results']
    for _, vecs in pred_vecs.items():
        pts_list = []
        for vec in vecs:
            if vec['type']==3:
                pts_list.append(vec['pts'])
        gen_results['centerline'].append(np.array(pts_list))
    for _, vecs in gt_vecs.items():
        pts_list = []
        for vec in vecs:
            if vec['type']==3:
                pts_list.append(vec['pts'])
        annotations['centerline'].append(np.array(pts_list))
    gen_results['centerline'] = np.array(gen_results['centerline'])
    annotations['centerline'] = np.array(annotations['centerline'])
    return gen_results, annotations

lat_top_left = 41.0 
lon_top_left = -71.0 
min_lat = 41.0
max_lon = -71.0

if args.model=='vectormapnet':
    gen_results, annotations = mmcv.load(args.result_dir)
elif args.model=='hdmapnet':
    with open(args.result_dir,'r') as jf:
        pred_vecs = json.load(jf) 
    with open(args.result_dir[:-11]+'submission.json','r') as jf:
        gt_vecs = json.load(jf) 
    gen_results, annotations = process_HDMapNet(pred_vecs,gt_vecs)
elif args.mode!='topo':
    try: 
        gen_results, annotations, connection_gen_results, _ = mmcv.load(args.result_dir)
        connection_gen_results = connection_gen_results['intersection']
    except:
        gen_results, annotations, _, _, _ = mmcv.load(args.result_dir)
else:
    gen_results, annotations, gen_segment_idxs, connection_gen_results, connection_cls_idxs,  connection_idxs = mmcv.load(args.result_dir)
    gen_segment_idxs = gen_segment_idxs['centerline']
    connection_gen_results = connection_gen_results['intersection']
    connection_idxs = connection_idxs['intersection']
    connection_cls_idxs = connection_cls_idxs['intersection']

gen_results = gen_results['centerline']
annotations = annotations['centerline']

frame_len = len(gen_results)
prec = 0
recall = 0
f1 = 0
pbar = tqdm(range(frame_len))
for frame_idx in pbar:
    frame_idx = 1
    vis(gen_results,annotations,connection_gen_results,frame_idx)
    lines = gen_results[frame_idx]
    if args.mode == 'topo':
        segment_idxs = gen_segment_idxs[frame_idx]
        segment_idxs = np.array([segment_idxs[ii] for ii, line in enumerate(lines) if line[-1]>=args.threshold])
        intersection_gen_results = connection_gen_results[frame_idx]
        intersection_idxs = connection_idxs[frame_idx]
        intersection_cls_idxs = connection_cls_idxs[frame_idx]
    else:
        segment_idxs, intersection_gen_results, intersection_idxs = None, None, None
    
    if args.model!='hdmapnet':
        lines = np.array([line[:-1] for line in lines if line[-1]>=args.threshold])
        if not lines.shape[0]:
            continue
        lines = lines.reshape(lines.shape[0],-1,2)
        indexs = np.linspace(0,lines.shape[1]-1,20,endpoint=True,dtype=int)
        if not len(lines):
            continue
        lines = lines[:,indexs]
    else:
        lines = np.array(lines)
    
    map_pred = pred_linestrings_to_graph(lines,segment_idxs,intersection_gen_results,intersection_idxs,intersection_cls_idxs)
    map_gt = gt_linestrings_to_graph(annotations[frame_idx])
    
    def xy2latlon(x,y):
        lat = lat_top_left - x * 1.0 / 111111.0
        lon = lon_top_left + (y * 1.0 / 111111.0) / math.cos(math.radians(lat_top_left))

        return lat, lon 


    def create_graph(m):
        global min_lat 
        global max_lon 

        graph = splfy.RoadGraph() 

        nid = 0 
        idmap = {}

        def getid(k, idmap):
            
            if k in idmap :
                return idmap[k]
        
            idmap[k] = nid 
            nid += 1 

            return idmap[k]


        for k, v in m.items():
            n1 = k 

            lat1, lon1 = xy2latlon(n1[0],n1[1])

            if lat1 < min_lat:
                min_lat = lat1 

            if lon1 > max_lon :
                max_lon = lon1 

            for n2 in v:
                lat2, lon2 = xy2latlon(n2[0],n2[1])

                if n1 in idmap:
                    id1 = idmap[n1]
                else:
                    id1 = nid 
                    idmap[n1] = nid 
                    nid = nid + 1

                if n2 in idmap:
                    id2 = idmap[n2]
                else:
                    id2 = nid 
                    idmap[n2] = nid 
                    nid = nid + 1

                graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)
        
        graph.ReverseDirectionLink() 

        for node in graph.nodes.keys():
            graph.nodeScore[node] = 100

        for edge in graph.edges.keys():
            graph.edgeScore[edge] = 100


        return graph 


    graph_gt = create_graph(map_gt)
    graph_prop = create_graph(map_pred)

    # print("load gt/prop graphs")

    region = [min_lat-300 * 1.0/111111.0, lon_top_left-500 * 1.0/111111.0, lat_top_left+300 * 1.0/111111.0, max_lon+500 * 1.0/111111.0]

    graph_gt.region = region
    graph_prop.region = region
    #pickle.dump(RoadGraph, open(sys.argv[8].replace('txt','graph'),"w"))
    # TOPORender.RenderGraphSVG(graph_gt, graph_prop, sys.argv[3].replace('txt','svg'))

    losm = topo.TOPOGenerateStartingPoints(graph_gt, region=region, image="NULL", check = False, direction = False, metaData = None)
    
    lmap = topo.TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold = 0.00010, region=region)

    # propagation distance 
    r = 0.00150 # around 100 meters
    # for spacenet, use a smaller distance
    # if lat_top_left - min_lat < 0.01000:
    #     r = 0.00150 # around 150 meters

    topoResult =  topo.TOPOWithPairs(graph_prop, graph_gt, lmap, losm, r =r, step = args.topo_interval, threshold = args.matching_threshold, one2oneMatching = True, metaData = None)

    # pickle.dump([losm, topoResult, region],  open(args.output.replace('txt','topo.p'),'w'))
    prec += topoResult['prec']
    recall += topoResult['recall']
    f1 += topoResult['f1']
    pbar.set_description(f'Precision: {prec/(frame_idx+1)} || Recall: {recall/(frame_idx+1)} || F1: {f1/(frame_idx+1)}')

prec /= frame_len
recall /= frame_len
f1 /= frame_len
print(f'Precision: {prec} || Recall: {recall} || F1: {f1}')
with open(f'./eval/TOPO_{args.model}_{args.mode}_{args.threshold}_{args.merge_threshold}.json','w') as jf:
    json.dump({'prec':prec,'recall':recall,'f1':f1},jf)







