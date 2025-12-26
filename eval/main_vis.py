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
import shapely
from shapely import ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString, Point

from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp

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
    def __init__(self,x,y,id):
        self.x = x
        self.y = y
        self.neighbor_edges = []
        self.id = id
        
    def add_neighbor(self,X):
        if isinstance(X,Edge):
            if X not in self.neighbor_edges:
                self.neighbor_edges.append(X)
    
    def remove_neighbor(self,X):
        if isinstance(X,Edge):
            if X in self.neighbor_edges:
                self.neighbor_edges.remove(X)

class Edge():
    def __init__(self,e,src,dst,id):
        self.e = e
        self.src = src
        self.dst = dst
        self.id = id

    def reverse_geom(self, geom):
        def _reverse(x, y, z=None):
            if z:
                return x[::-1], y[::-1], z[::-1]
            return x[::-1], y[::-1]

        return shapely.ops.transform(_reverse, geom)

    def reverse(self):
        self.src, self.dst = self.dst, self.src
        self.e = self.reverse_geom(self.e)

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = []
        self.edge_counter = 0
        self.v_counter = 0

    def add_v(self,v):
        v = np.array(v)
        if f'{v}' in self.vertices.keys():
            return self.vertices[f'{v}']
        else:
            self.vertices[f'{v}'] = Vertex(v[0],v[1],self.v_counter)
            self.v_counter += 1
            return self.vertices[f'{v}']

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.add_v(v1_coord)
        v2 = self.add_v(v2_coord)
        new_edge = Edge(LineString(edge),v1,v2,self.edge_counter)
        self.edge_counter += 1
        v1.add_neighbor(new_edge)
        v2.add_neighbor(new_edge)
        self.edges.append(new_edge)

    def merge(self,edges):
        e1, e2 = edges
        if e1.id == e2.id:
            return
        if (e1.dst == e2.src and e1.src == e2.dst) or (e1.dst == e2.dst and e1.src == e2.src):
            # circle
            circle = True
            if len(e1.src.neighbor_edges)==2 and len(e1.dst.neighbor_edges)==2:
                if e1.src == e2.src:
                    e1.reverse()
            else:
                if len(e1.src.neighbor_edges)==2:
                    e1.reverse()
                if len(e2.dst.neighbor_edges)==2:
                    e2.reverse()
            new_e = Edge(ops.linemerge(MultiLineString([e1.e,e2.e])),e1.src,e2.dst,self.edge_counter)
            self.edge_counter += 1
            e1.src.remove_neighbor(e1)
            e1.src.add_neighbor(new_e)
            e2.dst.remove_neighbor(e2)
            self.edges.remove(e1)
            self.edges.remove(e2)
            self.edges.append(new_e)
        else:
            # no circle
            if e1.dst == e2.src:
                pass
            elif e1.src == e2.src:
                e1.reverse()
            elif e1.dst == e2.dst:
                e2.reverse()
            elif e1.src == e2.dst:
                e1, e2 = e2, e1
            else:
                raise Exception('Error edge vertices...')
            circle = False
            new_e = Edge(ops.linemerge(MultiLineString([e1.e,e2.e])),e1.src,e2.dst,self.edge_counter)
            self.edge_counter += 1
            e1.src.remove_neighbor(e1)
            e1.src.add_neighbor(new_e)
            e2.dst.remove_neighbor(e2)
            e2.dst.add_neighbor(new_e)
            self.edges.remove(e1)
            self.edges.remove(e2)
            self.edges.append(new_e)

def naive_find_line_connections(lines):
    new_edges = []
    endpoints = []
    for line in lines:
        endpoints.append(line[0])
        endpoints.append(line[-1])
    endpoints = np.array(endpoints)
    if (endpoints.shape[0]):
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
                    new_edges.append([endpoints[i],endpoints[j]])
    return np.array(new_edges)

def distance(p1,p2):
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1-p2)

def scale_up(line):
    line = np.array(line)
    assert len(line.shape)==2
    new_line = np.zeros_like(line)
    new_line[:,0] = (line[:,0] ) #* 400/30
    new_line[:,1] = (line[:,1] )#* 400/60
    return new_line

def topo_find_line_connections(valid_vecs,valid_conns):
    
    # vis
    # visualization
    
    
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

    for i in range(len(valid_vecs)):
        line, _ = valid_vecs[i]
        line = np.array(line)
        plt.plot(line[:,0]+pc_range[0],line[:,1],'-',linewidth=1,
            color=colors_plt[-1], zorder=1
        )
    # for i in range(len(valid_conns)):
    #     conns = valid_conns[i]
    #     for conn in conns:
    #         conn_pt, _ = conn
    #         xx = conn_pt[0]+pc_range[0]
    #         yy = conn_pt[1]
    #         plt.scatter(xx,yy,color='cyan',s=2,zorder=2)
    map_path = osp.join(show_dir, f'{frame_idx}.jpg')
    plt.savefig(map_path, bbox_inches='tight', dpi=400)
    plt.close()

    new_edges = []
    for i in range(len(valid_vecs)-1):
        for j in range(i+1,len(valid_vecs)):
            line1, idx1 = valid_vecs[i]
            line2, idx2 = valid_vecs[j]
            connection_idx = (2*vec_len-idx1-3)*idx1//2+idx2
            
            if len(valid_conns[connection_idx]):
                line1, line2 = np.array(line1), np.array(line2)
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0]*2, pc_range[3]*2)
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')

                
                plt.plot(line1[:,0]+pc_range[0],line1[:,1],'-',linewidth=1,
                    color=colors_plt[-1], zorder=1
                )
                plt.plot(line2[:,0]+pc_range[0],line2[:,1],'-',linewidth=1,
                    color=colors_plt[-1], zorder=1
                )

            
                for inter_i in range(len(valid_conns[connection_idx])):
                    conn_pt = valid_conns[connection_idx][inter_i][0]
                    segment_idx = valid_conns[connection_idx][inter_i][1]

                    p1, p2 = line1[segment_idx[0]], line2[segment_idx[1]]

                    
                    if distance(p1,conn_pt)<10:
                        new_edges.append([p1,conn_pt])
                    if distance(p1,conn_pt)<10:
                        new_edges.append([p2,conn_pt])

                    new_line = np.array([p1,conn_pt,p2])
                    plt.plot(new_line[:,0]+pc_range[0],new_line[:,1],'-',linewidth=1,
                        color='orange', zorder=1
                    )

                    xx = conn_pt[0]+pc_range[0]
                    yy = conn_pt[1]
                    
                    plt.scatter(xx,yy,color='cyan',s=2,zorder=2)

                map_path = osp.join(show_dir, f'{frame_idx}_{i}_{j}.jpg')
                plt.savefig(map_path, bbox_inches='tight', dpi=400)
                plt.close()
    return np.array(new_edges)

def gt_line_sample(line):
    line = list(line)
    line = LineString(line)
    distances = np.linspace(0, line.length, 20)
    sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    return sampled_points


def vis_graph(lines, train=False,frame_idx=0):
    plt.figure(figsize=(2, 4))
    colors_plt = ['r', 'b', 'g','pink']
    show_dir = f'eval/vis/{args.model}'
    if not osp.isdir(osp.join(show_dir)):
        os.makedirs(osp.join(show_dir),exist_ok=True)
    pc_range=[-15, -30, -5.0, 15, 30, 3.0]
    plt.xlim(pc_range[0]*2, pc_range[3]*2)
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')

    for ii, line in enumerate(lines):
        line = np.array(line.coords)
        plt.plot(line[:,0]+pc_range[0],line[:,1],'-',linewidth=1,
            zorder=1
        )
    map_path = osp.join(show_dir, f'{frame_idx}_{("train" if train else "pred")}.jpg')
    plt.savefig(map_path, bbox_inches='tight', dpi=400)
    plt.close()


def vis_instance_graph(lines, train=False,frame_idx=0):
    plt.figure(figsize=(2, 4))
    colors_plt = ['r', 'b', 'g','pink']
    show_dir = f'eval/vis/{args.model}'
    if not osp.isdir(osp.join(show_dir)):
        os.makedirs(osp.join(show_dir),exist_ok=True)
    pc_range=[-15, -30, -5.0, 15, 30, 3.0]
    

    for ii, line in enumerate(lines):
        plt.xlim(pc_range[0]*2, pc_range[3]*2)
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        line = np.array(line.coords)
        plt.plot(line[:,0]+pc_range[0],line[:,1],'-',linewidth=1,
            zorder=1
        )
        map_path = osp.join(show_dir, f'{frame_idx}_{ii}_{("train" if train else "pred")}.jpg')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()

def gt_linestrings_to_graph(vecs,frame_idx):
    vecs = vecs['vectors']
    graph = Graph()
    lines = []
    for vec in vecs:
        if vec['type']==3:
            lines.append(gt_line_sample(vec['pts']))
    lines = np.array(lines)
    lines = lines.reshape(lines.shape[0],-1,2)
    for line in lines:
        new_line = scale_up(line)
        for i in range(len(line)-1):
            segment = [new_line[i],new_line[i+1]]
            graph.add(segment)

    output_graph = {}
    # for k, v in graph.vertices.items():
    #     output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

    # ========== vis
    line_list = []
    for _, v in graph.vertices.items():
        if len(v.neighbor_edges)==2:
            graph.merge(v.neighbor_edges)
    for ii, edge in enumerate(graph.edges):
        line_list.append(edge.e)

    
    # vis_instance_graph(line_list,train=True,frame_idx=frame_idx)
    vis_graph(line_list,train=True,frame_idx=frame_idx)

    return output_graph

def pred_linestrings_to_graph(valid_vecs,valid_conns=None,frame_idx=0):
    
    lines = [x[0] for x in valid_vecs]
    lines = np.array(lines)
    if args.mode == 'naive':
        new_edges = naive_find_line_connections(lines)
    elif args.mode == 'none':
        new_edges = []
    elif args.mode == 'topo':
        new_edges = topo_find_line_connections(valid_vecs,valid_conns)
    else:
        raise Exception

    graph = Graph()
    for line in lines:
        new_line = scale_up(line)
        for i in range(len(line)-1):
            segment = [new_line[i],new_line[i+1]]
            graph.add(segment)
    
    if len(new_edges):
        for edge in new_edges:
            edge = scale_up(edge)
            graph.add(edge)
    
    # ========== vis
    line_list = []
    for _, v in graph.vertices.items():
        if len(v.neighbor_edges)==2:
            graph.merge(v.neighbor_edges)
    for ii, edge in enumerate(graph.edges):
        line_list.append(edge.e)

    
    # vis_instance_graph(line_list,train=False,frame_idx=frame_idx)
    vis_graph(line_list,train=False,frame_idx=frame_idx)

    output_graph = {}
    # for k, v in graph.vertices.items():
    #     output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]
    return output_graph


   

    

lat_top_left = 41.0 
lon_top_left = -71.0 
min_lat = 41.0
max_lon = -71.0

with open(args.result_dir,'r') as jf:
    data_list = json.load(jf)['results']
with open(f'../data/nuscenes/nuscenes_map_anns_val.json','r') as jf:
    gt_list = json.load(jf)['GTs']

# vis_connection(data_list)

frame_len = len(gt_list)
prec = 0
recall = 0
f1 = 0
pbar = tqdm(range(frame_len))
for frame_idx in pbar:
    frame_idx += 2006
    data_frame = data_list[frame_idx]
    vectors = data_frame['vectors']
    connections = data_frame['connections']

    vec_len = len(vectors)
    conn_len = vec_len*(vec_len-1)//2+1
    valid_vecs = []
    for vec_idx, vec in enumerate(vectors):
        if vec['type']==3 and vec['confidence_level']>args.threshold:
            valid_vecs.append([vec['pts'],vec_idx])
    
    map_pred = pred_linestrings_to_graph(valid_vecs,frame_idx=frame_idx)
        
    map_gt = gt_linestrings_to_graph(gt_list[frame_idx],frame_idx)
 
    pbar.set_description()

    # break
# prec /= frame_len
# recall /= frame_len
# f1 /= frame_len
print(f'Precision: {prec} || Recall: {recall} || F1: {f1}')
with open(f'./eval/TOPO_{args.model}_{args.mode}_{args.threshold}_{args.merge_threshold}.json','w') as jf:
    json.dump({'prec':prec,'recall':recall,'f1':f1},jf)




