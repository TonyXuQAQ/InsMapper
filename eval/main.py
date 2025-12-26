import numpy as np
import math
import graph as splfy
import topo as topo
import json
import os
#import TOPORender
import shutil
import argparse
from tqdm import tqdm
from scipy.spatial import distance_matrix
from shapely.geometry import LineString

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
                    help='topo marble-hole matching distance ', required =False, default=0.0001)

parser.add_argument('-interval', action='store', dest='topo_interval', type=float,
                    help='topo marble-hole interval ', required =False, default=0.00005)

parser.add_argument('-model', type=str)

parser.add_argument('-threshold', type=float, default=0.5)
parser.add_argument('-merge_threshold', type=float, default=1)
parser.add_argument('-result_dir', type=str)
parser.add_argument('-mode', type=str, default='naive')

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
    new_line[:,0] = (line[:,0] + 15) * 400/30
    new_line[:,1] = (line[:,1] + 30)* 400/60
    return new_line


def gt_line_sample(line):
    line = list(line)
    line = LineString(line)
    distances = np.linspace(0, line.length, 20)
    sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    return sampled_points

def gt_linestrings_to_graph(vecs):
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
    for k, v in graph.vertices.items():
        output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]
    return output_graph

def pred_linestrings_to_graph(valid_vecs,valid_conns=None):
    
    lines = [x[0] for x in valid_vecs]
    lines = np.array(lines)
    if args.mode == 'naive':
        new_edges = naive_find_line_connections(lines)
    elif args.mode == 'none':
        new_edges = []
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

    output_graph = {}
    for k, v in graph.vertices.items():
        output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]
    return output_graph


   

    

lat_top_left = 41.0 
lon_top_left = -71.0 
min_lat = 41.0
max_lon = -71.0

with open(args.result_dir,'r') as jf:
    data_list = json.load(jf)['results']

if 'av2' in args.model:
    with open(f'../data/av2/sensor/av2_map_anns_val.json','r') as jf:
        gt_list = json.load(jf)['GTs']
else:
    with open(f'../data/nuscenes/nuscenes_map_anns_val.json','r') as jf:
        gt_list = json.load(jf)['GTs']

# vis_connection(data_list)

frame_len = len(gt_list)
prec = 0
recall = 0
f1 = 0
pbar = tqdm(range(frame_len))
for frame_idx in pbar:
    data_frame = data_list[frame_idx]
    vectors = data_frame['vectors']

    vec_len = len(vectors)
    conn_len = vec_len*(vec_len-1)//2+1
    valid_vecs = []
    for vec_idx, vec in enumerate(vectors):
        if vec['type']==3 and vec['confidence_level']>args.threshold:
            valid_vecs.append([vec['pts'],vec_idx])
    
    map_pred = pred_linestrings_to_graph(valid_vecs)
    map_gt = gt_linestrings_to_graph(gt_list[frame_idx])
    
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

    region = [min_lat-1000 * 1.0/111111.0, lon_top_left-1000 * 1.0/111111.0, lat_top_left+1000 * 1.0/111111.0, max_lon+1000 * 1.0/111111.0]
    
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




