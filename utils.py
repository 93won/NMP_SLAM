from numpy.linalg import pinv
import scipy.stats as stats
import copy
from itertools import product
import numpy as np
from scipy.stats import multivariate_normal
from functools import reduce
import matplotlib.pyplot as plt
import networkx as nx

from multiprocessing import Pool, Array, Process

class Gaussian(object):
    def __init__(self, mean, cov, weight=None):
        self.mean = mean
        self.cov = cov
        self.weight = weight




def invDiag(diag):
    inv = np.array([[1 / diag[0][0], 0, 0],
                    [0, 1 / diag[1][1], 0],
                    [0, 0, 1 / diag[2][2]]])
    return inv

def multDiags(diags):

    d1 = 1
    d2 = 1
    d3 = 1

    for i in range(len(diags)):
        d1 *= diags[i][0][0]
        d2 *= diags[i][1][1]
        d3 *= diags[i][2][2]

    return np.array([[d1, 0, 0], [0, d2, 0], [0, 0, d3]])

def gaussian_product_diag(d1, d2):

    if d1 == None:
        return d2

    elif d2 == None:
        return d1


    else:
        inv_cov1 = invDiag(d1.cov)
        inv_cov2 = invDiag(d2.cov)

        cov_3_inv = (inv_cov1 + inv_cov2)
        cov_3 = invDiag(cov_3_inv)

        d1_vec = t2v(d1.mean)
        d2_vec = t2v(d2.mean)

        if (d1_vec[2] < np.pi/2 and d2_vec[2] > 3/2*np.pi):
            _d2 = t2v(d2.mean)
            _d2[2] = (d2_vec[2] - np.pi*2)
            d2_vec = np.copy(_d2)

        elif (d2_vec[2] < np.pi / 2 and d1_vec[2] > 3 / 2 * np.pi):
            _d1 = t2v(d1.mean)
            _d1[2] = (d1_vec[2] - np.pi * 2)
            d1_vec = np.copy(_d1)
        mu = copy.deepcopy(v2t((cov_3 @ inv_cov1) @ d1_vec + (cov_3 @ inv_cov2) @ d2_vec))

        cov = copy.deepcopy(cov_3)

        cov_test = ((d1.cov) @ invDiag(d1.cov + d2.cov)) @ d2.cov

        return Gaussian(mu, cov)

def getRotation(heading):
    """
    (1) Function
        - Get rotation matrix from heading(radian)
    """
    cos = np.cos(heading)
    sin = np.sin(heading)

    normalize_factor = np.sqrt(cos**2 + sin**2)

    cos /= normalize_factor
    sin /= normalize_factor

    rotation = np.array([[cos, -sin],
                        [sin, cos]])

    return rotation

def v2t(pose):
    # convert pose [x, y, heading] to SE2 format

    """
    (1) Function
        - convert pose [x, y, heading] to SE2 matrix
    (2) Input
        - pose = [x, y, theta]
    (3) Output
        - SE2 matrix
    """

    x, y, heading = pose[0], pose[1], pose[2]

    if heading < 0:
        heading += 2 * np.pi

    if heading > np.pi * 2:
        heading -= np.pi*2

    rotation = getRotation(heading)

    SE2 = np.eye(3)

    SE2[:2, :2] = rotation
    SE2[0, 2] = x
    SE2[1, 2] = y

    return SE2


def t2v(SE2):
    # convert SE2 matrix to pose [x, y, heading]

    """
    (1) Function
        - convert SE2 matrix to pose [x, y, heading]
    (2) Input
        - SE2 matrix
    (3) Output
        - pose = [x, y, theta]
    """

    x, y = SE2[:2, 2]
    cos = np.clip(SE2[0, 0], -1, 1)
    sin = np.clip(SE2[1, 0], -1, 1)
    heading = np.arctan2(sin, cos)


    if heading < 0:
        heading += 2*np.pi

    if heading > np.pi * 2:
        heading -= np.pi*2

    pose = [x, y, heading]

    return pose


def v2tRaw(pose):
    # convert pose [x, y, heading] to SE2 format

    """
    (1) Function
        - convert pose [x, y, heading] to SE2 matrix
    (2) Input
        - pose = [x, y, theta]
    (3) Output
        - SE2 matrix
    """

    x, y, heading = pose[0], pose[1], pose[2]


    rotation = getRotation(heading)

    SE2 = np.eye(3)

    SE2[:2, :2] = rotation
    SE2[0, 2] = x
    SE2[1, 2] = y

    return SE2


def t2vRaw(SE2):
    # convert SE2 matrix to pose [x, y, heading]

    """
    (1) Function
        - convert SE2 matrix to pose [x, y, heading]
    (2) Input
        - SE2 matrix
    (3) Output
        - pose = [x, y, theta]
    """

    x, y = SE2[:2, 2]
    cos = np.clip(SE2[0, 0], -1, 1)
    sin = np.clip(SE2[1, 0], -1, 1)
    heading = np.arctan2(sin, cos)


    pose = [x, y, heading]

    return pose

def exactSampling(_messages, threshold=0.1, remove=True, rejectLoop=True):
    nb_messages = len(_messages)
    messages = copy.deepcopy(_messages)

    if rejectLoop:
        for msg in messages:
            if msg.type == "loop_closure" or msg.type == "triangulation":
                msg.gaussians.append(None)

    if nb_messages > 1:

        lists = []

        for i in range(nb_messages):
            nb_gaussians = len(messages[i].gaussians)
            temp_list = []
            for j in range(nb_gaussians):
                temp_list.append(j)

            lists.append(temp_list)

        combinations = list(product(*lists))

        results = []
        weights = []

        for combination in combinations:
            gaussians = []
            for i in range(len(combination)):
                gaussians.append(messages[i].gaussians[combination[i]])

            gaussian_mul = gaussian_product_diag(gaussians[0], gaussians[1])
            for i in range(2, len(gaussians)):
                gaussian_mul = gaussian_product_diag(gaussian_mul, gaussians[i])

            if gaussian_mul == None:
                continue
            gaussian_mul.mean = np.round(gaussian_mul.mean, 3)
            denominator = stats.norm(t2v((gaussian_mul.mean)), gaussian_mul.cov).pdf(t2v(gaussian_mul.mean))[0][0]
            numerator = 1.0
            for i in range(len(gaussians)):
                if gaussians[i] is not None:
                    numerator *= ((stats.norm(t2v(gaussians[i].mean), gaussians[i].cov).pdf(t2v(gaussian_mul.mean))[0][0]) + 1e-7)

            weight = numerator/(denominator + 1e-7)

            gaussian_mul.weight = weight
            results.append(gaussian_mul)

            weights.append(weight)

        weights /= np.sum(weights)

        max_weight = np.max(weights)
        valid = (weights >= max_weight*(threshold))

        results_new = []
        weights_new = []

        results = np.copy(np.array(results)[valid])
        weights = np.copy(np.array(weights)[valid])

        nb_gaussian_before = len(results)
        if results.shape[0] > 1:
            for i in range(results.shape[0]):
                if len(results_new) >= 1 and remove:
                    diffs = []
                    for j in range(len(results_new)):
                        diff = np.sum(np.sqrt(np.array(t2v(results[i].mean))-np.array(t2v(results_new[j].mean))))
                        diffs.append(diff)

                    if min(diffs) > 1e-3:
                        results_new.append(results[i])
                        weights_new.append(weights[i])
                else:
                    results_new.append(results[i])
                    weights_new.append(weights[i])
        else:
            results_new = results
            weights_new = weights

        nb_gaussian_after = len(results_new)
        print(nb_gaussian_before, " -> reduction -> ", nb_gaussian_after)



        results = copy.deepcopy(np.array(results_new))
        weights = copy.deepcopy(np.array(weights_new))
        weights /= np.sum(weights)
        weights = weights.tolist()
        results = results.tolist()



    else:
        results = messages[0].gaussians
        weights = np.ones(shape=(len(messages[0].gaussians),))/(np.float32(len(messages[0].gaussians)))
        weights = weights.tolist()

    return results, weights


def loadCity10000(max_vertex, max_edge_idx):

    f = open('data/ISAM2_GT_city10000.txt')
    gt_data = f.readlines()
    poses = []
    plt.xlim([-60, 60])
    plt.ylim([-60, 60])
    offset = 0
    for i in range(max_vertex+1):
        x, y, w = np.float32(gt_data[i].split(' '))
        poses.append(np.array([x, y, w]))
        """
        if i % 100 == 0:
            plt.plot(xs[offset:], ys[offset:], color='black')
            plt.savefig('process/{}.png'.format(i), dpi=100)
            offset = i
        """

    f.close()
    print("Load GT Done")
    f2 = open('data/mh_T1_city10000_04.txt')
    edge_mm_data = f2.readlines()[:max_edge_idx]
    f2.close()

    edges = []

    for i in range(len(edge_mm_data)):
        edge_data = edge_mm_data[i].split(' ')
        from_idx = np.int32(edge_data[1])
        to_idx = np.int32(edge_data[3])
        nb_mode = np.int32(edge_data[5])

        edge = [from_idx, to_idx, nb_mode, []]

        offset = 6
        for m in range(nb_mode):
            edge[3].append(np.float32(edge_data[offset:offset+3]))
            offset += 3

        edges.append(edge)


        #print(i, " th ", edge_data[0], " : from ", from_idx, " to ", to_idx, " // nb_mode : ", nb_mode)


    return poses, edges

def readData(nfile, efile):
    """
    (1) Function
        - Read data
    (2) Input
        - nfile : file path of node data
        - efile : file path of edge data
    (3) Output
        - nodes : [[pose_0], [pose_1], ... , [pose_n]]
        - edges : [[[idx of edges observed from nodes[0], e_0, omega_o],
                    ... ,
                   [idx of edges observed from nodes[0], e_k, omega_k]],
                    ... ,
                   [[idx of edges observed from nodes[n], e_0, omega_0],
                   ... ,
                   [idx of edges observed from nodes[n], e_k, omega_k]]]
    """

    nodes = np.loadtxt(nfile, usecols=range(1, 5))[:, 1:]
    edges_aux = np.loadtxt(efile, usecols=range(1, 12))
    idxs_edge_aux = edges_aux[:, :2]
    means_edge_aux = edges_aux[:, 2:5]
    infms_edge_aux = []

    edges = [[]]

    nb_edges = edges_aux.shape[0]


    idx_ref = -1


    for i in range(nb_edges):

        infm = np.zeros((3, 3), dtype=np.float64)
        # edges[i, 5:11] ... upper-triangular block of the information matrix (inverse cov.matrix) in row-major order
        infm[0, 0] = edges_aux[i, 5]
        infm[1, 0] = infm[0, 1] = edges_aux[i, 6]
        infm[1, 1] = edges_aux[i, 7]
        infm[2, 2] = edges_aux[i, 8]
        infm[0, 2] = infm[2, 0] = edges_aux[i, 9]
        infm[1, 2] = infm[2, 1] = edges_aux[i, 10]

        infms_edge_aux.append(infm)

    for i in range(nb_edges):
        idx_max = np.int32(np.max(idxs_edge_aux[i]))
        idx_min = np.int32(np.min(idxs_edge_aux[i]))


        if idx_max != idx_ref:
            idx_ref = idx_max
            edges.append([[idx_min, means_edge_aux[i], infms_edge_aux[i]]])

        else:
            idx_min = np.int32(np.min(idxs_edge_aux[i]))
            edges[idx_ref].append([idx_min, means_edge_aux[i], infms_edge_aux[i]])


    return nodes, edges

def show_data(ax, xs, ys, hs):
    """
        (1) Function
            - Show coordinates of nodes and headings
        (2) Input
            - x-coordinates, y-coordinates, headings
        """

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)

    nb_data = xs.shape[0]

    for i in range(nb_data):
        r = 0.5
        circle = plt.Circle((xs[i], ys[i]), radius=r, color='green')

        ax.add_patch(circle)
        el = 1

        label = ax.annotate("X"+str(i), xy=(xs[i], ys[i]), fontsize=7, ha="center")
        ax.arrow(xs[i] + r*el*np.cos(hs[i]), ys[i] + r*el*np.sin(hs[i]), el*np.cos(hs[i]), el*np.sin(hs[i]),
                 fc="k", ec="k", head_width=0.5, head_length=0.5, width=0.1)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale_view()

    plt.show()


def plotFactorGraph(graph, color='red', save=True, cnt=0, show=False, xs_ref=None, ys_ref=None, ref_color='orange', xs_f=None, ys_f=None, f_color='black', save_file=None):
    xs = []
    ys = []
    for var in graph.variables:
        pose = t2v(var.mean)
        xs.append(pose[0])
        ys.append(pose[1])


    plt.plot(xs, ys, color=color)



    if xs_f is not None:
        plt.plot(xs_f, ys_f, color=f_color)

    if xs_ref is not None:
        plt.plot(xs_ref, ys_ref, color=ref_color)

    if save:
        plt.savefig(save_file + '/{}.png'.format(cnt), dpi=150)


    if show:
        plt.show()

    plt.clf()


from numpy.linalg import *

def norm_pdf_multivariate(x, mu, sigma):
    x = np.array(x)
    mu = np.array(mu)
    sigma = np.array(sigma)
    norm_const = 1


    infm = np.array([[1/(sigma[0][0]**2), 0, 0],
                     [0, 1/(sigma[1][1]**2), 0],
                     [0, 0, 1/(sigma[2][2]**2)]])

    result = np.exp(-0.5 * np.dot(np.dot((x - mu).T, infm), (x - mu)))
    return norm_const * result

def norm_pdf_multivariate_multi(x, mu, sigma, idx, numerater_share):
    x = np.array(x)
    mu = np.array(mu)
    sigma = np.array(sigma)
    norm_const = 1


    infm = np.array([[1/(sigma[0][0]**2), 0, 0],
                     [0, 1/(sigma[1][1]**2), 0],
                     [0, 0, 1/(sigma[2][2]**2)]])

    result = np.exp(-0.5 * np.dot(np.dot((x - mu).T, infm), (x - mu)))

    numerater_share[idx] = result


def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes = [i[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


def draw_Graph(G, node_size=50, font_size=16, fig_size=(14, 4), path=None):
    plt.rcParams["figure.figsize"] = fig_size

    # pos = nx.shell_layout(self.G)
    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1, edge_color='black')

    labels = {}
    for n in G.nodes():
        labels[n] = n
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
    plt.savefig(path, dpi=300)
    plt.axis('off')
    plt.show()


import networkx as nx
import random


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)




def find_root(inputG):
    """
    tree를 입력받아서 diameter를 기준으로 중간의 node를 root로 인식합니다.
    만약 diameter가 균등하게 나누어지지 않을 경우에는 betweenness를 중심으로 root를 찾습니다.
    input: tree
    output:
    """
    return sorted(nx.betweenness_centrality(inputG).items(), key=lambda x: x[1], reverse=True)[0][0]


def find_node_level(inputG):
    """
    input: tree graph
    output: dictionary(key: level, value: node list)
    """
    r_dict = {0: {find_root(inputG)} }
    remain_node_set = {n for n in inputG.nodes()}.difference(r_dict[0])
    current_level = 1
    while len(remain_node_set)>=1:
        new_node_set = set()
        for n in r_dict[current_level-1]:
            new_node_set = new_node_set.union(set(inputG.neighbors(n)))
        new_node_set = new_node_set.intersection(remain_node_set)
        r_dict[current_level] = new_node_set
        remain_node_set = remain_node_set.difference(new_node_set)
        current_level+=1
    return r_dict


def read_data_2d(graph, noise_path, gt_path, nb_loop, prior_cov, threshold=0.1, falseLoop=False):
    gt = []
    cnt = 0
    f = open(noise_path)
    lines = f.readlines()
    for line in lines:

        data = line.split(' ')

        isVertex = (data[0] == 'VERTEX_SE2')
        isEdge = (data[0] == 'EDGE_SE2')

        if isVertex:
            idx_var = np.int32(data[1])
            key_var = 'x'+str(idx_var)
            pose = v2t(np.float32([data[2], data[3], data[4]]))
            graph.addVariable(idx_var, key_var, pose)

            if key_var == 'x0':
                graph.addFactor(idx=len(graph.factors), key='f0', measurements=[pose],
                                var_key_from=None,
                                var_key_to='x0', type='prior', noises=[prior_cov], threshold=threshold)

        elif isEdge:
            idx_from = np.int32(data[1])
            idx_to = np.int32(data[2])
            key_var_from = 'x'+str(idx_from)
            key_var_to = 'x'+str(idx_to)

            if len(data) == 12:
                transform = v2t(np.float32([data[3], data[4], data[5]]))
                cov = pinv(np.float32([[data[6], 0, 0],
                                       [0, data[9], 0],
                                       [0, 0, data[11]]]))

                key_factor = 'f'+str(len(graph.factor_keys))

                isLoop = (idx_to - idx_from > 1)




                if isLoop:

                    graph.addFactor(idx=len(graph.factors), key=key_factor, measurements=[transform],
                                        var_key_from=key_var_from,
                                        var_key_to=key_var_to, type='loop_closure', noises=[cov], threshold=threshold)

                else:
                    graph.addFactor(idx=len(graph.factors), key=key_factor, measurements=[transform],
                                    var_key_from=key_var_from,
                                    var_key_to=key_var_to, type='between', noises=[cov], threshold=threshold)

            else:
                transforms = [v2t(np.float32([data[3], data[4], data[5]])), v2t(np.float32([data[6], data[7], data[8]]))]
                cov = pinv(np.float32([[data[9], 0, 0],
                                        [0, data[12], 0],
                                        [0, 0, data[14]]]))
                covs = [cov, cov]

                key_factor = 'f' + str(len(graph.factor_keys))

                isLoop = (idx_to - idx_from > 1)

                if isLoop:

                    graph.addFactor(idx=len(graph.factors), key=key_factor, measurements=transforms,
                                    var_key_from=key_var_from,
                                    var_key_to=key_var_to, type='loop_closure', noises=covs, threshold=threshold)

                else:
                    graph.addFactor(idx=len(graph.factors), key=key_factor, measurements=transforms,
                                    var_key_from=key_var_from,
                                    var_key_to=key_var_to, type='between', noises=covs, threshold=threshold)
        else:
            continue
    f.close()

    if falseLoop:
        key_var_from = 'x6'
        key_var_to = 'x15'
        key_factor = 'f' + str(len(graph.factors))
        graph.addFactor(idx=len(graph.factors), key=key_factor, measurements=[np.eye(3)],
                        var_key_from=key_var_from,
                        var_key_to=key_var_to, type='loop_closure', noises=[np.eye(3)*0.001], threshold=threshold)

        key_var_from = 'x2'
        key_var_to = 'x60'
        key_factor = 'f' + str(len(graph.factors))
        graph.addFactor(idx=len(graph.factors), key=key_factor, measurements=[np.eye(3)],
                        var_key_from=key_var_from,
                        var_key_to=key_var_to, type='loop_closure', noises=[np.eye(3) * 0.001], threshold=threshold)

    if gt_path is not None:
        f = open(gt_path)

        lines = f.readlines()
        for line in lines:
            data = line.split(' ')
            gt.append(np.float32([data[2], data[3], data[4]]))
    else:
        gt = None

    return graph, gt



from TriangulationAlgorithms import LEX_M

def buildJunctionTree(graph, case='non', threshold=0.1, fake=True):
    G = graph.G.copy()

    print("[1] running triangulation...")

    alex = LEX_M.Algorithm_LexM(G)
    alex.run()

    if case == 'non':
        new_edges = []

    else:
        new_edges = alex.edges_of_triangulation


    print("[2] adding triangulation factors...")


    for e in range(int(len(new_edges) / 2)):


        new_edge = new_edges[e]
        Zs_edge = []
        Covs_edge = []

        path = []  # dfs(G, new_edge[0], new_edge[1])

        isLoop = False

        if not fake:
            index_path = graph.iG.get_shortest_paths(new_edge[0], to=new_edge[1], output='vpath')[0]

            for idx in index_path:
                key = graph.iG.vs[idx]['name']
                path.append(key)


            Zs_path, Covs_path, isLoop = graph.genMeasurement(path)

            for j in range(len(Zs_path)):
                Zs_edge.append(Zs_path[j])
                Covs_edge.append(Covs_path[j])


        factor_key = 'f' + str(len(graph.factor_keys))


        if case == 'triangulation' or case == 'triangulation_with_null' or case == 'junction_tree':
            if isLoop:
                graph.addFactor(idx=len(graph.factors), key=factor_key, measurements=Zs_edge,
                                var_key_from=new_edge[0],
                                var_key_to=new_edge[1], type='triangulation_with_loop', noises=Covs_edge,
                                threshold=threshold, fake=fake)

            else:
                graph.addFactor(idx=len(graph.factors), key=factor_key, measurements=Zs_edge,
                                var_key_from=new_edge[0],
                                var_key_to=new_edge[1], type='triangulation', noises=Covs_edge,
                                threshold=threshold, fake=fake)


    G_tri = copy.deepcopy(graph.G)


    all_cliques = nx.find_cliques(G_tri)

    cliques = []

    while True:
        try:
            # print(next(all_cliques))
            cliques.append(next(all_cliques))
        except:
            break

    # clique graph construction

    print("[3] constructing clique graph...")
    CG = nx.Graph()

    keys = []

    cntt = 0
    for c in cliques:
        key = ''
        cnt = 0
        for key_var in c:
            if cnt == 0:
                key += (key_var)
            else:
                key += (', ' + key_var)
            cnt += 1
        keys.append(key)
        CG.add_node(key)

        cntt += 1
    for c1 in range(len(cliques)):
        for c2 in range(c1 + 1, len(cliques)):
            separators = list(set(cliques[c1]).intersection(cliques[c2]))
            if len(separators) > 0:
                # CG.add_edge('c'+str(c1), 'c'+str(c2))
                CG.add_edges_from([(keys[c1], keys[c2], {
                    'separators': separators})])  # , {'weight':len(separators)})]) #{'separators':separators, 'weight':len(separators)})


    plt.clf()

    nodes = []

    for node1, node2, data in CG.edges(data=True):
        nodes.append(node1)
        #print(data['separators'])

    while True:
        cycles = nx.cycle_basis(CG)
        if len(cycles) == 0:
            break
        cycle = cycles[0]
        # for cycle in cycles:
        weights = []
        for p in range(len(cycle) - 1):
            pair_1 = p
            pair_2 = p + 1
            weights.append(len(CG[cycle[pair_1]][cycle[pair_2]]['separators']))

        idx_min = np.argmin(weights)
        delete_1 = idx_min
        delete_2 = idx_min + 1

        CG.remove_edge(cycle[delete_1], cycle[delete_2])

    root = find_root(CG)


    return graph, CG, root

def dfs(graph, start, end):
    stack = [[start]]
    while stack:
        path = stack.pop()
        if path[-1] == end:

            return path
            #yield path
            #continue
        for next_state in graph[path[-1]]:
            if next_state in path: # Stop cycles
                continue

            if next_state == end:
                path += [next_state]
                return path
            stack.append(path+[next_state])


def search(graph, origin, destination, search_type='bfs'):
    # initialize our path list with the origin node
    path_list = [[origin]]

    # empty lists return false, so the loop will keep running
    # as long as there are paths in the path_list to check
    while path_list:

        # pop out a path on the path list to examine.
        # Examine the first path for Breadth First Search
        # and the last path for Depth First Search

        if search_type == 'bfs':
            pop_index = 0
        if search_type == 'dfs':
            pop_index = -1
        path = path_list.pop(pop_index)

        # if the last node in that path is our destination,
        # we found a correct path
        last_node = path[-1]
        if last_node == destination:
            return path

        # if not, we have to add new paths with all of the
        # neighbors of that last node, as long as those neighbors
        # aren't on the path already
        else:
            for node in graph[last_node]:
                if node not in path:
                    # make a new path ending with the neighbor node
                    new_path = path + [node]
                    # add the new path to the path list
                    path_list.append(new_path)

                    # if the while loop continues without finding a path,
    # then no path exists
    print('No path exists between %s and %s' % (origin, destination))
import time

def sampling(_messages, nb_sample=10, sampling_method="importacne", nullLoop=False, nullTri=False, multi=False):
    nb_messages = len(_messages)
    messages = (_messages)
    dd = []


    for msg in messages:
        dd.append(len(msg.gaussians))
    #print(nb_messages, dd)

    if nullLoop:
        for msg in messages:
            if msg.type == "loop_closure": #or msg.type == 'triangulation':
                msg.gaussians.append(None)

    if nullTri:
        for msg in messages:
            if msg.type == 'triangulation_with_loop':
                msg.gaussians.append(None)


    if nb_messages > 1:
        results = []
        weights = []

        if sampling_method == 'exact':

            lists = []
            for i in range(nb_messages):
                nb_gaussians = len(messages[i].gaussians)
                temp_list = []
                for j in range(nb_gaussians):
                    temp_list.append(j)

                lists.append(temp_list)

            combinations = list(product(*lists))

            if not multi:

                for combination in combinations:

                    gaussians = []
                    for i in range(len(combination)):
                        gaussians.append(messages[i].gaussians[combination[i]])

                    gaussian_mul = gaussian_product_diag(gaussians[0], gaussians[1])

                    for i in range(2, len(gaussians)):
                        gaussian_mul = gaussian_product_diag(gaussian_mul, gaussians[i])

                    if gaussian_mul == None:
                        continue

                    denominator = norm_pdf_multivariate(t2v(gaussian_mul.mean), t2v(gaussian_mul.mean), gaussian_mul.cov)

                    numerator = 1.0


                    mul_mean = t2v(gaussian_mul.mean)

                    for i in range(len(gaussians)):

                        if gaussians[i] is not None:

                            numerator *= norm_pdf_multivariate(mul_mean, t2v(gaussians[i].mean), gaussians[i].cov) + 1e-7


                    weight = numerator / (denominator + 1e-7)
                    gaussian_mul.weight = weight

                    results.append(gaussian_mul)
                    weights.append(weight)


        elif sampling_method == 'importance':

            for c in range(nb_sample):

                gaussians = []

                for i in range(len(messages)):
                    nb_gaussians_message = len(messages[i].gaussians)
                    try:
                        idx = np.random.randint(0, nb_gaussians_message)
                    except:
                        print(nb_gaussians_message)
                    gaussians.append(messages[i].gaussians[idx])

                gaussian_mul = gaussian_product_diag(gaussians[0], gaussians[1])
                for i in range(2, len(gaussians)):
                    gaussian_mul = gaussian_product_diag(gaussian_mul, gaussians[i])

                if gaussian_mul == None:
                    continue

                denominator = norm_pdf_multivariate(t2v(gaussian_mul.mean), t2v(gaussian_mul.mean),
                                                    gaussian_mul.cov)
                numerator = 1.0

                for i in range(len(gaussians)):
                    if gaussians[i] is not None:
                        numerator *= norm_pdf_multivariate(t2v(gaussian_mul.mean), t2v(gaussians[i].mean), gaussians[i].cov) + 1e-7


                weight = numerator / (denominator + 1e-7)
                gaussian_mul.weight = weight
                results.append(gaussian_mul)
                weights.append(weight)



        weights /= np.sum(weights)



        weights = weights.tolist()

        nb_gaussian_after = len(results)
        #print(" NB Gaussian ", nb_gaussian_after)


    else:

        results_aux = messages[0].gaussians

        results = []

        for i in range(len(results_aux)):

            if results_aux[i] is not None:
                results.append(results_aux[i])

        weights = np.ones(shape=(len(results),)) / (np.float32(len(results)))

        weights = weights.tolist()
        results[0].weight = weights[0]

    """
    Mean Shift Moments
    """
    if len(results) >= 2:

        modes = getModes(results, dim=3)

        coords = []


        weights = []

        for mode in modes:
            weights.append(mode.weight)
            #coords.append(t2v(mode.mean))

        """
        plt.clf()
        coords = np.array(coords)
        plt.plot(coords[:, 0], coords[:, 1], 'x', color='red')
        plt.savefig('sample.png', dpi=300)
        plt.clf()
        """
        #print(len(modes), " mode of ", len(results), "mixture of Gaussians")
        results = modes




    return results, weights


MIN_DISTANCE = 3
GROUP_DISTANCE_TOLERANCE = 3


import time

def getModes(_gm, dim=2, max_iter=10):

    t1 = time.time()

    weight_sum = 0

    weights_filter = []

    for g in _gm:
        weight_sum += g.weight
        weights_filter.append(g.weight)

    for g in _gm:
        g.weight /= weight_sum

    weights_filter = np.array(weights_filter)
    max_weight = np.max(weights_filter)
    weight_mask = (weights_filter > 0.1*max_weight)

    weights_filter_aux = weights_filter
    weights_filter_aux[weight_mask] == 0.0
    weights_filter_aux /= np.sum(weights_filter_aux)

    gm = []

    weight_sum = 0

    for i in range(len(_gm)):
        if weight_mask[i]:
            g = _gm[i]
            g.weight = weights_filter_aux[i]
            gm.append(g)


    if(len(gm) == 1):
        return gm
    t2 = time.time()

    nb_gm = len(gm)

    # Calculate weights
    still_shifting = [True] * nb_gm

    for _iter in range(max_iter):

        newGm = []
        # print(still_shifting)

        moving = 0

        for i in range(nb_gm):

            if still_shifting[i]:
                moving += 1

                weights = []

                for j in range(nb_gm):
                    weight_temp = gm[i].weight * norm_pdf_multivariate(t2v(gm[j].mean), t2v(gm[i].mean), gm[i].cov)
                    weights.append(weight_temp)

                weights /= sum(weights)

                # Calculate new modes
                kCovSum = np.zeros(shape=(dim, dim))
                kCovMeanSum = np.zeros(shape=(dim,))
                for j in range(nb_gm):
                    kCov = weights[j] * invDiag(gm[j].cov)  # matrix
                    kCovMean = (kCov @ t2v(gm[j].mean))

                    kCovSum += kCov
                    kCovMeanSum += kCovMean

                mx = (invDiag(kCovSum) @ kCovMeanSum)

                dist = np.sqrt(np.sum(np.square(mx - t2v(gm[i].mean))))

                if dist < MIN_DISTANCE:
                    still_shifting[i] = False

                newGm.append(Gaussian(v2t(mx), gm[i].cov, gm[i].weight))

            else:
                newGm.append(gm[i])

        gm = copy.deepcopy(newGm)

        if moving == 0:
            break

    t3 = time.time()
    # Mode selection
    modes = []
    groups = []
    for i in range(len(gm)):
        if len(modes) == 0:
            modes.append(t2v(gm[i].mean))
            groups.append([i])
        else:
            # Find nearest mode
            min_dist = 1e7
            idx_group = 0
            for m in range(len(modes)):
                dist = np.sqrt(np.sum(np.square(np.array(modes[m]) - np.array(t2v(gm[i].mean)))))
                if dist < min_dist:
                    min_dist = dist
                    idx_group = m

            if min_dist < GROUP_DISTANCE_TOLERANCE:
                groups[idx_group].append(i)

            else:
                modes.append(t2v(gm[i].mean))
                groups.append([i])

    t4 = time.time()
    # New mode and its parameters
    newGm = []
    for i in range(len(groups)):

        mode = modes[i]

        mode_cov = np.zeros(shape=(dim, dim))
        mode_weight = 0
        for j in range(len(groups[i])):
            gm_idx = groups[i][j]
            mean = np.array(t2v(_gm[gm_idx].mean))
            cov = _gm[gm_idx].cov
            weight = _gm[gm_idx].weight

            mode_weight += weight

            mean_dist_sq = np.square(mean - np.array(mode))

            for d in range(dim):
                mode_cov[d][d] += weight * cov[d][d] + mean_dist_sq[d]

        newGm.append(Gaussian(v2t(mode), mode_cov, mode_weight))

    t5 = time.time()

    total = (t5-t1+1e-7)
    #print("DEBUGING TIME :", " ", ((t2-t1)/total), " ", ((t3-t2)/total), " ", ((t4-t3)/total), " ", ((t5-t4)/total))

    #print(len(gm), " -> ", len(newGm))

    return newGm
