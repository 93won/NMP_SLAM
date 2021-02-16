import copy
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from TriangulationAlgorithms import LEX_M

MIN_DISTANCE = 3
GROUP_DISTANCE_TOLERANCE = 3

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


def find_root(inputG):
    return sorted(nx.betweenness_centrality(inputG).items(), key=lambda x: x[1], reverse=True)[0][0]

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

def sampling(_messages, nb_sample=10, sampling_method="gibbs", nullLoop=False, nullTri=False, multi=False, gibbs_iter=10):
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

        if sampling_method == 'exact' or sampling_method == 'mode':

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


        elif sampling_method == 'gibbs':

            for iter in range(1000):


                # choose a starting label

                idxs = []

                for i in range(len(messages)):
                    nb_gaussians_message = len(messages[i].gaussians)
                    messages[i].weights /= np.sum(messages[i].weights)
                    try:
                        idx = np.random.choice(nb_gaussians_message, 1, p=messages[i].weights)[0]
                    except:
                        debug='on'
                    idxs.append(idx)

                for i in range(len(messages)):

                    gaussian_target = messages[i].gaussians[idxs[i]]

                    if i == 0:
                        gaussian_mul = messages[1].gaussians[idxs[1]]
                        for j in range(2, len(messages)):
                            gaussian_mul = gaussian_product_diag(gaussian_mul, messages[j].gaussians[idxs[j]])

                    else:
                        gaussian_mul = messages[0].gaussians[idxs[0]]
                        for j in range(1, len(messages)):
                            if j is not i:
                                gaussian_mul = gaussian_product_diag(gaussian_mul, messages[j].gaussians[idxs[j]])

                    gaussian_mul_full = gaussian_product_diag(gaussian_mul, gaussian_target)

                    denominator = norm_pdf_multivariate(t2v(gaussian_target.mean),
                                                        t2v(gaussian_mul_full.mean), gaussian_mul_full.cov)

                    numerator = norm_pdf_multivariate(t2v(gaussian_target.mean),
                                                      t2v(gaussian_target.mean), gaussian_target.cov)

                    numerator *= norm_pdf_multivariate(t2v(gaussian_target.mean),
                                                      t2v(gaussian_mul.mean), gaussian_mul.cov)

                    numerator *= messages[i].weights[idxs[i]]

                    numerator += 1e-14

                    messages[i].weights[idxs[i]] = copy.deepcopy((numerator/(denominator + 1e-7)))



                    messages[i].weights /= np.sum(messages[i].weights)

            for c in range(nb_sample):

                gaussians = []

                for i in range(len(messages)):
                    nb_gaussians_message = len(messages[i].gaussians)
                    messages[i].weights /= np.sum(messages[i].weights)

                    try:
                        idx = np.random.choice(nb_gaussians_message, 1, p=messages[i].weights)[0]
                    except:
                        debug='on'
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
    if len(results) >= 2 and sampling_method == 'mode':

        modes = getModes(results, dim=3)

        coords = []


        weights = []

        for mode in modes:
            weights.append(mode.weight)
            coords.append(t2v(mode.mean))
        """
        plt.clf()
        coords = np.array(coords)
        plt.plot(coords[:, 0], coords[:, 1], 'x', color='red')
        plt.savefig('sample.png', dpi=300)
        plt.clf()
        #print(len(modes), " mode of ", len(results), "mixture of Gaussians")
        """
        results = modes




    return results, weights


def getModes(_gm, dim=2, max_iter=10):

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
    nb_gm = len(gm)

    # Calculate weights
    still_shifting = [True] * nb_gm

    for _iter in range(max_iter):

        newGm = []

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

    return newGm


from numpy.linalg import pinv

def read_data_2d(graph, noise_path, gt_path, nb_loop, prior_cov, threshold=0.1, falseLoop=False):
    gt = []
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

