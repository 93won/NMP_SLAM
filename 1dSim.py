import warnings

warnings.filterwarnings(action='ignore')
from factor_graph import *

if __name__ == '__main__':

    gt = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [-1, 0, 0]])

    meas_cov = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])
    odom_cov = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

    # cases = ['triangulation', 'junction_tree']#, 'triangulation_with_null', 'm-triangulation_with_null']#, 'm-triangulation-3']  #, 'triangulation_with_null', 'm-triangulation', 'm-triangulation_with_null']
    cases = ['junction_tree']

    errors_all = []

    trajs_all = []

    origin_path = 'datasets/1D_Sim/'

    for case in cases:
        errors = []
        loops = []
        graph = None
        print("#################################### CASE " + case + " ###################################")

        if case == 'non' or case == 'triangulation' or case == 'm-triangulation' or case == 'm-triangulation-2' or case == 'm-triangulation-3' or case == 'junction_tree':
            graph = FactorGraph(3, sampling_mode='mode', nullTri=False, nullLoop=False)


        graph.addVariable(0, 'x0', v2t([0, 0, 0]))
        graph.addVariable(1, 'x1', v2t([1, 0, 0]))
        graph.addVariable(2, 'x2', v2t([2, 0, 0]))
        graph.addVariable(3, 'x3', v2t([0.5, 0, 0]))
        graph.addVariable(4, 'x4', v2t([-0.5, 0, 0]))


        graph.addFactor(idx=0, key='f0', measurements=[v2t([-1, 0, 0]), v2t([0, 0, 0]), v2t([1, 0, 0])],
                        var_key_from=None, var_key_to='x0', type='prior', noises=[meas_cov, meas_cov, meas_cov])

        graph.addFactor(idx=1, key='f1', measurements=[v2t([1, 0, 0])],
                        var_key_from='x0', var_key_to='x1', type='between', noises=[odom_cov])


        graph.addFactor(idx=2, key='f2', measurements=[v2t([0, 0, 0]), v2t([1, 0, 0])],
                        var_key_from=None, var_key_to='x1', type='prior', noises=[meas_cov, meas_cov])


        graph.addFactor(idx=3, key='f3', measurements=[v2t([1, 0, 0])],
                        var_key_from='x1', var_key_to='x2', type='between', noises=[odom_cov])

        graph.addFactor(idx=4, key='f4', measurements=[v2t([1, 0, 0])],
                        var_key_from=None, var_key_to='x2', type='prior', noises=[meas_cov])

        graph.addFactor(idx=5, key='f5', measurements=[v2t([-1.5, 0, 0])],
                        var_key_from='x2', var_key_to='x3', type='between', noises=[odom_cov])

        # bi-modal loop closure from x1 to x3
        graph.addFactor(idx=6, key='f6', measurements=[v2t([0, 0, 0]), v2t([-1, 0, 0])],
                        var_key_from='x1', var_key_to='x3', type='loop_closure', noises=[meas_cov, meas_cov])

        # bi-modal loop closure from x0 to x3
        graph.addFactor(idx=7, key='f7', measurements=[v2t([0, 0, 0]), v2t([1, 0, 0])],
                        var_key_from='x0', var_key_to='x3', type='loop_closure', noises=[meas_cov, meas_cov])

        graph.addFactor(idx=8, key='f8', measurements=[v2t([-1, 0, 0])],
                        var_key_from='x3', var_key_to='x4', type='between', noises=[odom_cov])

        # uni-modal loop closure from x0 to x4
        graph.addFactor(idx=9, key='f9', measurements=[v2t([0, 0, 0])],
                        var_key_from='x0', var_key_to='x4', type='loop_closure', noises=[meas_cov])

        graph, CG, root = buildJunctionTree(graph, case, fake=False)

        T = nx.bfs_tree(CG, root)

        if case == 'junction_tree':
            print("[4] constructing junction tree...")

            JT = JunctionTree(T, root)

            JT.buildTree()

            update_order = JT.fromLeafToRoot()
            reverse_order = np.flip(update_order).tolist()

            """
            JT.resetFlags()

            cnt = 0
            for key in update_order:
                JT.updateClique(key)
                JT.plotJT("dataset/1D_Sim/Tree/{}.png".format(cnt))
                cnt += 1

            JT.resetFlags()
            JT.plotJT('dataset/1D_Sim/Tree/reset.png')

            cnt = 0
            reverse_order = np.flip(update_order)
            for key in reverse_order:
                JT.updateClique(key)
                JT.plotJT("dataset/1D_Sim/Tree_Reverse/{}.png".format(cnt))
                cnt += 1

            JT.resetFlags()
            JT.plotJT('dataset/1D_Sim/Tree_Reverse/reset.png')
            """

            if case == 'junction_tree':
                clique_sep_pair_up = {}
                clique_sep_pair_down = {}

                for clique_key in update_order:

                    descendants_sep = JT.getClique(clique_key).descendants_sep
                    seperators = []
                    for seps in descendants_sep:
                        seperators += seps
                    clique_sep_pair_up[clique_key] = seperators

                    graph.buildCliqueFactor(clique_key, copy.deepcopy(seperators), up=True)

                for clique_key in reverse_order:
                    seperators = JT.getClique(clique_key).ancestor_sep
                    clique_sep_pair_down[clique_key] = seperators

                    graph.buildCliqueFactor(clique_key, copy.deepcopy(seperators), up=False)

            try:
                os.mkdir(origin_path + case)

            except:
                pass

                for i in range(100):

                    ng, ws = graph.allGaussians()
                    #print(ng)
                    counts, bins = np.histogram(ws, bins=100)
                    plt.clf()
                    hist = plt.hist(bins[:-1], bins, weights=counts, color='black')
                    plt.xlabel('Weights of Gaussians')
                    plt.ylabel('Counts')
                    plt.savefig('hists/{}.png'.format(i), dpi=300)
                    plt.clf()

                    _time = time.time()
                    if case == 'junction_tree':
                        graph.debug = 0
                        cnt = 0
                        for clique_key in update_order:
                            cnt += 1
                            descendants_sep = clique_sep_pair_up[clique_key]
                            graph.propagateClique(clique_key, 1, up=True)

                        for clique_key in reverse_order:
                            cnt += 1
                            seperators = clique_sep_pair_down[clique_key]
                            graph.propagateClique(clique_key, 1, up=False)

                        # print(graph.debug, len(graph.factors))

                    else:
                        for mm in range(5):
                            graph.propagateAll()

                    __time = time.time()

                    if i % 1 == 0:

                        poses_iter_1 = np.copy(np.array(graph.getAllPoses()))

                        ___time = time.time()
                        graph.updateAll(100)
                        ____time = time.time()

                        if gt is not None:
                            err = np.sqrt(np.sum(np.square(np.abs(poses_iter_1 - gt)[:, 0]))) / gt.shape[0]
                            errors.append(err)

                    if i % 1 == 0:
                        nb_ga = 0
                        for edge in graph.edges:
                            nb_ga += len(edge.message_factor_to_var.gaussians)
                            nb_ga += len(edge.message_var_to_factor.gaussians)

                        print("Iteration ", i, " ", "Propagation ", np.round(__time - _time, 3), " Update ",
                              np.round(____time - ___time, 3), " NB all gaussians ", nb_ga, " Error ", err)
