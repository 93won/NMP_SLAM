import warnings
warnings.filterwarnings(action='ignore')
from factor_graph import *
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    threshold = 0.1

    gt = []

    prior_cov = np.array([[1e-2, 0.0, 0.0], [0.0, 1e-2, 0.0], [0.0, 0.0, 1e-2]])

    #cases = ['triangulation', 'junction_tree']#, 'triangulation_with_null', 'm-triangulation_with_null']#, 'm-triangulation-3']  #, 'triangulation_with_null', 'm-triangulation', 'm-triangulation_with_null']
    cases = ['junction_tree']

    errors_all = []

    trajs_all = []

    origin_path = 'datasets/2D_Sim2/'

    for case in cases:
        gt = []
        errors = []
        loops = []
        graph = None
        print("#################################### CASE " + case + " ###################################")

        if case == 'non' or case == 'triangulation' or case == 'm-triangulation' or case == 'm-triangulation-2' or case == 'm-triangulation-3' or case == 'junction_tree':
            graph = FactorGraph(3, nb_sample=16, sampling_mode='gibbs', nullTri=False, nullLoop=False)

        errors = []

        graph, gt = read_data_2d(graph, origin_path+'slam.g2o', origin_path+'gt.g2o', 5, prior_cov, threshold, falseLoop=False)

        if gt is not None:
            gt = np.array(gt)



        if case == 'triangulation' or case == 'm-triangulation' or case == 'm-triangulation-3' or case == 'm-triangulation_with_null' or case == 'triangulation_with_null' or case == 'junction_tree':
            if case == 'triangulation' or case == 'triangulation_with_null' or case == 'junction_tree':
                max_tri_mode = 1
            #elif case == 'm-triangulation' or case == 'm-triangulation_with_null':
            #    max_tri_mode = 2
            #elif case == 'm-triangulation-3':
            #    max_tri_mode = 3

            graph, CG, root = buildJunctionTree(graph, case, threshold, fake=True)

            T = nx.bfs_tree(CG, root)

            # Junction tree belief propagation

            if case == 'junction_tree':
                print("[4] constructing junction tree...")

                JT = JunctionTree(T, root)

                JT.buildTree()

                #JT.plotJT(origin_path+"tree.png")
                update_order = JT.fromLeafToRoot()
                reverse_order = np.flip(update_order).tolist()


            """
            cnt = 0
            for key in update_order:
                JT.updateClique(key)
                JT.plotJT("2d_example/Tree/{}.png".format(cnt))
                cnt += 1
    
    
            JT.resetFlags()
            JT.plotJT('2d_example/Tree/reset.png')
    
    
            pdot.write_png("2d_example/Tree_Reverse/before.png")
    
            cnt = 0
            reverse_order = np.flip(update_order)
            for key in reverse_order:
                JT.updateClique(key)
                JT.plotJT("2d_example/Tree_Reverse/{}.png".format(cnt))
                cnt += 1
    
            JT.resetFlags()
            JT.plotJT('2d_example/Tree_Reverse/reset.png')
            """


        poses_iter_1 = []

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

                #print(graph.debug, len(graph.factors))

            else:
                for mm in range(5):
                    graph.propagateAll(threshold)

            __time = time.time()

            pairs = [] #[(0, 41), (4, 44), (5, 45), (9, 48), (10, 48)]

            if i % 1 == 0:

                poses_iter_1 = np.copy(np.array(graph.getAllPoses()))

                ___time = time.time()
                graph.updateAll(100, threshold=threshold)
                ____time = time.time()
                plt.plot(np.array(gt)[:, 0], np.array(gt)[:, 1], '--', color='black', label='ground truth')
                plt.plot(poses_iter_1[:, 0], poses_iter_1[:, 1], color='blue', markersize=5, label='slam pose')
                plt.legend(loc='lower right')
                #plt.plot(poses_iter_1[:, 0], poses_iter_1[:, 1], 'x', color='green', markersize=15)

                for pair in pairs:
                    plt.plot([poses_iter_1[pair[0]][0], poses_iter_1[pair[1]][0]], [poses_iter_1[pair[0]][1], poses_iter_1[pair[1]][1]], '--', color='gray')


                plt.axis('off')
                plt.xlabel('X')
                plt.ylabel('Y')

                plt.savefig(origin_path+case+'/{}.png'.format(i), dpi=300)
                plt.clf()

                if gt is not None:
                    err = np.sqrt(np.sum(np.square(np.abs(poses_iter_1 - gt)[:, :2]))) / gt.shape[0]
                    errors.append(err)

            if i % 1 == 0:
                nb_ga = 0
                for edge in graph.edges:
                    nb_ga += len(edge.message_factor_to_var.gaussians)
                    nb_ga += len(edge.message_var_to_factor.gaussians)

                print("Iteration ", i, " ", "Propagation ", np.round(__time - _time, 3), " Update ",
                      np.round(____time - ___time, 3), " NB all gaussians ", nb_ga, " Error ", err)

        errors_all.append(errors)



    if gt is not None:

        plt.ylabel('MSE')
        plt.xlabel('Iteration')

        colors = ['red', 'orange', 'purple', 'blue', 'green']

        for i in range(len(cases)):
            plt.plot(errors_all[i], color=colors[i], label=cases[i])


        plt.legend(loc='upper right')
        plt.savefig(origin_path + 'error_plot_case_{}.png'.format(len(cases)), dpi=300)

        plt.clf()


        """
        plt.plot(np.array(gt)[:, 0], np.array(gt)[:, 1], '--', color='black', label='ground truth')
    
        for i in range(len(cases)):
            plt.plot(trajs_all[i][:, 0], trajs_all[i][:, 1], color=colors[i], label=cases[i])
    
        plt.legend(loc='best')
        plt.savefig(origin_path + 'trajectory_plot_case_{}.png'.format(len(cases)), dpi=300)
    
        plt.clf()
        """