import warnings

warnings.filterwarnings(action='ignore')
from factor_graph import *

if __name__ == '__main__':

    gt = [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [-1, 0, 0]]

    meas_cov = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])
    odom_cov = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

    # cases = ['triangulation', 'junction_tree']#, 'triangulation_with_null', 'm-triangulation_with_null']#, 'm-triangulation-3']  #, 'triangulation_with_null', 'm-triangulation', 'm-triangulation_with_null']
    cases = ['junction_tree']

    errors_all = []

    trajs_all = []

    origin_path = 'datasets/1D_Sim/'

    for case in cases:
        gt = []
        errors = []
        loops = []
        graph = None
        print("#################################### CASE " + case + " ###################################")

        if case == 'non' or case == 'triangulation' or case == 'm-triangulation' or case == 'm-triangulation-2' or case == 'm-triangulation-3' or case == 'junction_tree':
            graph = FactorGraph(3, 1024, nb_sample=64, sampling_mode='exact', nullTri=False, nullLoop=False,
                                multi=False)


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

            # JT.plotJT(origin_path+"tree.png")
            update_order = JT.fromLeafToRoot()
            reverse_order = np.flip(update_order).tolist()

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

