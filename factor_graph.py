import numpy as np
from functools import reduce
from scipy.stats import multivariate_normal
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy
from multiprocessing import Process, Queue
import os
from utils import *
import time
import networkx as nx
from itertools import product
import multiprocessing

from multiprocessing import Process, Value, Array

from igraph import Graph as iGraph



class Gaussian(object):
    def __init__(self, mean, cov, weight=None):
        self.mean = mean
        self.cov = cov
        self.weight = weight

class Variable(object):
    def __init__(self, idx, key, initial=None):
        self.idx = idx
        self.key = key
        self.mean = initial
        self.edges = []
        self.neighbors = []
        self.connected_factors = []

class Factor(object):
    def __init__(self, idx, key, measurements, type, noises, var_key_from=None, var_key_to=None):
        self.idx = idx
        self.key = key
        self.measurements = measurements
        self.type = type # prior or between or loop closure
        self.noises = noises # odom, observation, prior
        self.from_edge = None
        self.to_edge = None
        self.isMultimodal = (len(measurements) > 1)
        self.var_key_from = var_key_from
        self.var_key_to = var_key_to

class Edge(object):
    def __init__(self, idx, key_var, key_factor):
        self.idx = idx
        self.key_var = key_var
        self.key_factor = key_factor
        self.message_var_to_factor = None
        self.message_factor_to_var = None

class Message(object):
    def __init__(self, key_factor, key_var, gaussians, weights, type):
        self.key_factor = key_factor
        self.key_var = key_var
        self.gaussians = gaussians
        self.weights = weights
        self.type = type

class FactorGraph():
    def __init__(self, dim, sampling_mode='importance', nb_sample=10, nullTri=False, nullLoop=False):
        self.dim = dim
        self.variables = []
        self.var_key_to_idx = {}
        self.var_keys = []

        self.factors = []
        self.factor_key_to_idx = {}
        self.factor_keys = []

        self.vars_key_to_factor_key = {}

        self.msgs = []
        self.msg_key_to_idx = {}

        self.fix_var_key = []
        self.edges = []
        self.sampling_mode = sampling_mode
        self.nb_sample = nb_sample

        self.G = nx.Graph()
        self.iG = iGraph()

        self.factor_of_vars = {}

        self.nullTri = nullTri
        self.nullLoop = nullLoop

        self.clique_factor_pair_up = {}
        self.clique_factor_pair_down = {}

        self.debug = 0

    def addVariable(self, idx, key, initial=None):
        var = Variable(idx, key, initial)
        self.var_key_to_idx[key] = idx
        self.variables.append(var)
        self.var_keys.append(key)
        self.G.add_node(key, kind='variable')
        self.iG.add_vertex(key)

    def getVariableIdxFromKey(self, key):
        return self.var_key_to_idx[key]

    def getVariableFromKey(self, key):
        return copy.deepcopy(self.variables[np.int32(self.getVariableIdxFromKey(key))])

    def getVariablePoseFromKey(self, key):
        return copy.deepcopy(self.getVariableFromKey(key).mean)

    def getAllPoses(self):
        poses = []
        for i in range(len(self.variables)):
            poses.append(t2v(self.variables[i].mean))

        return poses


    def getFactorFromKey(self, key_factor):
        return self.factors[self.factor_key_to_idx[key_factor]]

    def getFactorTypeFromKey(self, key):
        return copy.deepcopy(self.getFactorFromKey(key).type)

    def addFactor(self, idx, key, measurements, var_key_from=None, var_key_to=None, type=None, noises=None, threshold=0.1, fake=False):

        if fake and type is not 'priro':
            self.G.add_edge(var_key_from, var_key_to)
            self.iG.add_edge(var_key_from, var_key_to)

        else:

            self.factor_keys.append(key)

            if type == 'prior':
                factor = Factor(idx, key, measurements, type, noises, var_key_to=var_key_to)
                self.factor_of_vars[(var_key_to)] = key
                edge_idx = len(self.edges)
                edge = Edge(edge_idx, var_key_to, key)
                self.edges.append(edge)
                factor.to_edge = edge_idx
                var_idx = self.getVariableIdxFromKey(var_key_to)
                self.variables[var_idx].edges.append(edge_idx)
                self.variables[var_idx].connected_factors.append(key)

            elif type == 'between' or type =='loop_closure' or type=='triangulation' or type == 'triangulation_with_loop':
                factor = Factor(idx, key, measurements, type, noises, var_key_from=var_key_from, var_key_to=var_key_to)
                edge_idx = len(self.edges)

                var_idx_from = self.getVariableIdxFromKey(var_key_from)
                self.variables[var_idx_from].edges.append(edge_idx)
                self.variables[var_idx_from].neighbors.append(var_key_to)

                self.variables[var_idx_from].connected_factors.append(key)

                edge = copy.deepcopy(Edge(edge_idx, var_key_from, key))
                self.edges.append(edge)
                factor.from_edge = edge_idx

                edge_idx = len(self.edges)


                var_idx_to = self.getVariableIdxFromKey(var_key_to)
                self.variables[var_idx_to].edges.append(edge_idx)
                self.variables[var_idx_to].neighbors.append(var_key_from)

                self.variables[var_idx_to].connected_factors.append(key)

                edge = copy.deepcopy(Edge(edge_idx, var_key_to, key))
                self.edges.append(edge)
                factor.to_edge = edge_idx

                #print(np.linalg.norm(pinv(noises)), factor.key)

                self.G.add_edge(var_key_from, var_key_to)
                self.iG.add_edge(var_key_from, var_key_to)

                self.factor_of_vars[(var_key_from, var_key_to)] = key
                self.factor_of_vars[(var_key_to, var_key_from)] = key

            self.factors.append(factor)
            self.factor_key_to_idx[key] = idx

            self.propagateMessage(key, threshold=threshold)


    def buildMsgFromVarToFactor(self, key_factor, type):
        factor = copy.deepcopy(self.getFactorFromKey(key_factor))

        if type == 'from':
            edge = factor.from_edge
        elif type == 'to':
            edge = factor.to_edge

        isValid = (edge is not None)
        if isValid:
            key_var = copy.deepcopy(self.edges[edge].key_var)
            message_new = Message(key_factor, key_var, [], [], self.getFactorTypeFromKey(key_factor))

            # pull msgs from neighbor variables of factor

            edges_of_var = self.getVariableFromKey(key_var).edges
            if (len(edges_of_var)) == 1:
                # No other source except target factor -> direct message

                var_pose = copy.deepcopy(self.getVariableFromKey(key_var).mean)
                mean = (var_pose)
                cov = factor.noises[0]
                gaussian = Gaussian(mean, cov, 1.0)
                message_new.gaussians.append(gaussian)
                message_new.weights.append(1.0)

            else:
                # More than one source except target factor -> multiply messages from other factors

                messages = []
                for edge_of_var in edges_of_var:
                    if edge_of_var != edge:
                        messages.append(copy.deepcopy(self.edges[edge_of_var].message_factor_to_var))

                dd = 0

                for msg in messages:
                    dd += len(msg.gaussians)

                #print("NB_MESSAGES :", len(messages))


                gaussians, weights = sampling(messages, nb_sample=self.nb_sample, sampling_method=self.sampling_mode,
                                              nullLoop=self.nullLoop, nullTri=self.nullTri)


                message_new.gaussians += copy.deepcopy(gaussians)
                message_new.weights += copy.deepcopy(weights)

            message_new.weights /= np.sum(message_new.weights)
            self.edges[edge].message_var_to_factor = copy.deepcopy(message_new)

    def buildMsgFromFactorToVar(self, key_factor, type, gibbs_iteration=100):
        factor = copy.deepcopy(self.getFactorFromKey(key_factor))
        zs = factor.measurements

        edge_source = 0
        edge = 0

        if type == 'from':
            edge = factor.from_edge
            edge_source = factor.to_edge
            for i in range(len(zs)):
                zs[i] = copy.deepcopy(pinv(zs[i]))
        elif type == 'to':
            edge = factor.to_edge
            edge_source = factor.from_edge

        isValid = (edge is not None)

        if isValid:
            msg = copy.deepcopy(self.edges[edge_source].message_var_to_factor)
            key_var = self.edges[edge].key_var


            message_new = Message(key_factor, key_var, [], [], self.getFactorTypeFromKey(key_factor))


            cnt = 0
            for z in zs:


                """ Max Product """
                """
                max_idx = np.argmax(msg.weights)

                if msg.gaussians[max_idx] is not None:
                    mean = copy.deepcopy(msg.gaussians[max_idx].mean @ z)
                    cov = (factor.noises[cnt])#+ msg.gaussians[max_idx].cov)
                    gaussian = Gaussian(copy.deepcopy(mean), copy.deepcopy(cov))
                    message_new.gaussians.append(copy.deepcopy(gaussian))
                    message_new.weights.append(msg.weights[max_idx])

                """

                for i in range(len(msg.gaussians)):
                    if msg.gaussians[i] is not None:
                        mean = copy.deepcopy(msg.gaussians[i].mean @ z)
                        cov = (factor.noises[cnt])
                        gaussian = Gaussian(copy.deepcopy(mean), copy.deepcopy(cov), copy.deepcopy(msg.gaussians[i].weight))
                        message_new.gaussians.append(copy.deepcopy(gaussian))
                        message_new.weights.append(msg.weights[i])

            self.edges[edge].message_factor_to_var = copy.deepcopy(message_new)

    def allGaussians(self):
        nb_gaussians = 0
        weights = []
        for edge in self.edges:
            ws_1 = []
            if edge.message_factor_to_var is not None:
                cnt = 0
                for g in edge.message_factor_to_var.gaussians:
                    if g is not None:
                        nb_gaussians += 1
                        ws_1.append(edge.message_factor_to_var.weights[cnt])
                    cnt += 1
                ws_1 /= np.sum(ws_1)

            ws_2 = []
            if edge.message_var_to_factor is not None:
                cnt = 0
                for g in edge.message_var_to_factor.gaussians:
                    if g is not None:
                        nb_gaussians += 1
                        ws_2.append(edge.message_var_to_factor.weights[cnt])
                    cnt += 1

                ws_2 /= np.sum(ws_2)

            for w in ws_1:
                weights.append(w)

            for w in ws_2:
                weights.append(w)


        return nb_gaussians, weights

    def propagateMessage(self, key_factor, threshold=0.1, debug=False):

        factor = copy.deepcopy(self.getFactorFromKey(key_factor))
        isPrior = (factor.type == 'prior')

        if isPrior:
            edge = copy.deepcopy(factor.to_edge)
            key_var = self.edges[edge].key_var
            gaussians = []
            for i in range(len(factor.measurements)):
                gaussians.append(Gaussian(factor.measurements[i], factor.noises[i], 1/len(factor.measurements)))

            weights = np.ones(shape=(len(gaussians),))/np.float32(len(gaussians))
            msg = Message(key_factor, [key_var], gaussians, weights, self.getFactorTypeFromKey(key_factor))
            self.edges[edge].message_factor_to_var = msg
            self.edges[edge].message_var_to_factor = msg
        else:

            # step (1) : build msg from from_var to factor4 is 2 or more.
            self.buildMsgFromVarToFactor(key_factor, 'from')

            # step (2) : build msg from factor   to to_var
            self.buildMsgFromFactorToVar(key_factor, 'to')

            # step (3) : build msg from to_var   to factor
            self.buildMsgFromVarToFactor(key_factor, 'to')

            # step (4) : build msg from factor   to from_var
            self.buildMsgFromFactorToVar(key_factor, 'from')

    def updateVarPose(self, key_var, kde_sample=100, plot=False, title=None, isKDE=False):

        variable = copy.deepcopy(self.getVariableFromKey(key_var))
        messages = []

        for edge in variable.edges:
            messages.append(self.edges[edge].message_factor_to_var)

        gaussians, weights = sampling(messages, nb_sample=self.nb_sample, sampling_method=self.sampling_mode,
                                      nullLoop=self.nullLoop, nullTri=self.nullTri)

        if not isKDE:

            idx_var = self.getVariableIdxFromKey(key_var)
            self.variables[idx_var].mean = gaussians[np.argmax(weights)].mean

        else:

            isMultiModal = len(gaussians) > 1

            if isMultiModal:
                weights /= np.sum(weights)
                gaussian_idx = np.random.choice(len(weights), size=kde_sample, p=weights)

                samples = []
                for i in range(kde_sample):
                    gaussian = gaussians[gaussian_idx[i]]
                    mean = t2v(gaussian.mean)
                    cov = gaussian.cov
                    s = np.random.multivariate_normal(mean, cov, 1).T[:, 0]
                    samples.append(s)
                samples = np.array(samples)
                x = samples[:, 0]
                y = samples[:, 1]
                heading = samples[:, 2]

                values = np.vstack([x, y, heading])
                kernel = stats.gaussian_kde(values)
                xx, yy, hh = np.mgrid[np.min(x) - 1.0:np.max(x) + 1.0:25j,
                             np.min(y) - 1.0:np.max(y) + 1.0:25j,
                             np.min(heading):np.max(heading):25j]

                positions = np.vstack([xx.ravel(), yy.ravel(), hh.ravel()])
                f = np.reshape(kernel(positions).T, xx.shape)


                maxarg = np.unravel_index(f.argmax(), f.shape)
                pose = v2t(np.array([xx[maxarg[0], 0, 0], yy[0, maxarg[1], 0], hh[0, 0, maxarg[2]]]))
                idx_var = self.getVariableIdxFromKey(key_var)
                self.variables[idx_var].mean = pose


            else:
                idx_var = self.getVariableIdxFromKey(key_var)
                self.variables[idx_var].mean = gaussians[0].mean


            if plot:
                x = samples[:, 0]
                values = np.vstack([x])
                kernel = stats.gaussian_kde(values)
                gap = np.max(x) - np.min(x)
                xx = np.mgrid[np.min(x) - gap :np.max(x) + gap:300j]

                positions = np.vstack([xx.ravel()])
                f = np.float64(np.reshape(kernel(positions).T, xx.shape))
                f /= (np.sum(f) + 1e-7)
                plt.clf()
                plt.plot(positions[0], f, color='blue')
                plt.plot(x, np.zeros(shape=x.shape), 'rp', markersize=0.1)
                plt.title(title)
                plt.savefig(title+'.png', dpi=300)


    def propagateThread(self, id, idxs, threshold=0.1):
        for idx in idxs:
            factor = copy.deepcopy(self.factors[idx])
            key = factor.key
            self.propagateMessage(key, threshold)

    def propagateAll(self, threshold=0.1, multi=False, random=False):
        #rand_arr = np.arange(len(self.factors))
        #np.random.shuffle(rand_arr)

        if multi:
            list_idxs = []
            nb_process = 8
            nb_factors = len(self.factors)
            idxs_aux = np.int32(np.arange(0, nb_factors))
            offset = np.int32(nb_factors/nb_process)

            for i in range(nb_process-1):
                list_idxs.append(idxs_aux[offset*i : offset*(i+1)])

            list_idxs.append(idxs_aux[offset*(nb_process-1):])

            procs = []

            for i in range(nb_process):
                proc = Process(target=self.propagateThread, args=(i+1, list_idxs[0], threshold))
                procs.append(proc)

            for i in range(nb_process):
                procs[i].start()

            for i in range(nb_process):
                procs[i].join()


        else:
            #random = False
            rand_arr = np.arange(len(self.factors))
            if random:
                np.random.shuffle(rand_arr)
            for i in range(len(self.factors)):
                factor = copy.deepcopy(self.factors[rand_arr[i]])
                key = factor.key


                self.propagateMessage(key_factor=key, threshold=threshold)

    def updateAll(self, nb_sample=100, threshold=0.1, multi=False, isKDE=False):

        if multi:
            list_idxs = []
            nb_process = 8


            nb_variables = len(self.variables)
            idxs_aux = np.int32(np.arange(0, nb_variables))
            offset = np.int32(nb_variables/nb_process)

            for i in range(nb_process - 1):
                list_idxs.append(idxs_aux[offset * i: offset * (i + 1)])

            procs = []

            for i in range(nb_process):
                proc = Process(target=self.updateThread, args=(i + 1, list_idxs[0], threshold, nb_sample))
                procs.append(proc)

            for i in range(nb_process):
                procs[i].start()

            for i in range(nb_process):
                procs[i].join()



        else:
            for var in self.variables:
                key = var.key
                self.updateVarPose(key, nb_sample, isKDE)

    def printAllMessages(self, msg=False, pose=False):
        if msg:
            for edge in self.edges:
                print("=== Message from factor " + edge.key_factor + " to variable " + str(edge.key_var) + " ===")
                if edge.message_factor_to_var is not None:
                    for i in range(len(edge.message_factor_to_var.gaussians)):
                        estimate = np.round(t2v(edge.message_factor_to_var.gaussians[i].mean), 3)
                        print(str(i) + " : " + "x = " + str(estimate[0]) + " y = " + str(estimate[1]) + " z = " + str(
                            estimate[2]))

                print("=== Message from variable " + str(edge.key_var) + " to factor " + edge.key_factor + " ===")
                if edge.message_var_to_factor is not None:
                    for i in range(len(edge.message_var_to_factor.gaussians)):
                        estimate = np.round(t2v(edge.message_var_to_factor.gaussians[i].mean), 3)
                        print(str(i) + " : " + "x = " + str(estimate[0]) + " y = " + str(estimate[1]) + " z = " + str(
                            estimate[2]))
        if pose:
            for var in self.variables:
                position = np.round(t2v(var.mean), 3)
                print("=== Position of variable " + var.key + " ===")
                print(str(position))

    def plotTrajectory(self, save_file=None, show=False):
        xs = []
        ys = []
        nb_vars = len(self.variables)
        for i in range(nb_vars):
            x, y, w = t2v(self.getVariablePoseFromKey('x' + str(i)))
            xs.append(x)
            ys.append(y)

        plt.clf()

        _min = np.min([np.min(xs), np.min(ys)]) - 5
        _max = np.max([np.max(xs), np.max(ys)]) + 5
        plt.xlim([_min, _max])
        plt.ylim([_min, _max])


        plt.plot(xs, ys, color='black')
        if save_file is not None:
            plt.savefig(save_file, dpi=150)

        if show:
            plt.show()

    def getTrajectory(self):
        xs = []
        ys = []
        nb_vars = len(self.variables)
        for i in range(nb_vars):
            x, y, w = t2v(self.getVariablePoseFromKey('x' + str(i)))
            xs.append(x)
            ys.append(y)

        return xs, ys



    def getNumOfAllMsgs(self):
        result = 0
        for edge in self.edges:
            try:
                result += len(edge.message_var_to_factor.gaussians)
                result += len(edge.message_factor_to_var.gaussians)
            except:
                pass

        return (np.int32(result))

    def genMeasurement(self, path, initial=None):

        zs_all = []
        cov_all = []

        nb_combination = len(path) - 1

        isLoop = False

        lists = []

        Z_all = []
        Cov_all = []

        for i in range(nb_combination):

            key_1 = path[i]
            key_2 = path[i + 1]

            key_factor = self.factor_of_vars[(key_1, key_2)]
            factor = copy.deepcopy(self.getFactorFromKey(key_factor))

            if factor.type == 'loop_closure':
                isLoop = True

            isOrder = (key_1 == factor.var_key_from and key_2 == factor.var_key_to)

            zs = []
            covs = factor.noises
            if isOrder:
                zs = factor.measurements
            else:
                for m in range(len(factor.measurements)):
                    zs.append(pinv(factor.measurements[m]))

            temp_list = np.int32(np.arange(len(factor.measurements))).tolist()

            lists.append(temp_list)
            zs_all.append(zs)
            cov_all.append(covs)

        combinations = list(product(*lists))

        zzz = []

        for combination in combinations:
            Z = np.eye(3)
            Cov = np.zeros(shape=(3, 3))
            for c in range(len(combination)):
                # print(t2v((zs_all[c][combination[c]])))
                Z = np.copy(np.dot(Z, zs_all[c][combination[c]]))
                zzz.append(zs_all[c][combination[c]])
                Cov += cov_all[c][combination[c]]

            Z_all.append(Z)
            Cov_all.append(Cov)

            #print("RESULT : ", t2v(initial @ Z), np.linalg.det(Cov))




        return Z_all, Cov_all, isLoop

    def genMeasurementSimple(self, key1, key2):

        pose1 = self.getVariablePoseFromKey(key1)
        pose2 = self.getVariablePoseFromKey(key2)

        return pinv(pose1) @ pose2

    def buildCliqueFactor(self, clique_key, seperators=[], up=True):
        key_variables = clique_key.split(', ')
        key_propagate_factors = []
        for key in key_variables:
            if key not in seperators:
                key_factors = copy.deepcopy(self.getVariableFromKey(key).connected_factors)
                for key_factor in key_factors:
                    if key_factor not in key_propagate_factors:
                        key_propagate_factors.append(key_factor)

        if up:
            self.clique_factor_pair_up[clique_key] = key_propagate_factors

        else:
            self.clique_factor_pair_down[clique_key] = key_propagate_factors



    def propagateClique(self, clique_key, iter=1, up=True):

        time_1 = time.time()

        if up:
            key_propagate_factors = self.clique_factor_pair_up[clique_key]

        else:
            key_propagate_factors = self.clique_factor_pair_down[clique_key]

        time_2 = time.time()

        for i in range(iter):
            for key_factor in key_propagate_factors:
                self.propagateMessage(key_factor)
                self.debug += 1


        time_3 = time.time()

        #print("Time consumption check : ", (time_3 - time_2), " ", (time_2 - time_1))

class Clique:
    def __init__(self, key, ancestor=None, ancestor_sep=[], descendants=[], descendant_seps=[]):
        self.key = key
        self.ancestor = ancestor
        self.ancestor_sep = ancestor_sep
        self.descendants = descendants
        self.descendants_sep = descendant_seps
        self.updateFlag = False

class JunctionTree:
    def __init__(self, T, root):
        self.root = root
        self.cliques = []
        self.cliques_keys = []
        self.key_idx_pair = {}
        self.leaves = []
        self.T = T

    def addClique(self, key):
        clique = Clique(key)
        self.cliques.append(copy.deepcopy(clique))
        self.cliques_keys.append(key)
        self.key_idx_pair[key] = np.copy(np.int32(len(self.cliques) - 1))

    def getCliqueIdxFromKey(self, key):
        return self.key_idx_pair[key]

    def getClique(self, key):
        return copy.deepcopy(self.cliques[self.getCliqueIdxFromKey(key)])

    def getSeperators(self, key1, key2):
        list1 = key1.split(', ')
        list2 = key2.split(', ')

        return copy.deepcopy(list(set(list1) & set(list2)))

    def isUpdated(self, key):
        return copy.deepcopy(self.getClique(key).updateFlag)

    def raiseFlag(self, key):
        idx = self.getCliqueIdxFromKey(key)
        self.cliques[idx].updateFlag = True

    def lowerFlag(self, key):
        idx = self.getCliqueIdxFromKey(key)
        self.cliques[idx].updateFlag = False

    def resetFlags(self):
        for key in self.cliques_keys:
            self.lowerFlag(key)

    def isLeaf(self, key):
        return len(self.getClique(key).descendants) == 0

    def isReadyToUpdate(self, key):
        descendants = self.getClique(key).descendants

        for descendant in descendants:
            if self.getClique(descendant).updateFlag == False:
                return False

        return True


    def buildTree(self):

        for key in self.T.nodes():
            if self.T.out_degree(key) == 0 and self.T.in_degree(key) == 1:
                self.leaves.append(key)

            # only root case
            if key not in self.cliques_keys:
                self.addClique(key)

            nb_descendants = len(self.T[key])

            clique = self.getClique(key) # copy

            if nb_descendants > 0:

                for key_d in self.T[key]:
                    if key_d not in self.cliques_keys:
                        self.addClique(key_d)

                    if key_d not in clique.descendants:

                        seperators = self.getSeperators(key, key_d)

                        clique.descendants.append(key_d)
                        clique.descendants_sep.append(seperators)

                        clique_d = self.getClique(key_d)
                        clique_d.ancestor = key
                        clique_d.ancestor_sep = seperators


                        self.cliques[self.getCliqueIdxFromKey(key_d)] = copy.deepcopy(clique_d)


                self.cliques[self.getCliqueIdxFromKey(key)] = copy.deepcopy(clique)

    def updateClique(self, key):
        idx_clique = self.getCliqueIdxFromKey(key)
        self.cliques[idx_clique].updateFlag = True

    def fromLeafToRoot(self):

        cnt = 0

        update_order = []

        level_new = copy.deepcopy(self.leaves)
        isRoot = (len(level_new) == 0 and level_new[0] == self.root)

        while not isRoot:
            level = copy.deepcopy(level_new)

            level_new = []

            for clique_key in level:
                key = clique_key
                while True:

                    key_ref = copy.deepcopy(key)
                    isLeaf = self.isLeaf(key)
                    if isLeaf:
                        if key not in level_new:
                            self.updateClique(key)

                            # update debug plot
                            #JT.plotJT('2d_example/Tree/Tree_{}'.format(cnt))
                            #cnt += 1

                            update_order.append(copy.deepcopy(key))
                            if key == self.root:
                                break
                            key = copy.deepcopy(self.getClique(key).ancestor)

                    else:
                        isReady = self.isReadyToUpdate(key)
                        if isReady:
                            if key not in level_new:
                                self.updateClique(key)

                                # update debug plot
                                #JT.plotJT('2d_example/Tree/Tree_{}'.format(cnt))
                                #cnt += 1

                                update_order.append(copy.deepcopy(key))

                                if key == self.root:
                                    break

                                key = copy.deepcopy(self.getClique(key).ancestor)

                        else:
                            if key not in level_new:
                                level_new.append(key)
                            break

                    if key == key_ref:
                        break

            if len(level_new) == 0:
                break

            isRoot = (len(level_new) == 1 and level_new[0] == self.root)

        self.updateClique(self.root)

        # update debug plot
        #JT.plotJT('2d_example/Tree/Tree_{}'.format(cnt))

        cnt += 1

        update_order.append(copy.deepcopy(self.root))

        return update_order

    def plotJT(self, path):
        plt.clf()
        pdot = nx.drawing.nx_pydot.to_pydot(self.T)

        for node in pdot.get_nodes():
            name = node.get_name()[1:-1]

            if self.getClique(name).updateFlag == True:
                node.set_color('red')

        pdot.write_png(path)
        plt.clf()
