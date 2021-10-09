# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2019/12/20 20:36
@ Author  : Yaoming Cai
@ FileName: HyperGAE.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
"""
import os
import sys
import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from Toolbox.Preprocessing import Processor
sys.path.append('/home/caiyaom/python_codes/')
import tensorflow as tf
from hypergraph_utils import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

class HyperGAE:

    def __init__(self, task, image_name, n_clz=None, lr=0.001, epoch=1000, reg_graph=1., reg_task_specific=1., weight_decay=1.,
                 verb_per_iter=None, random_state=None):
        """
        :param task: str, 'dim' for dimentionality reduction, 'clu' for clustering, 'semi' for semi-classification
        :param n_clz: number of class, used in semi and clustering tasks
        :param lr: learning rate
        :param epoch: maximum iterations
        :param reg_graph: graph regularization term coefficient
        :param reg_task_specific: task specific regularization term coefficient, used in semi-classification and clustering tasks
        :param weight_decay: self-representation term regularization coefficient, only used in clustering task
        :param verb_per_iter:
        :param random_state:
        """
        self.image_name = image_name
        self.n_clz = n_clz
        tf.reset_default_graph()
        self.task_name = task
        self.lr = lr
        self.epoch = epoch
        self.reg_graph = reg_graph
        self.reg_task_specific = reg_task_specific
        self.weight_decay = weight_decay
        self.verb_per_iter = verb_per_iter
        if random_state is not None:
            tf.set_random_seed(random_state)
        if not os.path.exists(self.image_name + '-' + self.task_name):
            os.mkdir(self.image_name + '-' + self.task_name)
        self.model_root_dir = self.image_name + '-' + self.task_name
        self.model_path = self.model_root_dir + '/' + self.image_name + '-model'

    def net(self, task_name, x, is_training, n_samples=None, n_clz=None):
        # ============ encoder =================
        latent = self.encoder(x, is_training, 'encoder', reuse=tf.AUTO_REUSE)

        # ============ encoder =================
        x_pred = self.decoder(latent, x.get_shape().as_list()[-1], is_training, 'decoder', reuse=tf.AUTO_REUSE)

        # ============ TASK SPECIFIC =============
        if task_name == 'dim':
            # Z = self.task_specific_branch_dim(latent, 'task-dim', tf.AUTO_REUSE)
            Z, x_pred = self.dim_net_dense(x, is_training)
            return Z, x_pred
        elif task_name == 'clu':
            Z, C, Z_pred = self.task_specific_branch_clustering(latent, n_samples, 'task-clu', tf.AUTO_REUSE)
            return Z, C, Z_pred, x_pred
        elif task_name == 'semi':
            Z, Y_pred = self.task_specific_branch_semi(latent, n_clz, 'task-semi', tf.AUTO_REUSE)
            return Z, Y_pred, x_pred
        else:
            raise Exception('The task is not supported !')

    def task_specific_branch_clustering(self, latent, n_samples, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            Z = tf.layers.flatten(latent)
            C = tf.Variable(1.0e-8 * tf.ones([n_samples, n_samples], tf.float32), name='Coef')
            Z_pred = tf.matmul(C, Z)
            return Z, C, Z_pred

    def task_specific_branch_semi(self, latent, n_clz, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            Z = tf.layers.flatten(latent)
            # C = tf.Variable(1.0e-8 * tf.ones([Z.get_shape().as_list()[-1], n_clz], tf.float32), name='Coef')
            # Y_pred = tf.matmul(Z, C)
            Y_logit = tf.layers.dense(Z, n_clz, use_bias=False)
            return Z, Y_logit

    def task_specific_branch_dim(self, latent, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            Z = tf.layers.flatten(latent)
            return Z

    def dim_net_dense(self, x, is_training):
        x_ = tf.layers.flatten(x)
        n_fea = x_.get_shape().as_list()[-1]
        x_hidden = tf.layers.dense(x_, 1024)
        x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))
        x_hidden = tf.layers.dense(x_hidden, 512)
        x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))
        z = tf.layers.dense(x_hidden, 256)
        z = tf.nn.relu(tf.layers.batch_normalization(z, training=is_training))
        x_hidden = tf.layers.dense(z, 512)
        x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))
        x_pred = tf.layers.dense(x_hidden, n_fea, activation=None)
        x_pred = tf.reshape(x_pred, tf.shape(x))
        return z, x_pred

    def encoder(self, x, is_training, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            """========= Conv 1 ============"""
            x_hidden = tf.layers.conv2d(x, 32, (1, 1), strides=(1, 1), padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.initializers.zeros())
            x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))

            x_hidden = tf.layers.conv2d(x_hidden, 64, (3, 3), strides=(1, 1), padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.initializers.zeros())
            x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))

            x_hidden = tf.layers.conv2d(x_hidden, 64, (3, 3), strides=(1, 1), padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.initializers.zeros())
            x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))

            return x_hidden

    def decoder(self, latent, out_channel, is_training, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            """========= Conv 1 ============"""
            x_hidden = tf.layers.conv2d_transpose(latent, 32, (3, 3), strides=(1, 1), padding='same',
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                  bias_initializer=tf.initializers.zeros())
            x_hidden = tf.nn.relu(tf.layers.batch_normalization(x_hidden, training=is_training))

            """========= Conv output ============"""
            x_hidden = tf.layers.conv2d(x_hidden, out_channel, (1, 1), strides=(1, 1), padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.initializers.zeros())
            return x_hidden

    def loss_clustering(self, x_true, x_pred, z, z_pred, C, G):
        """
        calculate clustering-aware loss
        :param x_true: input X
        :param x_pred: reconstruct X of AE
        :param z: AE code
        :param z_pred: self representation
        :param C: affinity matrix
        :param G: laplacian matrix
        :return:
        """
        # =========== model reconstruction loss ==============
        loss_recon = tf.reduce_mean(tf.losses.mean_squared_error(x_true, x_pred))
        tf.summary.scalar('loss-recon', loss_recon)

        # =========== coefficient L2 loss ==============
        loss_l2 = tf.nn.l2_loss(C)
        tf.summary.scalar('loss-l2', loss_l2)

        # =========== latent reconstruction loss ==============
        loss_recon_latent = tf.reduce_mean(tf.losses.mean_squared_error(z, z_pred))
        tf.summary.scalar('loss-latent', loss_recon_latent)

        # # =========== laplacian loss ==============
        loss_graph = tf.trace(
            tf.matmul(tf.matmul(tf.transpose(z), tf.constant(G, dtype=tf.float32)), z))
        tf.summary.scalar('loss-graph', loss_graph)

        loss = loss_recon + self.reg_task_specific * loss_recon_latent + self.weight_decay * loss_l2 + self.reg_graph * loss_graph
        tf.summary.scalar('loss-total', loss)
        return loss

    def loss_dim(self, x_true, x_pred, z, G):
        # =========== model reconstruction loss ==============
        loss_recon = tf.reduce_mean(tf.losses.mean_squared_error(x_true, x_pred))
        tf.summary.scalar('loss-recon', loss_recon)
        # # =========== laplacian loss ==============
        loss_graph = tf.trace(
            tf.matmul(tf.matmul(tf.transpose(z), tf.constant(G, dtype=tf.float32)), z))
        tf.summary.scalar('loss-graph', loss_graph)
        loss = loss_recon + self.reg_graph * loss_graph
        tf.summary.scalar('loss-total', loss)
        return loss

    def loss_semi(self, x_true, x_pred, z, y_true, y_logit, G, mask):
        # =========== model reconstruction loss ==============
        loss_recon = tf.reduce_mean(tf.losses.mean_squared_error(x_true, x_pred))
        tf.summary.scalar('loss-recon', loss_recon)
        # # =========== laplacian loss ==============
        loss_graph = tf.trace(
            tf.matmul(tf.matmul(tf.transpose(z), tf.constant(G, dtype=tf.float32)), z))
        tf.summary.scalar('loss-graph', loss_graph)
        # =========== softmax ==============
        loss_softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_logit)
        mask = tf.cast(mask, dtype=tf.float32)
        mask_ = mask / tf.reduce_mean(mask)
        loss_softmax *= mask_
        loss_softmax = tf.reduce_mean(loss_softmax)
        # loss_softmax = tf.losses.softmax_cross_entropy(y_true, y_logit)
        tf.summary.scalar('loss-softmax', loss_graph)

        loss = loss_recon + self.reg_graph * loss_graph + self.reg_task_specific * loss_softmax
        return loss

    def init_net(self, task, x, G, n_clz=None, mask=None):
        x_placeholder = tf.placeholder(tf.float32, shape=(None, x.shape[1], x.shape[2], x.shape[3]))
        is_training = tf.placeholder(tf.bool)
        if task == 'dim':
            z, x_pred = self.net(task, x_placeholder, is_training)
            loss_op = self.loss_dim(x_placeholder, x_pred, z, G)
            self.z = z
        elif task == 'clu':
            z, C, z_pred, x_pred = self.net(task, x_placeholder, is_training, x.shape[0])
            loss_op = self.loss_clustering(x_placeholder, x_pred, z, z_pred, C, G)
            self.C = C
        elif task == 'semi':
            if n_clz is None:
                raise Exception('n_clz should be given!')
            y_placeholder = tf.placeholder(tf.float32, shape=(None, n_clz))
            z, y_pred, x_pred = self.net(task, x_placeholder, is_training, n_clz=n_clz)
            loss_op = self.loss_semi(x_placeholder, x_pred, z, y_placeholder, y_pred, G, mask)
            self.y_pred = y_pred
            self.y_placeholder = y_placeholder
        else:
            raise Exception('The task is not supported !')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss_op)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        self.train_op = train_op
        self.x_placeholder = x_placeholder
        self.is_training = is_training
        self.loss_op = loss_op
        self.sess = sess

    def generate_hypergraph(self, x, normalize=True):
        """
        generate hypergraph
        :param normalize:
        :param x:
        :return: hypergraph laplacian matrix
        """
        x = np.reshape(x, (x.shape[0], -1))
        tmp = construct_H_with_KNN(x, K_neigs=10, split_diff_scale=False, is_probH=True, m_prob=1.)
        H = hyperedge_concat(None, tmp)
        G = np.asarray(generate_G_from_H(H))
        I = np.eye(G.shape[0])
        if normalize:
            L = I - G
        else:
            L = G
        return L

    def predict(self, X, task):
        if task == 'dim':
            feed_dict = {self.x_placeholder: X, self.is_training: False}
            y_pre = self.sess.run(self.z, feed_dict=feed_dict)
            return y_pre
        elif task == 'clu':
            loss, Coef = self.sess.run([self.loss_op, self.C], feed_dict={self.x_placeholder: X, self.is_training: False})

            # C = 0.5 * (np.abs(Coef) + np.transpose(np.abs(Coef)))
            # spectral = SpectralClustering(n_clusters=self.n_clz, eigen_solver='arpack', affinity='precomputed',
            #                               assign_labels='discretize')
            # y_pred = spectral.fit_predict(C)
            Coef = self.thrC(Coef, 0.25)
            y_pred, C = self.post_proC(Coef, self.n_clz, 8, 18)
            np.savez(self.model_root_dir + '/Affinity.npz', coef=C)
            return loss, y_pred
        elif task == 'semi':
            feed_dict = {self.x_placeholder: X, self.is_training: False}
            y_pre = self.sess.run(self.y_pred, feed_dict=feed_dict)
            return y_pre


    def cluster_accuracy(self, y_true, y_pre):
        Label1 = np.unique(y_true)
        nClass1 = len(Label1)
        Label2 = np.unique(y_pre)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)
        ca = self.__class_acc__(y_true, y_best)
        return acc, nmi, kappa, ca

    def __class_acc__(self, y_true, y_pre):
        """
        calculate each classes's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca


    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L

    def train_clustering(self, X, y=None):
        print('constructing hypergraph...')
        G = self.generate_hypergraph(X, normalize=True)
        print('construction completed. training clustering network...')
        self.init_net(self.task_name, X, G, self.n_clz)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        saver = tf.train.Saver()
        loss_his = []
        acc_his = {'oa': [], 'nmi': [], 'kappa': [], 'ca': []}  # []
        for step_i in range(self.epoch):
            train_feed_dict = {self.x_placeholder: X, self.is_training: True}
            _, loss, summary = self.sess.run([self.train_op, self.loss_op, merged], feed_dict=train_feed_dict)
            print('epoch %s ==> loss=%s' % (step_i, loss))
            loss_his.append(loss)
            writer.add_summary(summary, step_i)
            # =============== test ==================
            # # print logs after self.verb_per_iter iterations
            if self.verb_per_iter is not None and (step_i + 1) % self.verb_per_iter == 0:
                loss_test, y_pre = self.predict(X, self.task_name)
                acc, nmi, kappa, ca = self.cluster_accuracy(y, y_pre)
                print('epoch %s ==> loss=%s, acc=%s' % (step_i, loss_test, (acc, nmi, kappa)))
                acc_his['oa'].append(acc)
                acc_his['nmi'].append(nmi)
                acc_his['kappa'].append(kappa)
                acc_his['ca'] = ca
                # saver.save(self.sess, self.model_path, write_meta_graph=False)
        np.savez(self.model_root_dir + '/history.npz', loss=loss_his, acc=acc_his)
        # saver.save(self.sess, self.model_path)
        if self.verb_per_iter is not None:
            return acc_his
        else:
            loss_test, y_pre = self.predict(X, self.task_name)
            acc, nmi, kappa, ca = self.cluster_accuracy(y, y_pre)
            return acc, nmi, kappa, ca

    def train_semi(self, X, Y, mask):
        print('constructing hypergraph...')
        G = self.generate_hypergraph(X, normalize=True)
        print('construction completed. training semi-classification network...')
        self.init_net(self.task_name, X, G, self.n_clz, mask)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        saver = tf.train.Saver()
        loss_his = []
        acc_his = {'oa': [], 'aa': [], 'kappa': [], 'ca': []}  # []
        for step_i in range(self.epoch):
            train_feed_dict = {self.x_placeholder: X, self.y_placeholder: Y, self.is_training: True}
            _, loss, summary = self.sess.run([self.train_op, self.loss_op, merged], feed_dict=train_feed_dict)
            print('epoch %s ==> loss=%s' % (step_i, loss))
            loss_his.append(loss)
            writer.add_summary(summary, step_i)
            # =============== test ==================
            # # print logs after self.verb_per_iter iterations
            if self.verb_per_iter is not None and (step_i + 1) % self.verb_per_iter == 0:
                y_pre = self.predict(X, self.task_name)
                p = Processor()
                ca, oa, aa, kappa = p.score(np.argmax(Y[np.nonzero(mask == 0)], axis=1),
                                            np.argmax(y_pre[np.nonzero(mask==0)], axis=1))
                print('epoch %s ==> acc=%s' % (step_i, (oa, aa, kappa)))
                acc_his['oa'].append(oa)
                acc_his['aa'].append(aa)
                acc_his['kappa'].append(kappa)
                acc_his['ca'] = ca
                # saver.save(self.sess, self.model_path, write_meta_graph=False)
        np.savez(self.model_root_dir + '/history.npz', loss=loss_his, acc=acc_his)
        # saver.save(self.sess, self.model_path)
        if self.verb_per_iter is not None:
            return acc_his

    def train_dim(self, X, y=None):
        print('constructing hypergraph...')
        G = self.generate_hypergraph(X, normalize=True)
        print('construction completed. training dimentionality reduction network...')
        self.init_net(self.task_name, X, G)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        saver = tf.train.Saver()
        loss_his = []
        acc_his = []
        for step_i in range(self.epoch):
            train_feed_dict = {self.x_placeholder: X, self.is_training: True}
            _, loss, summary = self.sess.run([self.train_op, self.loss_op, merged], feed_dict=train_feed_dict)
            print('epoch %s ==> loss=%s' % (step_i, loss))
            loss_his.append(loss)
            writer.add_summary(summary, step_i)
            # =============== test ==================
            # # print logs after self.verb_per_iter iterations
            if self.verb_per_iter is not None and (step_i + 1) % self.verb_per_iter == 0:
                z = self.predict(X, self.task_name)
                p = Processor()
                score = self.eval_feature_cv(z, y, times=3, test_size=0.9, random_state=331)
                print('epoch %s ==> acc=%s' % (step_i, (score['knn']['oa'][0], score['svm']['oa'][0])))
                # acc_his['oa'].append(oa)
                # acc_his['aa'].append(aa)
                # acc_his['kappa'].append(kappa)
                # acc_his['ca'] = ca
                acc_his.append(score)
                # saver.save(self.sess, self.model_path, write_meta_graph=False)
        np.savez(self.model_root_dir + '/history.npz', loss=loss_his, acc=acc_his, fea=z)
        # saver.save(self.sess, self.model_path)
        if self.verb_per_iter is not None:
            return acc_his

    def eval_feature_cv(self, X, y, times=3, test_size=0.95, random_state=None):
        print(X.shape)
        # X = normalize(X)
        p = Processor()
        estimator = [KNN(n_neighbors=5), SVC(C=1e6, kernel='rbf')]
        estimator_pre, y_test_all = [[], []], []
        for i in range(times):  # repeat N times K-fold CV
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=random_state, shuffle=True, stratify=y)
            # train_index, test_index = p_Cora.stratified_train_test_index(y, test_size)
            # X_train, X_test = X[train_index], X[test_index]
            # y_train, y_test = y[train_index], y[test_index]
            y_test_all.append(y_test)
            for c in range(len(estimator)):
                estimator[c].fit(X_train, y_train)
                y_pre = estimator[c].predict(X_test)
                estimator_pre[c].append(y_pre)
        # score_Cora = []
        score_dic = {'knn': {'ca': [], 'oa': [], 'aa': [], 'kappa': []},
                     'svm': {'ca': [], 'oa': [], 'aa': [], 'kappa': []}
                     }
        key_ = ['knn', 'svm']
        for z in range(len(estimator)):
            ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=None, verbose=False)
            # score_Cora.append([oa, kappa, aa, ca])
            score_dic[key_[z]]['ca'] = ca
            score_dic[key_[z]]['oa'] = oa
            score_dic[key_[z]]['aa'] = aa
            score_dic[key_[z]]['kappa'] = kappa
        return score_dic
