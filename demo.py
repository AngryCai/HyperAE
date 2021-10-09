import sys
sys.path.append('/root/python_codes/')
from HyperGAE import HyperGAE
import numpy as np
from sklearn.preprocessing import normalize, label_binarize


def order_sam_for_diag(x, y):
    x_new = np.zeros(x.shape)
    y_new = np.zeros(y.shape)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        start = stop
    return x_new, y_new


if __name__ == '__main__':
    # load img and gt
    from Toolbox.Preprocessing import Processor
    from sklearn.preprocessing import minmax_scale
    from sklearn.decomposition import PCA
    import time

    root = 'D:\Python\HSI_Files\\'
    # root = '/root/python_codes/HSI_Files/'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    # for nb_comps in range(2, 31, 1):
    # for size in range(5, 31, 2):
    NEIGHBORING_SIZE = 13
    EPOCH = 100
    LEARNING_RATE = 0.0002
    REG_GRAPH = 0.001  # beta
    REG_TASK = 100.  # alpha
    WEIGHT_DECAY = 0.001  # lambda
    SEED = 1333  # random seed
    nb_comps = 4
    VERBOSE_TIME = 10
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    if im_ == 'SalinasA_corrected':
        SEED = 10
        NEIGHBORING_SIZE = 7
        EPOCH = 100
        LEARNING_RATE = 0.0002
        REG_GRAPH = 0.001  # beta
        REG_TASK = 100.  # alpha
        WEIGHT_DECAY = 0.001  # lambda
        VERBOSE_TIME = 10
    n_row, n_column, n_band = img.shape
    img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)

    # perform PCA
    # pca = PCA(n_components=nb_comps)
    # img = pca.fit_transform(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))
    # print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))

    x_patches, y_ = p.get_HSI_patches_rw(img, gt, (NEIGHBORING_SIZE, NEIGHBORING_SIZE))
    x_patches = normalize(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
    print('img shape:', img.shape)
    print('img_patches_nonzero:', x_patches.shape)
    n_samples, n_width, n_height, n_band = x_patches.shape
    y = p.standardize_label(y_)
    x_patches, y = order_sam_for_diag(x_patches, y)
    print('x_patches shape: %s, labels: %s' % (x_patches.shape, np.unique(y)))

    N_CLASSES = np.unique(y).shape[0]  # wuhan : 5  Pavia : 6  Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8

    """
    =======================================
    Clustering
    ======================================
    """
    time_start = time.clock()
    model = HyperGAE('clu', im_, N_CLASSES, lr=LEARNING_RATE, epoch=EPOCH, reg_graph=REG_GRAPH, reg_task_specific=REG_TASK,
                     weight_decay=WEIGHT_DECAY, verb_per_iter=VERBOSE_TIME, random_state=SEED)
    model.train_clustering(x_patches, y)
    run_time = round(time.clock() - time_start, 3)
    print('running time', run_time)

    # """
    # ========================================
    # Semi-supervised learning
    # ========================================
    # """
    # train_idx, test_idx = p.stratified_train_test_index(y, train_size=5)
    # y_train, y_test = y[train_idx], y[test_idx]
    # Y = label_binarize(y, np.unique(y))
    # train_mask = np.arange(x_patches.shape[0])
    # train_mask[train_idx] = True
    # train_mask[test_idx] = False
    # model = HyperGAE('semi', im_, N_CLASSES, lr=LEARNING_RATE, epoch=EPOCH, reg_graph=REG_GRAPH,
    #                  reg_task_specific=REG_TASK,
    #                  weight_decay=WEIGHT_DECAY, verb_per_iter=VERBOSE_TIME, random_state=SEED)
    # model.train_semi(x_patches, Y, train_mask)

