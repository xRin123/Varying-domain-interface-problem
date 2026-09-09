import learner as ln
import torch


class Poisson_varying_domains_data(ln.data.Data_MIONet_Cartesian):
    '''Data for 2d Poisson equation defined on varying domains.
    '''

    def __init__(self, path):
        super(Poisson_varying_domains_data, self).__init__()
        import numpy as np
        X_train, X_test = np.load(path + '/X_train.npz'), np.load(path + '/X_test.npz')
        self.X_train = (X_train['arr_0'], X_train['arr_1'], X_train['arr_2'], X_train['arr_3'])
        self.y_train = np.load(path + '/y_train.npy')
        self.X_test = (X_test['arr_0'], X_test['arr_1'], X_test['arr_2'], X_test['arr_3'])
        self.y_test = np.load(path + '/y_test.npy')


def postprocessing(data, net, loss_history):
    #### post processing, for example, plot a figure if needed.
    pass


def main():
    #### device
    device = 'gpu'  # 'cpu' or 'gpu'
    #### data
    path = '../data/polar_line'  # the directory of the dataset
    #### MIONet
    sizes = [
        [200] + [500] * 3 + [1000],
        [5000] + [500] * 3 + [1000],
        [5000] + [500] * 3 + [1000],
        [2] + [500] * 3 + [1000],
    ]
    activation = 'relu'
    #### training
    lr = 1e-5
    iterations = 100000
    batch_size = None
    print_every = 1

    training_args = {
        'criterion': 'MSE',
        'optimizer': 'Adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': 'best_only',
        'callback': None,
        'dtype': 'double',
        'device': device,
    }

    ln.Brain.Start()
    data = Poisson_varying_domains_data(path)
    net = ln.nn.MIONet_Cartesian(sizes, activation, bias=False)
    ln.Brain.Init(data, net)
    ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    ln.Brain.Output(data=False)
    postprocessing(data, ln.Brain.Best_model(), ln.Brain.Loss_history())
    ln.Brain.End()


if __name__ == '__main__':
    main()