from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import random
import os

def create_dataset(data, n=10):
    """
    Create dataset from processed_data.pkl

    Parameters
    ----------
    data : pkl file
        processed_data.pkl.
    n : int, optional
        number of cycles in each sample. The default is 10.

    Returns
    -------
    TYPE
        Time-dependent features (X1)
        Scalar featuresm (X2)
        Remaining cycles (Y)
        Sample name list (data_names)

    """
    
    # the size of the sliding window
    batches = list(data.keys()) # ['b1c0','b1c1',...]
    X1, X2, Y = [], [], []
    data_names = []
    for batch in tqdm(batches):
        cyclelife = data[batch]['cycle_life']
        cyclelist = list(data[batch]['cycles'].keys())
        for idx, cyclenum in enumerate(cyclelist):
            i = int(cyclenum)
            if i % n == 1 and i < cyclelife-2*n:
                X1_i, X2_i, Y_i, data_names_i = [], [], [], []
                
                for loc, num in enumerate(cyclelist[idx:idx+n]):
                    # print(num, loc+idx)
                    T = data[batch]['cycles'][num]['Tdlin']
                    Q = data[batch]['cycles'][num]['Qdlin']
                    V = data[batch]['cycles'][num]['Vdlin']
                      
                    Discharge_time = data[batch]['summary']['Discharge_time'][loc+idx]
                    IR = data[batch]['summary']['IR'][loc+idx]
                    QD = data[batch]['summary']['QD'][loc+idx]
                    Remaining_cycles = data[batch]['summary']['Remaining_cycles'][loc+idx]
                      
                    x1 = pd.DataFrame(np.array([Q, T, V]).T, columns=[
                                      'Qdlin', 'Tdlin', 'Vdlin'])
                    x2 = pd.DataFrame(np.array([Discharge_time, IR, QD]).reshape(
                        (1, 3)), columns=['Discharge_time', 'IR', 'QD'])
                    
                    # y = Remaining_cycles
                    y = cyclelife-i+n
                    
                    X1_i.append(x1)
                    X2_i.append(x2)
                    Y_i.append(y)
                    data_names_i.append([batch, num, int(Remaining_cycles)])
                    
                X1.append(X1_i)
                X2.append(X2_i)
                Y.append(Y_i)
                data_names.append(data_names_i)
    
    X2 = np.array(X2)
    X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], X2.shape[3]))
    Y = np.mean(Y, axis=1, keepdims=True)
    print('total num of samples:', Y.shape[0])
    return np.array(X1), X2, Y, np.array(data_names)

def get_min_max(X1, X2, Y):
     norm_paras = {'X1_min': np.min(np.min(np.min(X1, axis=0), axis=0), axis=0, keepdims=True),
                   'X1_max': np.max(np.max(np.max(X1, axis=0), axis=0), axis=0, keepdims=True),
                   'X2_min': np.min(np.min(X2, axis=0), axis=0, keepdims=True),
                   'X2_max': np.max(np.max(X2, axis=0), axis=0, keepdims=True),
                   'Y_min': int(np.min(Y)),
                   'Y_max': int(np.max(Y))}
     print('\t\t Tdlin \t\t Qdlin \t\t Vdlin')
     print('X1_min:', norm_paras['X1_min'][0])
     print('X1_max:', norm_paras['X1_max'][0])
     
     print('\n\t\t Discharge_time \t IR \t QD')
     print('X2_min:', norm_paras['X2_min'][0])
     print('X2_max:', norm_paras['X2_max'][0])
     
     print('\nY_min:', norm_paras['Y_min'], '\tY_max', norm_paras['Y_max'])
     return norm_paras


def normalize(xy, parameters):
    X1, X2, Y = xy
    X1_min = parameters['X1_min']
    X2_min = parameters['X2_min']
    X1_max = parameters['X1_max']
    X2_max = parameters['X2_max']
    Y_min = parameters['Y_min']
    Y_max = parameters['Y_max']
    # y_range = parameters['y_range']

    X1 = (X1 - X1_min) / (X1_max - X1_min)
    X2 = (X2 - X2_min) / (X2_max - X2_min)
    Y = (Y - Y_min) / (Y_max - Y_min)

    return X1, X2, Y

def inverse_normalize(xy, parameters):
    X1, X2, Y = xy
    X1_min = parameters['X1_min']
    X2_min = parameters['X2_min']
    X1_max = parameters['X1_max']
    X2_max = parameters['X2_max']
    Y_min = parameters['Y_min']
    Y_max = parameters['Y_max']
    # y_range = parameters['y_range']

    X1 = X1 * (X1_max - X1_min) + X1_min
    X2 = X2 * (X2_max - X2_min) + X2_min
    Y = Y * (Y_max - Y_min) + Y_min

    return X1, X2, Y

def get_split_idx(num_sample, split_ratio, seed=1):

    idx = np.arange(num_sample) 
    split_num = num_sample * np.array(split_ratio)
    random.seed(seed)
    random.shuffle(idx)    
    
    return {'train': idx[: int(split_num[0])],
            'val': idx[int(split_num[0]) : int(split_num[0]) + int(split_num[1])],
            'test': idx[int(split_num[0]) + int(split_num[1]):]
            }

def split_train_val_test(data, split_idx, norm_paras, n, save):
    X1, X2, Y, data_names = data
    
    X1_train = X1[split_idx['train']]
    X2_train = X2[split_idx['train']]
    Y_train = Y[split_idx['train']]
    data_names_train = data_names[split_idx['train']]
    
    X1_val = X1[split_idx['val']]
    X2_val = X2[split_idx['val']]
    Y_val = Y[split_idx['val']]
    data_names_val = data_names[split_idx['val']]
    
    X1_test = X1[split_idx['test']]
    X2_test = X2[split_idx['test']]
    Y_test = Y[split_idx['test']]
    data_names_test = data_names[split_idx['test']]
    
    print(f'Train:Val:Test = {str(len(Y_train))}: \
          {str(len(Y_val))}:{str(len(Y_test))}')
    if save:
        print(f'Saving to train_val_test_{n}.npz...')
        # save train, valid, test
        np.savez(f'train_val_test_{n}.npz',
                 X1_train=X1_train, X2_train=X2_train, Y_train=Y_train,
                 X1_val=X1_val, X2_val=X2_val, Y_val=Y_val,
                 X1_test=X1_test, X2_test=X2_test, Y_test=Y_test,
                 data_names_train = data_names_train,
                 data_names_val = data_names_val,
                 data_names_test = data_names_test,
                 norm_paras = norm_paras
                 )
    
    return ((X1_train, X2_train, Y_train, data_names_train), \
            (X1_val, X2_val, Y_val, data_names_val), \
                (X1_test, X2_test, Y_test, data_names_test))

def create(n=5, split_ratio=[0.8, 0.1, 0.1], save=True):
    
    print('-'*50)
    print('Loading \'processed_data.pkl\'')
    
    print()
    print('-'*50)
    print(f'Extracting X1, X2, Y from \'processed_data.pkl\' (circle = {n})')
    print('-'*50)
    path = 'processed_data.pkl'
    data = pickle.load(open(path, 'rb'))
    X1, X2, Y, data_names = create_dataset(data, n)
    
    print()
    print('-'*50)
    print('Normalizing X1, X2, Y to 0~1')
    print('-'*50)
    norm_paras = get_min_max(X1, X2, Y)
    X1, X2, Y = normalize((X1, X2, Y), norm_paras)
    
    print()
    print('-'*50)
    print('Spliting samples into train, val, test')
    print('-'*50)
    print('split ratio:',split_ratio)
    split_idx = get_split_idx(Y.shape[0], split_ratio, seed=1)
    train, val, test = split_train_val_test((X1, X2, Y, data_names), split_idx,
                                            norm_paras=norm_paras, n=n, save=True)
    X1_train, X2_train, Y_train, data_names_train = train
    # X1_val, X2_val, Y_val, data_names_val = val
    # X1_test, X2_test, Y_test, data_names_test = test
    # del train, val, test
    print()
    print('-'*50)
    print('Sahpe of Train')
    print('-'*50)
    print('X1_train Shape:', X1_train.shape)
    print('X2_train Shape:', X2_train.shape)
    print('Y_train Shape:', Y_train.shape)
    
    return train, val, test, norm_paras

def retrieve_train_val_test(data_config):

    n = data_config['n']
    data_xy = np.load(f'train_val_test_{n}.npz', allow_pickle=True)
    
    # Load train data
    train = (data_xy['X1_train'], data_xy['X2_train'], 
             data_xy['Y_train'], data_xy['data_names_train'])
    
    # Load validation data
    val = (data_xy['X1_val'], data_xy['X2_val'], 
           data_xy['Y_val'], data_xy['data_names_val'])
    
    # Load test data
    test = (data_xy['X1_test'], data_xy['X2_test'], 
            data_xy['Y_test'], data_xy['data_names_test'])

    return train, val, test, data_xy['norm_paras'].tolist()

def generate_model_name(data_config, model_config):
    n = data_config['n']
    model_name = str(len(os.listdir(f'circle_{n}/model')) + 1)
    
    return model_name

def create_new_dir(data_config, model_config):
    
    # create new dir
    n = data_config['n']
    if not os.path.exists(f'circle_{n}'):
        os.mkdir(f'circle_{n}')
    if not os.path.exists(f'circle_{n}/model'):
        os.mkdir(f'circle_{n}/model')
    if not os.path.exists(f'circle_{n}/model plot'):
        os.mkdir(f'circle_{n}/model plot')
    if not os.path.exists(f'circle_{n}/real_pre'):
        os.mkdir(f'circle_{n}/real_pre')
    if not os.path.exists(f'circle_{n}/model loss'):
        os.mkdir(f'circle_{n}/model loss')
    if not os.path.exists(f'circle_{n}/model error'):
        os.mkdir(f'circle_{n}/model error')
    if not os.path.exists(f'circle_{n}/model_table.xlsx'):
        table_idx = list(model_config.keys())+list(data_config.keys()) \
            + ['test_mae', 'test_rmse', 'test_mape', 'test_r2',
               'val_mae', 'val_rmse', 'val_mape', 'val_r2',
               'train_mae', 'train_rmse', 'train_mape', 'train_r2',]
        model_table = pd.DataFrame(index=table_idx)
        model_table.to_excel(f'circle_{n}/model_table.xlsx',header=0)


def resample_x1(X1_train, X1_val, X1_test, data_config):
    
    step = data_config['resample_step']
    method = data_config['resample_method']
    def resample(x, step, method):
        if method == 'first':
            return x[:,:,::step,:]
        if method == 'end':
            return x[:,:,step-1::step,:]
        if method == 'mean':
            nx = np.zeros_like(x[:,:,::step,:])
            for i in range(step):
                if nx.shape[2] == x[:,:,i::step,:].shape[2]:
                    nx = nx + x[:,:,i::step,:]
            nx = nx / step
            return nx 

    new_X1_train = resample(X1_train, step, method)
    new_X1_val = resample(X1_val, step, method)
    new_X1_test = resample(X1_test, step, method)
    
    return new_X1_train, new_X1_val, new_X1_test

def reshape_x1(X1_train, X1_val, X1_test, data_config):
    
    method = data_config['reshape_x1_method']
    def reshape(X, method):
        if method == 'circle to row':
            return X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
        if method == 'circle to col':
            return np.array([np.concatenate(x,axis=1) for x in X])
    
    new_X1_train = reshape(X1_train, method)
    new_X1_val = reshape(X1_val, method)
    new_X1_test = reshape(X1_test, method)
    
    return new_X1_train, new_X1_val, new_X1_test

def prepare_x(X1_train, X1_val, X1_test, 
              X2_train, X2_val, X2_test, 
              data_config, model_config):
    
    def prepare(x1, x2, model_type):
        if model_type == 'm1':
            return np.concatenate((x1, x2), axis = 2)
        if model_type == 'm2':
            return (x1, x2)
        
    X1_train, X1_val, X1_test = resample_x1(X1_train, X1_val, X1_test, data_config)
    X1_train, X1_val, X1_test = reshape_x1(X1_train, X1_val, X1_test, data_config)
    
    train_x = prepare(X1_train, X2_train, model_config['model_type'])
    val_x = prepare(X1_val, X2_val, model_config['model_type'])
    test_x = prepare(X1_test, X2_test, model_config['model_type'])
    
    return train_x, val_x, test_x

# train, val, test, norm_paras = create()







