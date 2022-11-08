import numpy as np
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import prepare_data 
import RNN_model


def fit_model(model, data, model_config, data_config):
    
    X_train, Y_train, X_val, Y_val, _, _ = data
    
    n = data_config['n']
    model_name = model_config['model_name']
    print(model_name)
    # EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss',
                                    patience=model_config['patience'])
    
    history = model.fit(X_train, Y_train,
              epochs=model_config['epochs'],
              batch_size=model_config['batch_size'],
              validation_data=(X_val, Y_val),
              shuffle=model_config['shuffle'],
              callbacks=[early_stopping],
              verbose=1)
    
    plt.close('all')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.savefig(f'circle_{n}/model loss/{model_name}.png')
    model.save(f'circle_{n}/model/{model_name}')
    tf.keras.utils.plot_model(model,to_file = f'circle_{n}/model plot/{model_name}.png',show_shapes=True,)

def pre_eval(model, X_test, Y_test, norm_paras):

    print('Y_test.shape', Y_test.shape)
    yhat = model.predict(X_test)
    print('yhat.shape' ,yhat.shape)
    
    Y_max = norm_paras['Y_max']
    Y_min = norm_paras['Y_min']
    Y_test = Y_test * (Y_max - Y_min) + Y_min
    yhat = yhat * (Y_max - Y_min) + Y_min
    
    mae = np.mean(np.abs(Y_test - yhat))
    # print(f'MAE: {mae}')
    
    rmse = np.sqrt(np.mean((Y_test - yhat) **2))
    # print(f'RMSE: {rmse}')
    
    # Y_test shape: (None, 1)
    # yhat shape: (None, 1)
    mape = np.mean(np.abs(Y_test - yhat) / Y_test)
    # print(f'MAPE: {mape}')
    
    r2 = r2_score(Y_test,yhat)
    # print(f'R2: {r2}')
    
    return Y_test, yhat, [mae, rmse, mape, r2]


def evaluate(model, data, data_names_test, data_config, model_config, norm_paras):
    def to_csv(yhat, Y_test, data_names_test, data_config):
        
        model_name = model_config['model_name']
        n = data_config['n']
        
        test_y = pd.DataFrame(np.c_[Y_test, yhat],
                          index=[x[0,0] for x in data_names_test],
                          columns=['Y_test', 'Yhat'])
        test_y.index.name = 'battery'
        
        test_y.to_csv(f'circle_{n}/real_pre/{model_name}' + '.csv')
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data
    
    _, _, error_train = pre_eval(model, X_train, Y_train, norm_paras)
    _, _, error_val = pre_eval(model, X_val, Y_val, norm_paras)
    Y_test, yhat, error_test = pre_eval(model, X_test, Y_test, norm_paras)
    
    model_name = model_config['model_name']
    n = data_config['n']
    
    error = pd.DataFrame([error_train,error_val,error_test], 
                         index=['train', 'val', 'test'],
                          columns=['mae', 'rmse', 'mape', 'r2'],
                         )
    print(error)
    
    error.to_csv(f'circle_{n}/model error/{model_name}.csv')
    
    model_table = pd.read_excel(f'circle_{n}/model_table.xlsx', header=0,index_col=0)
    model_table[model_name] = list(model_config.values())[1:]+list(data_config.values()) \
        + error_test + error_val + error_train
    model_table.to_excel(f'circle_{n}/model_table.xlsx')
    
    to_csv(yhat, Y_test, data_names_test, data_config)

def train_and_eval(data_config, model_config):
     # create dir to save model
    prepare_data.create_new_dir(data_config, model_config)
    
    model_config['model_name'] = prepare_data.generate_model_name(data_config, model_config)
    
    # create xy data
    train, val, test, norm_paras = prepare_data.create(n = data_config['n'],
                                                        split_ratio=data_config['split_ratio'])

    # OR load xy data
    # train, val, test, norm_paras = retrieve_train_val_test(data_config)
    
    X1_train, X2_train, Y_train, data_names_train = train
    X1_val, X2_val, Y_val, data_names_val = val
    X1_test, X2_test, Y_test, data_names_test = test
    # del train, val, test

    X_train, X_val, X_test = prepare_data.prepare_x(X1_train, X1_val, X1_test, 
                                       X2_train, X2_val, X2_test, 
                                       data_config, model_config)

    # build model
    model = RNN_model.build_model_1(X_train, Y_train, model_config)

    data_xy = (X_train, Y_train, X_val, Y_val, X_test, Y_test)

    fit_model(model, data_xy, model_config, data_config)
    
    evaluate(model, data_xy, data_names_test, data_config, model_config, norm_paras)