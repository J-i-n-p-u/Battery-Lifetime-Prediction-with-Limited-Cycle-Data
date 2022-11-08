import tensorflow as tf
import prepare_data 
import RNN_model
import train_test

data_config = dict(
    split_ratio=[0.8, 0.1, 0.1],
    n = 5,
    resample_step = 10,
    resample_method = 'first',
    reshape_x1_method = 'circle to row',
    # reshape_x1_method = 'circle to col',
    )

# Define the model paras
model_config = dict(
    model_name = '',
    model_type = 'm1',
    # model_type = 'Bi_LSTM',
    networks = 'Bi_LSTM',
    epochs=5,
    batch_size=64,
    rnn_units= [512],
    dropout_rate=0.1,
    attention_units=0,
    dense_units = 300,
    shuffle=True,
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    patience=100,
)
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

train_test.fit_model(model, data_xy, model_config, data_config)

train_test.evaluate(model, data_xy, data_names_test, data_config, model_config, norm_paras)