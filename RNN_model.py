from tensorflow import keras
from tensorflow.keras import layers
from attention import Attention


def build_model_1(X_train, Y_train, model_config):
    
    rnn_units = model_config['rnn_units']
    attention_units = model_config['attention_units']
    dense_units = model_config['dense_units']
    model_type = model_config['networks']
    
    # Input
    x_input = keras.Input(shape=(X_train.shape[1], X_train.shape[2]), name='input')
    
    if len(rnn_units) == 1:
        if model_type == 'LSTM':
            x = layers.LSTM(rnn_units[0], return_sequences=False)(x_input)
        if model_type == 'GRU':
            x = layers.GRU(rnn_units[0], return_sequences=False)(x_input)
        if model_type == 'Bi_LSTM':
            x = layers.Bidirectional(layers.LSTM(rnn_units[0], return_sequences=False))(x_input)
        if model_type == 'Bi_GRU':
            x = layers.Bidirectional(layers.GRU(rnn_units[0], return_sequences=False))(x_input)
            
    if len(rnn_units) == 2:
        if model_type == 'LSTM':
            x = layers.LSTM(rnn_units[0], return_sequences=True)(x_input)
            if attention_units !=0:
                x = Attention(attention_units)(x)
            else:
                x = layers.LSTM(rnn_units[1], return_sequences=False)(x)
        if model_type == 'GRU':
            x = layers.GRU(rnn_units[0], return_sequences=True)(x_input)
            if attention_units !=0:
                x = Attention(attention_units)(x)
            else:
                x = layers.GRU(rnn_units[1], return_sequences=False)(x)
        if model_type == 'Bi_LSTM':
            x = layers.Bidirectional(layers.LSTM(rnn_units[0], return_sequences=True))(x_input)
            if attention_units !=0:
                x = Attention(attention_units)(x)
            else:
                x = layers.Bidirectional(layers.LSTM(rnn_units[1], return_sequences=False))(x)
        if model_type == 'Bi_GRU':
            x = layers.Bidirectional(layers.GRU(rnn_units[0], return_sequences=True))(x_input)
            if attention_units !=0:
                x = Attention(attention_units)(x)
            else:
                x = layers.Bidirectional(layers.GRU(rnn_units[1], return_sequences=False))(x)
    
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    # x = layers.BatchNormalization()(x)
    
    x = layers.Dense(dense_units, activation = 'relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation = 'relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation = 'relu')(x)
    
    x = layers.Dropout(model_config['dropout_rate'])(x)
    # x = layers.BatchNormalization()(x)
    paras_pred = layers.Dense(1, activation = 'sigmoid')(x)
    print(paras_pred.shape)
    model = keras.Model(inputs=x_input, outputs=paras_pred)
    
    # compile
    model.compile(
        loss=model_config['loss'],
        optimizer=model_config['optimizer'],
    )
    
    print(model.summary())
    
    return model

