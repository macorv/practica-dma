#PANDAS
import pandas as pd
#KERAS
import keras as kr
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model


# MODELO DE REGRESION
# Especificacion del Modelo
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compilacion del Modelo: Especifico la funcion de perdida y el optimizador
model.compile(loss='mean_squa_error', optimizer = 'adam')

# Ajuste del Modelo
model.fit(predictor, target)

# Guardar modelo
model.save('model_file.h5')

# Cargar el modelo
my_model = load_model('model_file.h5')

# Predicciones
predictions = my_model.predict(data_para_predecir)



# MODELO DE CLASIFICACION
# Especificacion del Modelo
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))

# Compilacion del Modelo: En el caso de clasificacion se usa ademas 'metrics'
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Ajuste del Modelo
model.fit(predictor, target)

# Guardar modelo, de esta manera lo guarda para despues poder hacer predicciones.
model.save('model_file.h5')

# Cargar el modelo. Cargo el modelo. Porque para usar el metodo predict neccesito cargar el modelo previamente guardado.
my_model = load_model('model_file.h5')

# Predicciones
predictions = my_model.predict(data_para_predecir)

# Probabilidades
probability_true = predictions[:,1]
