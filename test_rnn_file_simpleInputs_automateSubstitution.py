from keras.layers import Input, Embedding, Reshape, merge, Dropout, Dense, LSTM, core, Activation
from keras.layers import TimeDistributed, Flatten, concatenate

from keras.engine import Model
from keras.models import Sequential

from keras import layers

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

sc = StandardScaler()

## data prep

# import numpy as np
dataset_1 = pd.read_csv('./diagSet_7p_withID.csv')

X = dataset_1.iloc[:, 1:8].values
y = dataset_1.iloc[:, 8].values

# order of parameters: "LinkId", "age", "ethnicity", "sex", "hba1c", "sbp", "dbp", "bmi", "anyInsulin"

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_ethnicity = LabelEncoder()
X[:, 1] = labelencoder_X_ethnicity.fit_transform(X[:, 1])

# don't technically need to do this bit
labelencoder_X_sex = LabelEncoder()
X[:, 2] = labelencoder_X_sex.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# remove first dummy variable to avoid the DV trap
X = X[:, 1:]


# split
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# scale inputs
# age - numerical input 0
ageScaler = preprocessing.StandardScaler().fit(X_train[:, 22])
X_train[:, 22] = ageScaler.transform(X_train[:, 22])

X_test[:, 22] = ageScaler.transform(X_test[:, 22])
X_val[:, 22] = ageScaler.transform(X_val[:, 22])

# hba1c - numerical input 0
hba1cScaler = preprocessing.StandardScaler().fit(X_train[:, 24])
X_train[:, 24] = hba1cScaler.transform(X_train[:, 24])

X_test[:, 24] = hba1cScaler.transform(X_test[:, 24])
X_val[:, 24] = hba1cScaler.transform(X_val[:, 24])

# sbp - numerical input 0
sbpScaler = preprocessing.StandardScaler().fit(X_train[:, 25])
X_train[:, 25] = sbpScaler.transform(X_train[:, 25])

X_test[:, 25] = sbpScaler.transform(X_test[:, 25])
X_val[:, 25] = sbpScaler.transform(X_val[:, 25])

# dbp - numerical input 0
dbpScaler = preprocessing.StandardScaler().fit(X_train[:, 26])
X_train[:, 26] = dbpScaler.transform(X_train[:, 26])

X_test[:, 26] = dbpScaler.transform(X_test[:, 26])
X_val[:, 26] = dbpScaler.transform(X_val[:, 26])

# bmi - numerical input 0
bmiScaler = preprocessing.StandardScaler().fit(X_train[:, 27])
X_train[:, 27] = bmiScaler.transform(X_train[:, 27])

X_test[:, 27] = bmiScaler.transform(X_test[:, 27])
X_val[:, 27] = bmiScaler.transform(X_val[:, 27])

## ANN setup

# simple version
nodes = 256

# initialise
classifier = Sequential()
# add input layer with Dropout
classifier.add(Dense(activation="relu", units=nodes, input_dim=28))
classifier.add(Dropout(0.5))

# add hidden layer
#classifier.add(Dense(activation="relu", units=nodes))
#classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=nodes))

# add output layer
classifier.add(Dense(activation="sigmoid", units=1))

# compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit the ann to the training set
history = classifier.fit(X_train, y_train, batch_size = 128, epochs = 16, validation_data = (X_val, y_val))

# plot losses
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.legend()
plt.savefig('loss_valLoss.png', dpi = 300)
plt.clf()

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.legend()
plt.savefig('loss_valAcc.png', dpi = 300)
plt.clf()

# predict test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)

##



dense_node_n = 28
# a = input the drug dataset (2-dimensional: IDs, timesteps)
diag_set = Input(shape = (len(X_train[0]), ), dtype='float32', name = 'diag_set')
# embed drug layer
#emb = Embedding(input_dim = 1000, output_dim = 64)(drug_set) # lower output dimensions seems better

# numericTS_set = input the numerical data (3-dimensional: IDs, timesteps, dimensions(n parameters))
#numericTS_set = Input(shape = (len(set1[0]), 3), name = 'numericTS_set')

# merge embedded and numerical data
# merged = keras.layers.concatenate([emb, numericTS_set])
#merged = merge([emb, numericTS_set], mode='concat')

#lstm_out = LSTM(return_sequences=True, input_shape = (len(set1[0]), 1027), units=64, recurrent_dropout = 0.5)(merged)
#lstm_out = LSTM(units = 128)(lstm_out)
#lstm_out = LSTM(return_sequences=True, input_shape = (len(set1[0]), 67), units=4, dropout = 0.1, recurrent_dropout = 0.5)(merged)
#lstm_out = LSTM(units = 8)(lstm_out)


#auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

#auxiliary_input = Input(shape=(1,), name='aux_input')
# x = merge([lstm_out, auxiliary_input], mode = 'concat')
x = diag_set

x = Dense(dense_node_n, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(dense_node_n, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(dense_node_n)(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[X_train], outputs=[main_output])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc'])

history = model.fit([X_train_drugs, X_train_numericalTS, X_train_age], [y_train, y_train], epochs=12, batch_size=256, validation_data = ([[X_val_drugs, X_val_numericalTS, X_val_age], [y_val, y_val]]))

# plot losses
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.legend()
plt.savefig('loss_valLoss.png', dpi = 300)
plt.clf()

plt.plot(epochs, loss, 'bo', label = 'Training acc')
plt.plot(epochs, val_loss, 'b', label = 'Validation acc')
plt.legend()
plt.savefig('loss_valAcc.png', dpi = 300)
plt.clf()




y_pred_asNumber = model.predict([X_test_drugs, X_test_numericalTS, X_test_age])
from sklearn.metrics import roc_auc_score
mainOutput = roc_auc_score(y_test, y_pred_asNumber[0])
auxOutput = roc_auc_score(y_test, y_pred_asNumber[1])

print(mainOutput)
print(auxOutput)

## change final year of therapy and output probabilities
# import lookup table
lookup = pd.read_csv('./inputFiles/lookup.csv')
lookup = lookup.sort_values('vectorNumbers')

therapyFrame = lookup.loc[(lookup['vectorWords'] == 'Metformin_') | # 1st line
(lookup['vectorWords'] == 'Metformin_SU_') |    # 2nd line
(lookup['vectorWords'] == 'DPP4_Metformin_') |
(lookup['vectorWords'] == 'Metformin_SGLT2_') |
(lookup['vectorWords'] == 'GLP1_Metformin_') |
(lookup['vectorWords'] == 'DPP4_Metformin_SU_') |   # 3rd line - MF/SU base
(lookup['vectorWords'] == 'DPP4_Metformin_SGLT2_') |
(lookup['vectorWords'] == 'DPP4_GLP1_Metformin_') |
(lookup['vectorWords'] == 'Metformin_SGLT2_SU_') |
(lookup['vectorWords'] == 'GLP1_Metformin_SU_') |
(lookup['vectorWords'] == 'Metformin_TZD_') |
(lookup['vectorWords'] == 'DPP4_Metformin_TZD_') |
(lookup['vectorWords'] == 'GLP1_Metformin_SGLT2_')]

therapyArray = np.array(therapyFrame['vectorNumbers'])
print(therapyArray)

numberTimeSteps = 60
nTimeStepsToReplace = 12
startTS = (numberTimeSteps - nTimeStepsToReplace)

for r_count in range(0, len(therapyArray), 1):
    print(r_count)
    X_test_drugs_substitute = X_test_drugs
    X_test_drugs_substitute[:, startTS:numberTimeSteps] = therapyArray[r_count]
    print(X_test_drugs_substitute)
    y_pred_asNumber_substitute = model.predict([X_test_drugs_substitute, X_test_numericalTS, X_test_age])
    np.savetxt('./pythonOutput/y_pred_asNumber_combinationNumber_' + str(r_count) + '.csv', y_pred_asNumber_substitute[0], fmt='%.18e', delimiter=',')

# write out X_test_drugs to send back to R for decoding/recoding
np.savetxt('./pythonOutput/X_test_drugs.csv', X_test_drugs, fmt='%.18e', delimiter=',')
np.savetxt('./pythonOutput/decoded_Xtest_hba1c.csv', sc_hba1c.inverse_transform(X_test_numericalTS[:, :, 0]), fmt='%.18e', delimiter=',')
np.savetxt('./pythonOutput/decoded_Xtest_sbp.csv', sc_sbp.inverse_transform(X_test_numericalTS[:, :, 1]), fmt='%.18e', delimiter=',')
np.savetxt('./pythonOutput/decoded_Xtest_bmi.csv', sc_bmi.inverse_transform(X_test_numericalTS[:, :, 2]), fmt='%.18e', delimiter=',')

np.savetxt('./pythonOutput/X_test_age.csv', X_test_age, fmt='%.18e', delimiter=',')

np.savetxt('./pythonOutput/y_pred_asNumber.csv', y_pred_asNumber, fmt='%.18e', delimiter=',')


'''
# build the RNN model
rnn = Sequential([
    LSTM(return_sequences=True, input_shape = (30, 11), units=128),
    Dropout(0.5),
    LSTM(16),
    Dropout(0.5),
#    LSTM(4),
#    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])(merged)
M = Model(input=[a,b], output=[rnn])
M.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## fit and evaluate
M.fit([X_train_set4, X_train_set1_to_3], y_train, batch_size = 128, epochs = 4)
score = M.evaluate([X_test_set4, X_test_set1_to_3], y_test, batch_size=128)
y_pred_asNumber = M.predict([X_test_set4, X_test_set1_to_3])
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_asNumber)
'''

# plot ROC
from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_asNumber[0])

fpr = fpr # false_positive_rate
tpr = tpr # true_positive_rate

# This is the ROC curve
plt.plot(fpr,tpr)
# plt.show()
plt.savefig('roc_mortality.png')
plt.clf()
auc = np.trapz(tpr, fpr)

print(auc)

plt.hist(y_pred_asNumber[0], bins = 100)
plt.savefig('y_pred_distribution')
plt.clf()
