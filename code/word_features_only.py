import json

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2, l1
from sklearn.metrics import accuracy_score

dataset = "cora"
#cora Accuracy test : 0.5328820116054158


def get_features_only_model(n_features, n_classes):
    in_ = Input((n_features,))
    x = Dense(10, activation="relu", kernel_regularizer=l1(0.001))(in_)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(in_, x)

    model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer="adam")

    model.summary()

    return model



train_samples = json.load(open("../input/%s/%s.train.json" % (dataset, dataset), 'r'))
val_samples = json.load(open("../input/%s/%s.val.json" % (dataset, dataset), 'r'))
test_samples = json.load(open("../input/%s/%s.test.json" % (dataset, dataset), 'r'))
edges_lines = json.load(open("../input/%s/%s.graph.json" % (dataset, dataset), 'r'))
class_int_mapping, node_int_mapping, node_int_class_mapping, node_class_mapping = \
    json.load(open("../input/%s/%s.mappings.json" % (dataset, dataset), 'r'))

X_train = [x['features'] for x in train_samples]
Y_train = [class_int_mapping[x['label']] for x in train_samples]

X_val = [x['features'] for x in val_samples]
Y_val = [class_int_mapping[x['label']] for x in val_samples]

X_test = [x['features'] for x in test_samples]
Y_test = [class_int_mapping[x['label']] for x in test_samples]

X_train, Y_train = np.array(X_train), np.array(Y_train)[:, np.newaxis]
X_test, Y_test = np.array(X_test), np.array(Y_test)[:, np.newaxis]
X_val, Y_val = np.array(X_val), np.array(Y_val)[:, np.newaxis]

model = get_features_only_model(n_features=int(X_train.shape[1]), n_classes=int(max(Y_train) + 1))

early = EarlyStopping(monitor="val_acc", patience=10, restore_best_weights=True)

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=400, verbose=2, callbacks=[early])

Y_test_pred = model.predict(X_test)
Y_test_pred = Y_test_pred.argmax(axis=-1).ravel()

print("Accuracy test : %s" % (accuracy_score(Y_test, Y_test_pred)))
