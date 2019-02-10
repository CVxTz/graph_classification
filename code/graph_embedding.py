import json

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, Multiply, Concatenate
from keras.models import Model
from keras.regularizers import l2, l1
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import sample, choice
from sklearn.neighbors import NearestNeighbors
import networkx as nx


dataset = "cora"
#cora Accuracy test : 0.7306576402321083

batch_size = 32
n_neighbors = 5

train_samples = json.load(open("../input/%s/%s.train.json" % (dataset, dataset), 'r'))
val_samples = json.load(open("../input/%s/%s.val.json" % (dataset, dataset), 'r'))
test_samples = json.load(open("../input/%s/%s.test.json" % (dataset, dataset), 'r'))
edges_lines = json.load(open("../input/%s/%s.graph.json" % (dataset, dataset), 'r'))
class_int_mapping, node_int_mapping, node_int_class_mapping, node_class_mapping = \
    json.load(open("../input/%s/%s.mappings.json" % (dataset, dataset), 'r'))

node_int_features_mapping = {node_int_mapping[k['node']]:np.array(k['features']) for k in train_samples+val_samples+test_samples}

G=nx.Graph()

for node, int_node in node_int_mapping.items():
    G.add_node(int_node)

for edge in edges_lines:
    G.add_edge(node_int_mapping[edge[0]], node_int_mapping[edge[1]])

spl = dict(nx.all_pairs_shortest_path_length(G))


def get_graph_embedding_model(n_nodes):
    in_1 = Input((1,))
    in_2 = Input((1,))

    emb = Embedding(n_nodes, 100, name="node1")
    x1 = emb(in_1)
    x2 = emb(in_2)

    x1 = Flatten()(x1)
    x1 = Dropout(0.1)(x1)
    x2 = Flatten()(x2)
    x2 = Dropout(0.1)(x2)

    x = Multiply()([x1, x2])
    x = Dropout(0.1)(x)

    x = Dense(1, activation="linear", name="spl")(x)

    model = Model([in_1, in_2], x)

    model.compile(loss="mae", optimizer="adam")

    model.summary()

    return model


def get_features_graph_model(n_features, n_classes, n_nodes):
    in_1 = Input((n_features,))
    in_2 = Input((n_features,))

    in_3 = Input((1,))
    emb = Embedding(n_nodes, 100, name="node1", trainable=False)
    x1 = emb(in_3)
    x1 = Flatten()(x1)
    d = Dense(10, kernel_regularizer=l2(0.0005))
    x2 = d(in_1)
    x3 = d(in_2)

    x = Concatenate()([x1, x2, x3])

    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax", kernel_regularizer=l1(0.001))(x)

    model = Model([in_1, in_2, in_3], x)

    model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer="adam")

    model.summary()

    return model


def gen(list_edges, node_int_mapping, batch_size=batch_size):

    while True:
        positive_samples = sample(list_edges, batch_size//2)
        positive_samples = [[node_int_mapping[x[0]], node_int_mapping[x[1]]] for x in positive_samples]

        negative_samples = [[choice(range(len(node_int_mapping))), choice(range(len(node_int_mapping)))] for _ in
                             range(batch_size//2)]

        samples = positive_samples+negative_samples

        X1 = [x[0] for x in samples]
        X2 = [x[1] for x in samples]

        labels = [1/max(spl[x[0]].get(x[1], 100), 1) for x in samples]
        #print(labels)

        yield [np.array(X1),np.array(X2)], np.array(labels)


train, test = train_test_split(edges_lines, test_size=0.05)

model_g = get_graph_embedding_model(len(node_int_mapping))

# early = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
#
# model_g.fit_generator(gen(train, node_int_mapping), validation_data=gen(test, node_int_mapping), nb_epoch=500, verbose=2,
#                     callbacks=[early], steps_per_epoch=1000, validation_steps=100)
#
# model_g.save_weights("graph_model_1.h5")

model_g.load_weights("graph_model_1.h5")

W = model_g.layers[2].get_weights()[0]

nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
nearest_neighbors.fit(W)
W_pred = nearest_neighbors.kneighbors(W, return_distance=False)


n_features = len(train_samples[0]['features'])

neigh_features = {}

for i, neighs in enumerate(W_pred):

    v = np.zeros((n_features, ))
    cnt = 0.0001
    for n in neighs[1:]:
        print(n)
        v += node_int_features_mapping[n]
        cnt += 1

    neigh_features[i] = v/cnt




X_train = [x['features'] for x in train_samples]
X_G_train = [node_int_mapping[x['node']] for x in train_samples]
X_train_2 = [neigh_features[node_int_mapping[x['node']]] for x in train_samples]
Y_train = [class_int_mapping[x['label']] for x in train_samples]

X_val = [x['features'] for x in val_samples]
X_G_val = [node_int_mapping[x['node']] for x in val_samples]
X_val_2 = [neigh_features[node_int_mapping[x['node']]] for x in val_samples]
Y_val = [class_int_mapping[x['label']] for x in val_samples]

X_test = [x['features'] for x in test_samples]
X_G_test = [node_int_mapping[x['node']] for x in test_samples]
X_test_2 = [neigh_features[node_int_mapping[x['node']]] for x in test_samples]
Y_test = [class_int_mapping[x['label']] for x in test_samples]

X_train, X_train_2, X_G_train, Y_train = np.array(X_train), np.array(X_train_2), np.array(X_G_train)[:, np.newaxis],\
                                         np.array(Y_train)[:, np.newaxis]
X_val, X_val_2, X_G_val, Y_val = np.array(X_val), np.array(X_val_2), np.array(X_G_val)[:, np.newaxis],\
                                         np.array(Y_val)[:, np.newaxis]
X_test, X_test_2, X_G_test, Y_test = np.array(X_test), np.array(X_test_2), np.array(X_G_test)[:, np.newaxis],\
                                         np.array(Y_test)[:, np.newaxis]


model = get_features_graph_model(n_features=int(X_train.shape[1]), n_classes=int(max(Y_train) + 1), n_nodes = len(node_int_mapping))
model.load_weights("graph_model_1.h5", by_name=True)
early = EarlyStopping(monitor="val_acc", patience=5000, restore_best_weights=True)

model.fit([X_train, X_train_2, X_G_train], Y_train, validation_data=([X_val, X_val_2, X_G_val], Y_val),
          nb_epoch=1000, verbose=2, callbacks=[early])

Y_test_pred = model.predict([X_test, X_test_2, X_G_test])
Y_test_pred = Y_test_pred.argmax(axis=-1).ravel()

print("Accuracy test : %s" % (accuracy_score(Y_test, Y_test_pred)))
