from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from random import shuffle
import json

dataset = "cora"
edges_path = "../input/%s/%s.cites"%(dataset, dataset)
content_path = "../input/%s/%s.content"%(dataset, dataset)

with open(content_path, "r") as f:
    content_lines = [l.split() for l in f.read().split("\n") if l]

with open(edges_path, "r") as f:
    edges_lines = [l.split() for l in f.read().split("\n") if l]


content_data = [{"node":l[0],
                 "features":[float(x) for x in l[1:-1]],
                 "label":l[-1]} for l in content_lines]

labels = [x['label'] for x in content_data]

class_int_mapping = {k: i for i, k in enumerate(Counter(labels))}
node_int_mapping = {k['node']:i for i, k in enumerate(content_data)}
node_int_class_mapping = {node_int_mapping[c['node']]: class_int_mapping[c['label']] for c in content_data}
node_class_mapping = {c['node']: class_int_mapping[c['label']] for c in content_data}


colors = ['r', 'g', 'k', 'y', 'm', 'c', 'w', 'b']

nodes = list(range(len(node_int_mapping)))
node_color = [colors[node_int_class_mapping[n]] for n in nodes]
G=nx.DiGraph()

for node, int_node in node_int_mapping.items():
    G.add_node(int_node)

for edge in edges_lines:
    G.add_edge(node_int_mapping[edge[0]], node_int_mapping[edge[1]])

plt.figure(figsize=(14, 24))
nx.draw(G, node_size=30, width=0.2, nodelist=nodes, node_color=node_color)
plt.savefig('graph.png', transparent=True)


shuffle(content_data)

train_samples = []
train_class_counter = defaultdict(int)
val_samples = []
val_counter = 0
test_samples = []

for c in content_data:
    if train_class_counter[c['label']] <20:
        train_samples.append(c)
        train_class_counter[c['label']] += 1
    elif val_counter < 500:
        val_samples.append(c)
        val_counter += 1
    else:
        test_samples.append(c)

print(len(train_samples), len(val_samples), len(test_samples))


json.dump(train_samples, open("../input/%s/%s.train.json"%(dataset, dataset), 'w'), indent=4)
json.dump(val_samples, open("../input/%s/%s.val.json"%(dataset, dataset), 'w'), indent=4)
json.dump(test_samples, open("../input/%s/%s.test.json"%(dataset, dataset), 'w'), indent=4)
json.dump(edges_lines, open("../input/%s/%s.graph.json"%(dataset, dataset), 'w'), indent=4)
json.dump((class_int_mapping, node_int_mapping, node_int_class_mapping, node_class_mapping),
          open("../input/%s/%s.mappings.json"%(dataset, dataset), 'w'), indent=4)