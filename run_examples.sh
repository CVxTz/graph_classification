wget -c -Oinput/ https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
cd input
tar -zxvf cora.tgz
cd ../code
python3 eda.py
python3 graph_features_embedding.py
