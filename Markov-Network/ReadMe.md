Markov Networks are Probabilistic Graphical Models that are undirected in nature. They do not have CPD's, rather they have factors that are essential for describing the relation of various attributes in the network. 

1. In this project, we aim to construct a Markov Network on data collected from Kaggle and implement inference algorithms on this network.
2. The dataset collected contains the details of the clients of a Portugese banking insitution. Some attributes in this dataset are age, huosing loan, personal loan, last month contacted, etc which help predict whether the client is interested in subscribing for a term deposit in the bank or not.
3. Markov Network is again drawn based on intuition and is modeled using 'pgmpy' library in python.
4. It is not possible to fit the data to this model, hence the co-relation between variables (edges) is taken as factors. This is obtained using groupby().size operation in Python.
5. Now that factors are obtained, inference algorithms can be trained on this model to see how well it can predict the term subscription of a client.
6. BeliefPropagation and VariableElimination inference algorithms aer used to answer queries using the CPD's of the model.
7. Sampling is performed and the resulting performance is measured in terms of mean and entropy.

