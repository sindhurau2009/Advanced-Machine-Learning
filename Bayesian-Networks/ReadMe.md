Bayesian Networks are Probabilistic Graphical Models that are directed in nature.

1. In this project, we aim to construct a Bayesian Network on data collected from Kaggle and implement inference algorithms on this network.
2. The dataset "Mental Health in Tech Survey" is obtained from Kaggle which is a 2014 survey that aims to measure the frequency and attitude towards mental health in tech workplace.
3. Bayesian Network is drawn based on intuition and is modeled using 'pgmpy' library in python.
4. The data is loaded into a variable and is fitted to the Bayesian Network constructed just by using the node names and edges.
5. Once the data is fitted onto the model, Conditional Probability Distributions are obtained from this and the model is now fit to train on inference algorithms.
6. BeliefPropagation and VariableElimination inference algorithms aer used to answer queries using the CPD's of the model.
7. Sampling is performed and the resulting performance is measured in terms of mean and entropy.
