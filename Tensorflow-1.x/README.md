# ELDR
Explaining Low Dimensional Representations

A common workflow in data exploration is to learn a low-dimensional representation of the data, identify groups of points in that representation, and examine the differences between the groups to determine what they represent. 
We treat this as an interpretable machine learning problem by leveraging the model that learned the low-dimensional representation to help identify the key differences between the groups. 
To solve this problem, we introduce a new type of explanation, a Global Counterfactual Explanation (GCE), and our algorithm, Transitive Global Translations (TGT), for computing GCEs. 
TGT identifies the differences between each pair of groups using compressed sensing but constrains those pairwise differences to be consistent among all of the groups.
Empirically, we demonstrate that TGT is able to identify explanations that accurately explain the model while being relatively sparse, and that these explanations match real patterns in the data.


This repo contains an implementation of TGT as well as all of the code to reproduce the results in the [paper](https://proceedings.icml.cc/book/2020/hash/ccbd8ca962b80445df1f7f38c57759f0).  

# Dependencies
* svcis - https://github.com/shahcompbio/scvis
* Integrated Gradients - https://github.com/ankurtaly/Integrated-Gradients

# Project structure
* Each dataset in the project has 2 associated folders
* Each dataset has a {Dataset}-K folder has the run.ipynb where we can generate explanations for different sparisity levels
* It is also possible to tune the explanations by chaning the lamda parameter
* Each dataset also has a {Dataset} folder which has a Data subfolder containing the data and run.ipynb notebook to plot the results.
* vertices.py is provided in the {Dataset} folder which can be used for manually picking the clusters
* train_scvis.sh script is provided in the {Dataset} folder to train a vae model for the data
* All the datasets have trained models in {Dataset}/Model folder which can be used by default to reproduce the results
# Setup
* Clone the svcis github project to svcis folder
* Clone the Integrated Gradients github project to Integrated-Gradients folder
* If you are using conda use the provided env.yml to load the environment
* Run scvis/setup.py


# Steps to run the experiments
* Use the run.ipynb notebooks to reproduce the results
* Alternatively, one can experiment from the scratch by following the below steps
    * Train the model using {Dataset}/train_scvis.sh providing the arguments as necessary for the dataset
    * Set run = True in the {Dataset-K}/run.ipynb notebook and choose appropriate K values for which explanations are to be generated. Optionally, one can also choose a list of values for the hyper-parameter lambda.
    * Once the explanations are generated for different sparsity levels, the results can be plotted using {Dataset}/run.ipynb notebook.
    * We can choose different pairs of groups for analysis of coverage, correctness and consistency of the explanations.

# Steps to run on a new dataset
* Prepare the dataset in a tsv file in the {Dataset}/Data folder.
* Train the model using {Dataset}/train_scvis.sh providing the arguments as necessary for the dataset
* Run the vertices.py and manually choose the clusters in the latent dimensional space
* Set run = True in the {Dataset-K}/run.ipynb notebook and choose appropriate K values for which explanations are to be generated. Optionally, one can also choose a list of values for the hyper-parameter lambda.
* Once the explanations are generated for different sparsity levels, the results can be plotted using {Dataset}/run.ipynb notebook.
* We can choose different pairs of groups for analysis of coverage, correctness and consistency of the explanations.
