# Google - Fast or Slow?
## Graph Neural Network AI Model Runtime Prediction
Google Kaggle competition for predicting how fast an AI model runs


## Project Description

Submission for the Google AI Kaggle named "Google - Fast or Slow? Predict AI Model Runtime"


## Task Description

### Overview

Alice is an AI model developer, but some of the models her team developed run very slow. She recently discovered compiler's configurations that change the way the compiler compiles and optimizes the models, and hence make the models run faster (or slower)! Your task is to help Alice find the best configuration for each model.

Your goal: Train a machine learning model based on the runtime data provided to you in the training dataset and further predict the runtime of graphs and configurations in the test dataset.

The task is to predict how long it will take to run a machine learning model. The data provided is a graph of nodes and edges, with each node having a set of features. The goal is to predict the runtime of each node.

### About the data

Our dataset, called TpuGraphs, is the performance prediction dataset on XLA HLO graphs running on Tensor Processing Units (TPUs) v3. There are 5 data collections in total:layout:xla:random, layout:xla:default, layout:nlp:random, layout:nlp:default, and tile:xla. The final score will be the average across all collections. To download the entire dataset and view more information, you may navigate to the Data tab.

For a detailed task description, please refer to the Kaggle page: https://www.kaggle.com/competitions/predict-ai-model-runtime/data

### Timeline 

 - August 29, 2023 - Start Date.

 - November 10, 2023 - Entry Deadline. You must accept the competition rules before this date in order to compete.

 - November 10, 2023 - Team Merger Deadline. This is the last day participants may join or merge teams.

 - November 17, 2023 - Final Submission Deadline.


# Part 1: Data Preprocessing

In this section of the project, we will be working to preprocess and clean the data for analysis. We will be working with the following steps:

1) Visualize the data and identify any trends and patterns

2) Identify the target features and the predictor features

3) Identify any missing data and outliers

4) Identify how best to tackle the challenge


# Step 2: Understanding the Data

The data provided came from a 'npz' file, which is a numpy array file holding numerous numpy arrays. This data format is different from traditional csv files, which is typically two-dimensional. numpy arrays allow for multi-dimensional arrays, which is useful for storing data in a more complex format.

When looking to understand the data, it is useful to visualize the number of dimensions within each specific array and the shape of each array. This will help us understand the data and how to best work with it.

For example, we can see that the 'node_feat' array is two-dimensional, with a shape of: (41522,140), meaning that there are 41522 rows and 140 columns. This means that there are 41522 nodes and 140 features for each node.

We can also see that the 'node_config_feat' array is three-dimensional, with a shape of: (1000,2244,18), meaning that there are 1000 rows, 2244 columns, and 18 layers. This means that there are 1000 nodes, 2244 nodes, and 18 features for each node.

![image_1](images/image_1.png)

# Step 3: Identifying the Target and Predictor Features


# Step 4: Visualization and Plannng

## **Preliminary Analysis and Approach**:

1. **Graph Structure and Features**:  
    - AI models are represented as graphs, where nodes represent tensor operations and edges represent tensors.
    - `node_feat` and `node_opcode` likely represent features related to tensor operations.
    - `edge_index` is the representation of tensors or connections between tensor operations.

2. **Compiler Configurations**:
    - There are two main configurations: layout configuration and tile configuration.
    - `node_config_feat` with its 3D representation probably captures the essence of these configurations across different nodes.
    - `config_runtime` is crucial as it represents the runtime associated with these configurations - which is our main target to predict.

3. **ML Model Development**:
    - Given the nature of the data (graphs), Graph Neural Networks (GNNs) or similar models would be suitable for this task as they can capture the topology and features of graphs.
    - Input features would include node and edge features. The target variable would be `config_runtime`.
    - The model would be trained on the provided runtime data to understand the relationship between configurations and their resulting runtimes. The model can then predict runtimes for configurations in the test dataset.

4. **Evaluation and Optimization**:
    - Once the initial model is trained, it's important to evaluate its performance on a validation set. This will give insights into how well the model is likely to perform on unseen data.
    - Hyperparameter tuning, feature engineering, and other optimization techniques can be applied to improve the model's accuracy.

5. **Recommendation System**:
    - Based on the predicted runtimes, a recommendation system can be built to suggest the most optimal configuration for a given graph. The configuration that yields the lowest predicted runtime would be the recommended one.

6. **Handling Different Data Collections**:
    - The provided datasets (`layout:xla:random`, `layout:xla:default`, `layout:nlp:random`, `layout:nlp:default`, and `tile:xla`) might have subtle differences. It might be beneficial to train separate models for each collection or to include the collection type as an additional feature if using a unified model.

### Specific Observations:

1. node_feat:
This seems to represent features associated with nodes in a graph. The size of the second dimension is consistently 140, suggesting that every node has 140 features.
The number of nodes (i.e., the first dimension) varies across the dataset, ranging from 490 to 43,615.

2. node_opcode:
This is likely an opcode (operation code) or a label associated with each node. It's a 1D array, and its length aligns with the number of nodes in the node_feat array.

3. edge_index:
This array possibly represents the source and destination indices of edges in the graph. Each row could denote an edge, with the two columns representing the source and target nodes, respectively.
The number of edges varies between files, from 749 to 73,881.

4. node_config_feat:
This array seems to contain configurations related to nodes. It is consistently sampled 1,000 times (first dimension), but the number of nodes (second dimension) and features per node (third dimension) varies. However, each configuration has 18 features (third dimension).

5. node_config_ids:
A 1D array that might represent unique identifiers or labels for node configurations. The length of this array matches the second dimension of the node_config_feat array.

6. config_runtime:
This is likely a measure of runtime for the configurations. Each file has data for 1,000 configurations.

7. node_splits:
Represents some split on the nodes. The meaning is not immediately clear, but it's noteworthy that the number of splits varies across files.

### Summary:
The datasets are graph-based with associated configurations and runtime measurements.

Nodes have features (node_feat) and opcodes (node_opcode), and there's information on edges connecting them (edge_index).

Nodes have configurations (node_config_feat) with associated IDs (node_config_ids) and runtimes (config_runtime).

There's some unknown split or categorization on nodes (node_splits).

### ChatGPT Recommendations for Further Analysis:

#### Visualize Graphs:

Plot some of the smaller graphs to visualize their structure. For larger graphs, consider plotting a subset or use algorithms to find a meaningful subgraph.
Examine the distribution of node degrees and other graph metrics.


#### Examine Node Features:

Analyze the distribution of values in the node_feat arrays.
Consider dimensionality reduction techniques (e.g., PCA) to visualize the high-dimensional node features in 2D or 3D.
Study Configuration Run Times:

Analyze the config_runtime array to see the distribution of runtimes across configurations.
Correlate node configurations (node_config_feat) with runtimes to determine if specific configurations lead to faster or slower runtimes.
Node Splits Investigation:

Investigate node_splits more deeply to understand its purpose and relevance.
General Data Exploration:

For each array, check for missing values or anomalies.
Consider normalization or standardization if values vary widely.
Data Correlation:

Explore if there's a correlation between node features and other attributes like configurations, opcodes, or runtime.

Remember, understanding domain-specific context and having access to experts or documentation can provide even more insight into the nature and importance of each dataset feature.


