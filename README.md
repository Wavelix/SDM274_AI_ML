# SDM274_AI_and_Machine_Learning

![Static Badge](https://img.shields.io/badge/2024-Autumn-red)

---

# Introduction

### Lab

- **Linear Regression**: Using ```numpy```, perform linear regression with BGD, SGD, and MBGD respectively, and explore the impact of normalization. Visualize the results.
- **Perceptron**: Using ``numpy``, split the ``wine.data`` dataset into training and test sets (7:3 ratio), and perform binary classification by perceptron. Visualize the results.
- **Logistic Regression**: Using ``numpy``, consider the logistic regression model and cross-entropy loss functiong to address the binary classification problem of ``wine.data``. Visualize the results.
- **MLP**: Using ``numpy`` and ``matplotlib``, complete the following tasks:
  - Develop a MLP model, which can handle any number of layers and units per layer. Both MBGD and BGD can be implemented during the training.
  - Apply k-fold validation to assess model's performance.
  - Select a nonlinear function to generate a dataset, and use the model above to train and analyze.
  - Create a dataset which is suitable for binary classification, and use the model above to train and analyze.
- **KNN**: Using ``numpy`` and ``KDtree`` from ``scipy.spatial``, Anayze the Breast Cancer Wisconsin (Original) data using KNN algorithm. Test the algorithm with varying $k$ from 1 to 10 using the testing set, and provide the accuracy for each $k$ . Visualize the results.
- **Decision Tree**:  Use a portion of the data as the test dataset, and verify the classification performance of the decision tree use the test dataset. Evaluate using accuracy as the metric. Visualize the result by graphviz.
- **Multi-class Classification**: Using ``numpy`` and ``matplotlib``, construct a MLP model. Train the model using ``optdigits.tra``, and evaluate the model's performance using ``optdigits.tes``. Visualize the result.
- **PCA and Autoencoder**: Using ``numpy`` and ``matplotlib``, complete the following tasks:
  - Implement PCA on the ``wine.data``, with only the first 2 components considered. Then, use the 2 components to reconstruct the data, and visualize the output.
  - Implement both linear and non-linear autoencoder on the ``wine.data``. Then reconstruct the data using the trained autoencoder, and visualize the output.

### Project

- **Midterm Project**: Using ``numpy`` and ``matplotlib``, implement linear regression, perceptron, logistic regression and MLP models to predict machine failures in [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Train each model on the training set and evaluate their performance on the testing set. Use appropriate metrics such as accuracy, precision, recall, F1-score, to evaluate the models.
- **Final Project**: Analyze and classify the wheat seed dataset using K-Means++, soft K-Means, PCA, Nonlinear Autoencoders, MLP, SVM, SVM with Gaussian kernel and AdaBoost, and compare the performance of these methods on both multi-class and binary classification tasks.
