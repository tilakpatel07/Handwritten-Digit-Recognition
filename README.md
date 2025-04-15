# Handwritten-Digit-Recognition
Python implementation of Logistic Regression and SVM for MNIST handwritten digit recognition.  Demonstrates  data handling, model training, and performance evaluation using scikit-learn.

# Handwritten Digit Recognition with Logistic Regression and SVM

## Overview

This project implements machine learning models to recognize handwritten digits from the MNIST dataset. The MNIST dataset is a widely used benchmark dataset in the field of computer vision and machine learning. It consists of grayscale images of handwritten digits (0-9).  This project uses Python with the scikit-learn library. Two classification algorithms are implemented:

* **Logistic Regression:** A linear model for classification.
* **Support Vector Machine (SVM):** A powerful algorithm that can perform both linear and non-linear classification.  Here, we explore its linear form.

## Project Structure

The project code is organized as follows:

* `logistic.ipynb`: Python script for training and evaluating a Logistic Regression model.
* `support.ipynb`: Python script for training and evaluating a Support Vector Machine (SVM) model.
* `README.md`: Project documentation (this file).

## Dependencies

* Python 3.x
* NumPy
* scikit-learn
* struct
* pickle

## Setup
1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    cd <your_repository_name>
    ```
2.  **Install the required packages:**
    ```bash
    pip install numpy scikit-learn
    ```
3.  **Download the MNIST dataset:**
    * The code assumes the MNIST dataset is located in a directory named `handwritten_numbers_dataset`. The structure should be:

        ```
        handwritten_numbers_dataset/
        ├── train-labels-idx1-ubyte
        ├── train-images-idx3-ubyte
        ├── t10k-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        ```

    * Download the MNIST dataset from the official source: [The MNIST Database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
    * Download the following files:
        * train-images-idx3-ubyte.gz
        * train-labels-idx1-ubyte.gz
        * t10k-images-idx3-ubyte.gz
        * t10k-labels-idx1-ubyte.gz
    * Extract (unzip) the downloaded files.
    * Place the extracted files in the `handwritten_numbers_dataset` directory.  Ensure the files are named as extracted (e.g., `train-images-idx3-ubyte`, not `train-images-idx3-ubyte.gz`).

## Usage

1.  **Run the Logistic Regression script:**
    ```bash
    python logistic_regression.py
    ```
    This script will:
    * Load the MNIST training and testing data.
    * Preprocess the data (normalize and reshape).
    * Split the training data into training and validation sets.
    * Perform hyperparameter tuning using GridSearchCV.
    * Train a Logistic Regression model with the best hyperparameters.
    * Evaluate the model on the validation and test sets.
    * Save the trained model to a pickle file (`my_trained_logistic_regression.pkl`).

2.  **Run the SVM script:**
    ```bash
    python svm.py
    ```
    This script will:
    * Load the MNIST training and testing data.
    * Preprocess the data (normalize and reshape).
    * Split the training data into training and validation sets.
    * Perform hyperparameter tuning using GridSearchCV.
    * Train a Support Vector Machine (SVM) model with the best hyperparameters (using a linear kernel).
    * Evaluate the model on the validation and test sets.
    * Save the trained model to a pickle file (`my_trained_svm_model.pkl`).

## Results

### Model Comparison

| Model             | Validation Accuracy | Validation F1-Score | Test Accuracy | Test F1-Score |
| ----------------- | ------------------- | --------------------- | ------------- | ------------- |
| Logistic Regression | \[0.9236152769446111]     |  \[0.9234478331801074]       |  \[0.9247]   |  \[0.9245159783642534]    |
| SVM (Linear)      |   \[0.943011397720456]    |    \[0.942856096943864]     |   \[0.9447]    |   \[0.9445740358444961]    |


As you can see, the SVM model with a linear kernel achieves a higher accuracy and F1-score than the Logistic Regression model on this task.  This suggests that, for this specific problem, the SVM is able to learn a more effective decision boundary.  However,  Logistic Regression provides a very strong baseline and is computationally more efficient.


