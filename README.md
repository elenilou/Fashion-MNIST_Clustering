# Fashion-MNIST Clustering with Dimensionality Reduction

## 📜 Project Overview

This project explores the application of various dimensionality reduction techniques (**PCA**, **Stacked Autoencoder**, **UMAP**) combined with different clustering algorithms (**MiniBatchKMeans**, **DBSCAN**, **Agglomerative Clustering**) on the Fashion-MNIST dataset. The goal is to evaluate how dimensionality reduction affects the performance of clustering on image data.

This was developed as the first assignment for the 

*Machine Learning* course at the *Department of Applied Informatics* of the *University of Macedonia*.

---

## 🖼️ Dataset

The dataset used is **Fashion-MNIST**, a collection of 60,000 training images and 10,000 testing images. Each image is a 28x28 grayscale image, associated with a label from 10 classes of clothing.

---

## 🛠️ Techniques Explored

### Dimensionality Reduction

* **Principal Component Analysis (PCA):** A linear dimensionality reduction technique that identifies the principal components (directions of maximum variance) in the data.
* **Stacked Autoencoder (SAE):** A type of neural network used for unsupervised dimensionality reduction by learning an encoding of the data.
* **UMAP (Uniform Manifold Approximation and Projection):** A non-linear dimensionality reduction technique that aims to preserve the local and global structure of the data.

### Clustering Algorithms

* **MiniBatchKMeans:** A variation of the K-Means algorithm that uses mini-batches to reduce computation time, especially for large datasets.
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** A density-based clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.
* **Agglomerative Clustering:** A type of hierarchical clustering that builds a hierarchy of clusters by successively merging clusters.

---

## ⚙️ Methodology

1.  **Data Loading and Preprocessing:** The Fashion-MNIST dataset is loaded and preprocessed using `MinMaxScaler` to normalize the pixel values. The image data is reshaped for processing and split into training and testing sets.
2.  **Dimensionality Reduction:** Each technique (PCA, SAE, UMAP) is applied to the training data.
3.  **Encoding Test Data:** The trained dimensionality reduction models are used to encode the test data.
4.  **Clustering:** Each clustering algorithm is applied to the original scaled test data and the encoded test data from each reduction technique.
5.  **Evaluation:** The performance is evaluated using various metrics:
    * **Silhouette Score:** Higher values indicate better-defined clusters.
    * **Calinski-Harabasz Index:** Higher values indicate better clustering.
    * **Davies-Bouldin Index:** Lower values indicate better clustering.
    * **Adjusted Rand Score (ARI):** A score of 1.0 indicates perfect agreement with true labels, and 0.0 indicates random assignment.
6.  **Results Storage:** The evaluation results are stored in a Pandas DataFrame and saved to a CSV file (`results_data.csv`).
7.  **Visualization:** Visualizations are generated to show original vs. reconstructed images and 2D/3D projections of the data.

---

## 📊 Results and Analysis

The `results_data.csv` file contains the performance metrics for each combination of dimensionality reduction and clustering. By comparing these scores, we can analyze:
* The effectiveness of each dimensionality reduction technique in preserving cluster structure.
* The performance of different clustering algorithms on the raw and reduced data.
* The trade-offs between training time, execution time, and clustering performance.

> **Σημείωση:** *Συμπεριέλαβε εδώ μια σύνοψη των βασικών σου ευρημάτων με βάση τα αποτελέσματα και τις οπτικοποιήσεις. Για παράδειγμα, ποιος συνδυασμός απέδωσε καλύτερα σύμφωνα με κάθε μετρική και γιατί πιστεύεις ότι συνέβη αυτό;*

---

## 📂 Code Structure

The project is implemented in a Google Colab notebook. Key sections include:
* Installation and import of libraries.
* Loading and preprocessing the Fashion-MNIST dataset.
* Helper functions for visualization and encoding.
* A centralized function for clustering and evaluation.
* Applying clustering directly to raw scaled data.
* Applying and evaluating PCA, Stacked Autoencoder, and UMAP.
* Saving results to Google Drive.

---

## 🚀 How to Run the Code

1.  Open the Colab notebook.
2.  Run the cells sequentially.
3.  Ensure you have connected your Google Drive if you want to save the results CSV.

### Dependencies
The necessary libraries are listed in the first code cell of the notebook. The main ones include:
* `umap-learn`
* `numpy`
* `pandas`
* `seaborn`
* `matplotlib`
* `tensorflow`
* `scikit-learn`

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## 📬 Contact
https://github.com/elenilou

https://www.linkedin.com/in/eleni-loula-3381a7253/
