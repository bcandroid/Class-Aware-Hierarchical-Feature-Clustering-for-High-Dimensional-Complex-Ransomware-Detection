<img width="1536" height="1024" alt="abc" src="https://github.com/user-attachments/assets/3473c4ab-9509-4f54-8469-8cabc6b12f4c" />

## Overview

In this repository, we propose a **correlation-driven, class-aware hierarchical feature clustering framework** that primarily groups features into **ransomware-specific**, **benign-specific**, and **shared clusters**, with **model-optimized thresholds** to enhance discriminative power and reduce redundancy. This framework is implemented in the `feature_class.py` file.

The framework has been applied to a large-scale dataset for ransomware and benignware detection [1], producing a reduced, class-aware clustered dataset (`clustered_dataset.csv`) that preserves the most informative features while eliminating irrelevant or redundant ones.

## Dataset

The primary dataset used in this work is available at:  
[1] [https://github.com/bcandroid/A-Hybrid-Behavioral-Analysis-Dataset-for-Ransomware-and-Benignware-Detection](https://github.com/bcandroid/A-Hybrid-Behavioral-Analysis-Dataset-for-Ransomware-and-Benignware-Detection)

The `clustered_dataset.csv` file is the reduced version of the original dataset after applying the **class-aware hierarchical feature clustering framework**.

## Files Description

* **feature_class.py**: Python implementation of the class-aware hierarchical feature clustering framework.  
* **clustered_dataset.csv**: Reduced dataset obtained by applying the framework to the original data from [1].  
* **all_clusters.txt**: Contains subclasses with high correlation shared between ransomware and benignware samples.  
* **rans_clusters.txt**: Contains subclasses with high correlation exclusively in ransomware samples.  
* **good_clusters.txt**: Contains subclasses with high correlation exclusively in benignware samples.  
* **misclassified_file_names.txt**: Lists the file names that were misclassified when the reduced dataset was used for model training.  
* **column_names.txt**: Column names corresponding to the reduced `clustered_dataset.csv` dataset.  
* **test_rf.py**: Script for validating the performance of the reduced dataset using a Random Forest model.

**Framework Citation (BibTeX):**
If you use this dataset or framework in your research, please cite this work as follow:

@misc{caliskan2025framework,
  title={Class-Aware Hierarchical Feature Clustering for High-Dimensional Complex Ransomware Detection},
  author={B. Caliskan},
  year={2025},
  howpublished={\url{https://github.com/bcandroid/Class-Aware-Hierarchical-Feature-Clustering-for-High-Dimensional-Complex-Ransomware-Detection}},
  note={Python implementation available at this repository}
}
