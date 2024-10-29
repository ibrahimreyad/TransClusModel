# TransClusModel: Transformer-Based Topic Modeling with Density-Based Clustering

TransClusModel is a topic modeling approach that combines pre-trained language models with density-based clustering methods like UMAP and HDBSCAN to improve topic coherence and document clustering. It is designed for handling large text datasets and provides flexibility in topic discovery.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ibrahimreyad/TransClusModel.git
   cd TransClusModel
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

**Dependencies**:
- Python 3.8 or higher
- Transformers library
- UMAP
- HDBSCAN
- Scikit-learn
- (List any other libraries you used)


## Data Preparation

TransClusModel works with various text datasets, such as Reuters-21578, 20 Newsgroups, and BBC News. Make sure the datasets are in a structured format (e.g., CSV or TXT) with one document per row. Preprocessing includes:
- Lowercasing text
- Tokenization
- Stop-word removal
- Ngram




### 5. **Project Structure**

## Project Structure
this Repository contain all attepmts to reach my model

## Results and Evaluation

The results, including topic coherence scores and clustering metrics, will be saved in the specified output directory. Check the output files to review:
- Document-topic distributions
- Topic coherence scores
- Cluster visualizations


## Limitations

TransClusModel may have performance constraints on very large datasets due to memory requirements for clustering. Additionally, topic coherence can vary with different pre-trained models, so results may not be consistent across models.






## Contact

For any questions or further information, please contact [ibrahim.elsayed@fci.bu.edu.eg].

