# CS-4053: Recommender Systems Project

# Title: Anime Recommendation System
**Team Members:**
- K21-4677 Haris Ahmad
- K21-3964 Saud Jamal

This report presents an in-depth technical analysis of an anime
recommendation system developed using a dual-mode approach:
content-based filtering and neural network collaborative filtering.
Using a dataset of 4,708,211 ratings from 45,003 users over 12,294 anime
titles, this project demonstrates the practical implementation of both
classical and deep learning-based recommender algorithms.

The system is designed to address a wide range of user scenarios, from
new users with no history to long-term users with substantial
interaction data.

## System Architecture Overview

### Dual Recommendation Approach

**A. Content-Based Filtering**

-   **Uses:** Anime metadata (genre, type, number of episodes)

-   **Technique:** Cosine similarity over TF-IDF or genre vectors

-   **Strengths:**

    -   Works without user history (cold start safe)

    -   Ideal for recommending similar anime

-   **Limitation:** Limited by metadata granularity

**B. Neural Network Collaborative Filtering**

-   **Uses:** User-anime interaction history

-   **Technique:** Embedding layers for users and anime IDs

-   **Strengths:**

    -   Learns complex user-item patterns

    -   Highly personalized predictions

-   **Limitation:** Cannot handle new users/items without retraining

## Technical Stack

### Backend

-   **TensorFlow / Keras:** Neural network implementation

-   **Pandas / NumPy:** Data preprocessing and numerical operations

-   **Scikit-learn:** Data splitting, preprocessing

### Frontend

-   **Streamlit:** Lightweight web interface for real-time user
    interaction

## Neural Network Architecture

### Model Design

-   **Inputs:** User ID and Anime ID

-   **Embeddings:** 128-dim vectors for both user and anime

<img
src="./attachments/Project Report - Anime Recommendation System/media/image2.svg"
style="width:6.26806in;height:4.70139in" />

### Training Configuration

-   **Optimizer:** Adam (default settings)

-   **Loss Function:** Mean Squared Error (MSE)

-   **Evaluation Metric:** Mean Absolute Error (MAE)

-   **Batch Size:** 64

-   **Regularization:**

    -   Dropout (20%)

    -   Early stopping (patience: 5 epochs)

    -   Checkpointing for best weights

### Training Results

-   **Final Validation Loss:** 1.1061

-   **Final Validation MAE:** 0.7408

## Data Pipeline

### Data Cleaning & Preprocessing

-   **Source:** CSV files containing ratings and anime metadata

-   **Steps:**

    -   Remove invalid records

    -   Map IDs to sequential integers

    -   Split dataset (80% train / 20% test)

### Dataset Stats

-   **Total Ratings (original):** 4,708,211

-   **Ratings After Cleaning:** 4,708,204

-   **Train Samples:** 3,766,563

-   **Test Samples:** 941,641

-   **Unique Users:** 45,003

-   **Unique Anime Titles:** 12,294

### Feature Engineering

-   **User Embeddings:** Represent historical preferences

-   **Anime Embeddings:** Represent content-based characteristics

-   **Concatenated Vector:** Input to the dense layers for prediction

## Recommendation Quality

<table>
<colgroup>
<col style="width: 35%" />
<col style="width: 27%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>Metric</th>
<th>Content-Based</th>
<th>Neural Network</th>
</tr>
</thead>
<tbody>
<tr>
<td>Personalization</td>
<td>❌</td>
<td>✅</td>
</tr>
<tr>
<td>Cold-Start Friendly</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr>
<td>Genre Precision</td>
<td>✅</td>
<td>Limited</td>
</tr>
<tr>
<td>MAE (Validation)</td>
<td>–</td>
<td><strong>0.7408</strong></td>
</tr>
<tr>
<td>Preference Learning</td>
<td>No</td>
<td>Yes (via embeddings)</td>
</tr>
</tbody>
</table>

## Limitations

### Technical

-   **Cold Start:** Neural net fails for unseen users/items

-   **Scalability:** Embeddings require retraining for new data

-   **Resource Intensive:** GPU/TPU recommended for large-scale training

### Data Quality

-   **User Activity Variability:** Inactive users impact performance

-   **Metadata Dependency:** Content-based filtering limited by tags

## Conclusion

This project showcases a robust, scalable anime recommendation system
that leverages both content similarities and deep learning to deliver
relevant suggestions. The dual-model design ensures broad user coverage,
serving both new and experienced users effectively.

The neural collaborative filtering model achieves a validation MAE of
0.7408, affirming its predictive strength. Meanwhile, the thoughtful
architecture and Streamlit integration make the tool interactive and
user-friendly.

The system is well-positioned for further research and improvement,
including hybrid approaches and real-time learning pipelines.
