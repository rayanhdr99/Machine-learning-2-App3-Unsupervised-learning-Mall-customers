# Mall Customer Segmentation

An unsupervised machine learning project that applies **KMeans clustering** to segment 200 mall customers into meaningful groups based on their Annual Income and Spending Score.

---

## Project Description

Retailers need to understand customer behaviour to personalise marketing strategies. By clustering customers with similar income and spending patterns, this project identifies five distinct customer personas that empower marketing teams to craft targeted campaigns, personalise offers, and allocate resources more effectively.

Key facts:
- **Algorithm**: KMeans (unsupervised learning)
- **Dataset**: 200 mall customers with no missing values
- **Optimal k**: 5, selected via the Elbow Method (WCSS) and Silhouette Analysis over k = 3 to 8
- **Features used for clustering**: `Annual_Income` and `Spending_Score`
- **Interface**: Interactive Streamlit web application with four pages

---

## Project Structure

```
customer_segmentation/
|
|-- app.py                  # Streamlit application entry point
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
|
|-- data/
|   +-- mall_customers.xls  # Raw dataset (200 customers, Excel format)
|
+-- src/
    |-- __init__.py         # Source package initialiser
    |-- data_loader.py      # Excel loading (pd.read_excel) and column validation
    |-- preprocessor.py     # Feature extraction for clustering
    |-- model.py            # KMeans training, Elbow and Silhouette evaluation
    +-- evaluator.py        # Inertia and silhouette score computation
```

---

## Setup and Running Instructions

### Prerequisites

- Python 3.10 or later
- The dataset file `data/mall_customers.xls` must exist at the path shown above

### 1. Navigate to the project directory

```bash
cd "c:/Users/rayan/Desktop/machine learning project/customer_segmentation"
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## Algorithm

### KMeans Clustering

KMeans partitions n observations into k clusters by iteratively minimising the Within-Cluster Sum of Squares (WCSS / inertia). Each customer is assigned to the cluster whose centroid is nearest in feature space.

**Final model configuration:**

```python
KMeans(
    n_clusters=5,
    init='k-means++',   # Smart centroid initialisation reduces poor local minima
    n_init='auto',      # Scikit-learn auto-determines the number of runs
    random_state=42     # Reproducible results
)
```

### Choosing k = 5

Both the Elbow Method and Silhouette Analysis were applied across k = 3 to 8:

| Method | What it measures | Result |
|--------|-----------------|--------|
| **Elbow Method (WCSS)** | Plots inertia vs k; the "elbow" marks diminishing returns | Elbow clearly at k = 5 |
| **Silhouette Score** | Cluster cohesion vs separation; range -1 to 1, higher is better | Peak score at k = 5 |

Both methods independently confirm **k = 5** as optimal for the Annual Income + Spending Score feature space.

---

## Features Used

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `Annual_Income` | Customer annual income in thousands of dollars | 15k – 137k |
| `Spending_Score` | Mall-assigned behavioural spending score | 1 – 100 |

`Age` and `Gender` are present in the dataset and available via the `df_3feat` output of `prepare_features()` in `preprocessor.py`, but the primary model uses the two-feature subset because it produces the most interpretable and well-separated clusters.

---

## Results: 5 Meaningful Customer Segments

The KMeans model with k = 5 discovers the following customer profiles:

| Cluster | Approximate Income | Approximate Score | Segment Name |
|---------|--------------------|-------------------|--------------|
| 0 | Medium | Low | Careful Spenders — mid-income, budget-conscious |
| 1 | Low | High | Impulsive Buyers — lower income but high spending |
| 2 | Medium | Medium | Standard Customers — average across both axes |
| 3 | High | High | Target Customers — ideal segment; high income and willingness to spend |
| 4 | High | Low | Conservative Savers — high earners who spend cautiously |

> Exact cluster numbering may vary; the shapes and separations are consistent across runs due to `random_state=42`.

**Model performance (k = 5, 2-feature model):**
- Silhouette Score: approximately 0.55
- WCSS (Inertia): approximately 44,448

---

## Application Pages

| Page | Description |
|------|-------------|
| **Dataset Overview** | Total customers, average income, average spending score, sample rows, statistical summary, and pairplots |
| **Optimal K Analysis** | Elbow plot (WCSS) and Silhouette score chart for k = 3 to 8, with score tables |
| **Cluster Visualisation** | Scatter plot with colour-coded clusters and centroids; adjustable k slider (3-8); cluster statistics table |
| **Segment a Customer** | Sliders for Annual Income and Spending Score; predicts cluster and displays customer position on the scatter plot |

---

## Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| streamlit | 1.32.0 | Interactive web application framework |
| pandas | 2.0.0 | Data loading and manipulation |
| numpy | 1.24.0 | Numerical operations |
| scikit-learn | 1.3.0 | KMeans algorithm and silhouette metric |
| matplotlib | 3.7.0 | Plots and visualisations |
| seaborn | 0.12.0 | Statistical plot styling |

---

## License

This project is for educational and portfolio purposes.
