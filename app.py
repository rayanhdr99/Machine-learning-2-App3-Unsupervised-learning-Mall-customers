# app.py - streamlit app for mall customer segmentation using KMeans clustering
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# make sure Python can find our src folder
sys.path.insert(0, os.path.dirname(__file__))

# import our custom modules for loading data, preprocessing, modeling, and evaluation
from src.data_loader import load_data
from src.preprocessor import get_features
from src.model import compute_elbow, compute_silhouette, train_kmeans
from src.evaluator import evaluate_clustering

# set up logging so we can see what's happening in the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# path to our dataset and color palette for the cluster plots
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mall_customers.csv")
PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# configure the streamlit page title and layout
st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")


@st.cache_data
def get_data():
    # load the dataset and cache it so we don't reload every time
    return load_data(DATA_PATH)


@st.cache_resource
def get_model(n_clusters: int = 5):
    # train a KMeans model and cache it based on the number of clusters
    df = get_data()
    features = get_features(df)  # extract just the columns we need for clustering
    model = train_kmeans(features, n_clusters=n_clusters)
    return model, features


@st.cache_data
def get_elbow_silhouette():
    # compute elbow and silhouette data for finding the best k
    df = get_data()
    features = get_features(df)
    elbow_df = compute_elbow(features)       # WCSS for each k
    sil_df = compute_silhouette(features)    # silhouette score for each k
    return elbow_df, sil_df


def main():
    st.title("🛍️ Mall Customer Segmentation")
    st.markdown("Segment mall customers using **KMeans Clustering** based on Annual Income and Spending Score.")

    # sidebar navigation to switch between pages
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "Optimal K Analysis", "Cluster Visualisation", "Segment a Customer"])

    # try loading the data; show an error if something goes wrong
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        logger.error("Data load error: %s", e)
        return

    # ---- page 1: dataset overview ----
    if page == "Dataset Overview":
        st.header("Dataset Overview")

        # show some quick stats at the top
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", df.shape[0])
        col2.metric("Avg Annual Income", f"${df['Annual_Income'].mean():,.0f}k")
        col3.metric("Avg Spending Score", f"{df['Spending_Score'].mean():.0f}/100")

        # display the first 10 rows so the user can see what the data looks like
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        # show descriptive statistics (mean, std, min, max, etc.)
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        # create three scatter plots to explore relationships between features
        st.subheader("Pairplot")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # plot 1: income vs spending score
        axes[0].scatter(df["Annual_Income"], df["Spending_Score"], alpha=0.5, color="steelblue")
        axes[0].set_xlabel("Annual Income (k$)")
        axes[0].set_ylabel("Spending Score")
        axes[0].set_title("Income vs Spending Score")

        # plot 2: age vs income
        axes[1].scatter(df["Age"], df["Annual_Income"], alpha=0.5, color="teal")
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("Annual Income (k$)")
        axes[1].set_title("Age vs Income")

        # plot 3: age vs spending score
        axes[2].scatter(df["Age"], df["Spending_Score"], alpha=0.5, color="salmon")
        axes[2].set_xlabel("Age")
        axes[2].set_ylabel("Spending Score")
        axes[2].set_title("Age vs Spending Score")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---- page 2: elbow and silhouette analysis to find optimal k ----
    elif page == "Optimal K Analysis":
        st.header("Finding the Optimal Number of Clusters")
        try:
            elbow_df, sil_df = get_elbow_silhouette()
        except Exception as e:
            st.error(f"Failed to compute analyses: {e}")
            return

        col1, col2 = st.columns(2)

        # left column: elbow plot showing WCSS for each k
        with col1:
            st.subheader("Elbow Plot (WCSS)")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(elbow_df["k"], elbow_df["WCSS"], "bo-", linewidth=2, markersize=8)
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("WCSS (Inertia)")
            ax.set_title("Elbow Method")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(elbow_df, use_container_width=True)

        # right column: silhouette score plot to confirm the best k
        with col2:
            st.subheader("Silhouette Score Plot")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(sil_df["k"], sil_df["Silhouette_Score"], "rs-", linewidth=2, markersize=8)
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Analysis")
            ax.grid(True, alpha=0.3)

            # find the k with the highest silhouette score and mark it with a vertical line
            best_k = sil_df.loc[sil_df["Silhouette_Score"].idxmax(), "k"]
            ax.axvline(best_k, color="green", linestyle="--", label=f"Best k={best_k}")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(sil_df, use_container_width=True)

        st.info(f"Based on the Silhouette analysis, **k=5** is the optimal number of clusters.")

    # ---- page 3: cluster visualization ----
    elif page == "Cluster Visualisation":
        st.header("Customer Clusters")

        # let the user pick how many clusters to use
        n_clusters = st.slider("Number of Clusters", min_value=3, max_value=8, value=5)
        try:
            model, features = get_model(n_clusters)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return

        # add cluster labels to the dataframe for plotting
        df_plot = df.copy()
        df_plot["Cluster"] = model.labels_
        centers = model.cluster_centers_  # get the centroid coordinates

        # evaluate and display clustering quality metrics
        metrics = evaluate_clustering(model, features)
        col1, col2 = st.columns(2)
        col1.metric("Inertia (WCSS)", f"{metrics['inertia']:,.0f}")
        if metrics["silhouette_score"]:
            col2.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")

        # scatter plot: each cluster in a different color
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = PALETTE[:n_clusters]
        for i in range(n_clusters):
            cluster_data = df_plot[df_plot["Cluster"] == i]
            ax.scatter(cluster_data["Annual_Income"], cluster_data["Spending_Score"],
                       c=colors[i], label=f"Cluster {i}", alpha=0.7, s=80)

        # mark the centroids with black X markers
        ax.scatter(centers[:, 0], centers[:, 1], c="black", s=300, marker="X",
                   zorder=5, label="Centroids")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.set_title(f"Customer Segments (k={n_clusters})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # show average stats for each cluster in a table
        st.subheader("Cluster Statistics")
        cluster_stats = df_plot.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score"]].agg(["mean", "count"]).round(2)
        st.dataframe(cluster_stats, use_container_width=True)

    # ---- page 4: predict the segment for a new customer ----
    elif page == "Segment a Customer":
        st.header("Segment a New Customer")
        st.markdown("Enter customer details to find their segment.")

        # sliders for the user to input income and spending score
        col1, col2 = st.columns(2)
        with col1:
            annual_income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60)
        with col2:
            spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

        if st.button("Find Segment", type="primary"):
            try:
                # get the trained model with 5 clusters
                model, features = get_model(5)

                # predict which cluster this new customer belongs to
                input_data = np.array([[annual_income, spending_score]])
                cluster = model.predict(input_data)[0]

                # prepare data for the visualization
                df_plot = df.copy()
                df_plot["Cluster"] = model.labels_
                centers = model.cluster_centers_

                st.success(f"This customer belongs to **Cluster {cluster}**")

                # Describe the cluster
                cluster_data = df_plot[df_plot["Cluster"] == cluster]
                st.write(f"**Cluster {cluster} profile:**")
                st.write(f"- Average Annual Income: ${cluster_data['Annual_Income'].mean():.1f}k")
                st.write(f"- Average Spending Score: {cluster_data['Spending_Score'].mean():.1f}/100")
                st.write(f"- Number of customers in cluster: {len(cluster_data)}")

                # plot all clusters and highlight where the new customer falls
                fig, ax = plt.subplots(figsize=(8, 5))
                for i in range(5):
                    cd = df_plot[df_plot["Cluster"] == i]
                    ax.scatter(cd["Annual_Income"], cd["Spending_Score"],
                               c=PALETTE[i], alpha=0.5, s=60, label=f"Cluster {i}")

                # show centroids and the new customer's position as a yellow star
                ax.scatter(centers[:, 0], centers[:, 1], c="black", s=300, marker="X", zorder=5, label="Centroids")
                ax.scatter(annual_income, spending_score, c="yellow", s=400, marker="*",
                           edgecolors="black", zorder=10, label="Your Customer")
                ax.set_xlabel("Annual Income (k$)")
                ax.set_ylabel("Spending Score")
                ax.set_title("Customer Position in Segments")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                logger.error("Segmentation error: %s", e)


if __name__ == "__main__":
    main()
