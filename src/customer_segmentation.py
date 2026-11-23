# Customer segmentation
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN_DATA_PATH, PROCESSED_DIR, MODELS_DIR, RANDOM_STATE
from database.db_utils import save_table

SEGMENT_NAMES = ["Champions", "Loyal Customers", "At-Risk Customers", "Lost / Low-Value"]


def load_customer_transactions():
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate", "Date"])
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


def build_rfm(df):
    """Recency (days), Frequency (orders), Monetary (total spend) per customer."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()


def scale_rfm(rfm):
    """Log transform (skewed data) + standard scaling."""
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def silhouette_table(X, k_range=range(2, 7)):
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        rows.append({"k": k, "Silhouette": round(silhouette_score(X, labels), 3)})
    return pd.DataFrame(rows)


def name_segments(rfm):
    """Rank clusters by their own RFM averages (data-derived, not manual labels)."""
    profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    score = profile["Frequency"].rank() + profile["Monetary"].rank() - profile["Recency"].rank()
    order = score.sort_values(ascending=False).index
    mapping = {cluster: SEGMENT_NAMES[i] for i, cluster in enumerate(order)}
    return rfm["Cluster"].map(mapping)


def run_segmentation(k=4):
    df = load_customer_transactions()
    rfm = build_rfm(df)
    X, scaler = scale_rfm(rfm)

    print("Silhouette score by k:")
    print(silhouette_table(X).to_string(index=False))

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    rfm["Cluster"] = km.fit_predict(X)
    rfm["Segment"] = name_segments(rfm)

    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
    ).round(1)
    summary["Revenue_Share_%"] = (summary["Total_Revenue"] / summary["Total_Revenue"].sum() * 100).round(1)
    summary = summary.sort_values("Total_Revenue", ascending=False)

    rfm.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)
    summary.reset_index().to_csv(PROCESSED_DIR / "segment_summary.csv", index=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kmeans": km, "scaler": scaler}, MODELS_DIR / "customer_model.pkl")
    save_table(rfm, "customer_segments")

    print(f"\nCustomers segmented: {len(rfm)}")
    print(summary.to_string())
    return rfm, summary


if __name__ == "__main__":
    run_segmentation()