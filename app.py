import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import plotly.express as px

st.title("🛍️ 3D Customer Segmentation Dashboard")
st.markdown("K-Means ile müşteri segmentasyonu (3D görselleştirme)")


df = pd.read_csv("Mall_Customers.csv")



st.sidebar.title("Ayarlar")
cluster_sayisi = st.sidebar.slider("Cluster sayısı", 2, 10, 5)


X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]


kmeans = KMeans(n_clusters=cluster_sayisi, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)


st.subheader("📊 Veri")
st.write(df)

st.subheader("📌 Cluster Analizi")
st.write(df.groupby("Cluster").mean(numeric_only=True))


st.subheader("🌐 3D Görselleştirme")

fig = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster",
    title="Customer Segments (3D K-Means)"
)

st.plotly_chart(fig)