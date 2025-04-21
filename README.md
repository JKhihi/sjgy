# sjgy
数据归约
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载与预处理
data = pd.read_csv('winequality-red.csv', delimiter=';')
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
features = data.drop('quality', axis=1)
labels = data['quality']

# 标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# 优化DBSCAN参数（示例：通过k-distance图选择eps）
# 此处假设通过分析发现eps=1.2更合适
dbscan = DBSCAN(eps=1.2, min_samples=10)
clusters = dbscan.fit_predict(scaled_data)

# 聚类质量评估
noise_ratio = (clusters == -1).sum() / len(clusters) * 100
if len(set(clusters)) > 1:  # 至少需要两个非噪声簇计算轮廓系数
    silhouette = silhouette_score(scaled_data, clusters)
else:
    silhouette = -1
print(f"噪声点比例: {noise_ratio:.2f}%")
print(f"轮廓系数: {silhouette:.2f}")

# 动态调整t-SNE的perplexity
tsne = TSNE(n_components=2, perplexity=25, random_state=42)
tsne_data = tsne.fit_transform(scaled_data)

# 可视化
df_plot = pd.DataFrame(tsne_data, columns=['TSNE1', 'TSNE2'])
df_plot['Cluster'] = clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_plot, 
    x='TSNE1', 
    y='TSNE2', 
    hue='Cluster', 
    palette='viridis', 
    s=60,
    style=df_plot['Cluster'],  # 区分噪声点（可选）
    markers={-1: 'X', **{i: 'o' for i in set(clusters) if i != -1}}
)
plt.title(f't-SNE + DBSCAN 聚类 (噪声比例: {noise_ratio:.1f}%)')
plt.xlabel('t-SNE 维度1')
plt.ylabel('t-SNE 维度2')
plt.tight_layout()
plt.savefig('improved_wine_cluster.png', dpi=300)
plt.show()
