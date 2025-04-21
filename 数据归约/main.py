import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载数据
data = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')

# 2. 数据预处理：去掉缺失值
data = data.dropna()

# 3. 构造用于聚类的特征数据（用产品数量和价格进行聚类）
features = data[['Quantity', 'Price']]
features = features[(features > 0).all(axis=1)]  # 去除负数

# 4. 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# 5. PCA降维
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 6. 聚类分析
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# 7. 可视化 PCA + 聚类结果
df_plot = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_plot['Cluster'] = clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=60)
plt.title('PCA + KMeans 聚类图')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.tight_layout()
plt.savefig('cluster_plot.png', dpi=300)
plt.show()
