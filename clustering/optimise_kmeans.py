import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def optimise_kmeans(data,max_k):
  inertias=[]
  
  for k in range(1,max_k):
    kmeans=KMeans(n_clusters=k, n_init=20, random_state=37)
    kmeans.fit(data)
    
    inertias.append(kmeans.inertia_)
  
  #genera elbow plot
  fig=plt.subplots(figsize=(10,5))
  plt.plot(range(1,max_k+1),inertias,'o-')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inerzia')
  plt.grid(True)
  plt.show()