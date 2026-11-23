from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_inerzia_ninit(data):

 # Valori di n_init da testare
 n_init_values = [20,50,100,200,400]

 inertias = []

 for n in n_init_values:
     kmeans = KMeans(n_clusters=3,   # <-- metti qui il numero di cluster che vuoi
                     n_init=n,
                     random_state=37)
     kmeans.fit(data)
     inertias.append(kmeans.inertia_)

 # Grafico
 plt.plot(n_init_values, inertias, marker="o")
 plt.xlabel("n_init")
 plt.ylabel("Inertia")
 plt.title("Inertia al variare di n_init")
 plt.grid(True)
 plt.show() 