import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyproj

def plot_clustering(temperature, colonna_cluster):
# colonna_cluster è una stringa, nome della colonna che contiene i cluster da visualizzare

    # Converti in GeoDataFrame
    gdf = gpd.GeoDataFrame(
        temperature,
        geometry=gpd.points_from_xy(temperature['LONGITUDINE'], temperature['LATITUDINE']),
        crs='EPSG:4326'
    )

    # Proietta su Web Mercator per contextily
    gdf = gdf.to_crs(epsg=3857)

    # Crea mappa
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax,
             c=gdf[colonna_cluster],
             markersize=100,
             edgecolor='black',
             linewidth=0.5)

    # Aggiungi etichette con i nomi delle località (indice del dataframe)
    for idx, row in gdf.iterrows():
        
         # offset opzionale in x e y
        dx, dy = 0, 1000

         # esempio di personalizzazione
        if idx == 'BUSTO ARSIZIO':
            dx = 1000
        elif idx == 'CANTÙ ASNAGO':
            ha = 'left'
            dx, dy= 2000, -1000
        elif idx == 'MARIANO COMENSE':
            va = 'top'
            dx, dy = 2000,-1000
        elif idx == 'CUVEGLIO':
            va = 'top'
            dy=-1000
        elif idx == 'FERNO':
            va = 'top'
            dy=-1000
        elif idx == 'LURAGO MARINONE':
            va = 'top'
            dy=-1000
        elif idx == 'MILANO LAMBRATE':
            dx, dy = -2000, 4000
        else:
            ha, va = 'center', 'bottom' 
            
        ax.text(row.geometry.x + dx,
                row.geometry.y + dy,
                s=idx, # il nome della località è nell'indice
                fontsize=9,
                color='black',
                va=va,
                ha=ha)

    # Aggiunge base map
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    

    # Migliora aspetto
    ax.set_title("Mappa delle località divise per cluster")
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")

    # Trasforma da EPSG:3857 a EPSG:4326 per formattazione
    proj = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def format_lon(x, pos=None):
        lon, _ = proj.transform(x, gdf.geometry.y.mean())
        return f"{lon:.2f}°"

    def format_lat(y, pos=None):
        _, lat = proj.transform(gdf.geometry.x.mean(), y)
        return f"{lat:.2f}°"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

    xmin, ymin, xmax, ymax = gdf.total_bounds
    
    ax.set_xlim(xmin-12000, xmax+12000)
    ax.set_ylim(ymin-5000, ymax+5000)
    ax.set_axis_off()  # toglie assi

    plt.show()
