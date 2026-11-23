import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Funzione che categorizza le temperature
def categorizza_temp(temp, soglie, etichette):
    try:
        temp = float(temp)
    except:
        return 'missing'
    for i, soglia in enumerate(soglie):
        if temp < soglia:
            return etichette[i]
    return etichette[-1]   # Per i valori >= di tutte le soglie viene assegnata l'ultima etichetta
# NB: in Python, l’indice -1 fa riferimento all’ultima posizione di una lista


# Funzione che seleziona le regole del tipo temperatura_vecchia -> temperatura nuova
def is_old_to_new(antecedents, consequents):
    return all('OLD' in a for a in antecedents) and all('NEW' in c for c in consequents)
    #Se tutti gli item nell'antecedente contengono la parola OLD e tutti quelli nel conseguente contengono NEW ritorna True, altrimenti False.
    

# Funzione che genera le regole associative
def genera_regole(soglie, etichette, supporto_min, confidenza_min, df_vecchie, df_nuove):
    
    # 1. Categorizzazione di tutte le colonne dei due dataset (esclusa le prime tre contenente i nomi delle località, latitudine e longitudine)
    # Creo due nuove copie dei DataFrame originali: questo serve per lavorare sulle copie, mantenendo intatti i dataset originali.
    df_vc = df_vecchie.copy()
    df_nc = df_nuove.copy()
    # Per ogni colonna del DataFrame si applica la funzione categorizza_temp a ciascun valore di quella colonna usando il metodo .apply; il risultato
    # categorizzato viene assegnato alla stessa colonna corrispondente nella copia.
    for col in df_vc.columns[3:]:
        df_vc[col] = df_vc[col].apply(lambda x: categorizza_temp(x, soglie, etichette))
        df_nc[col] = df_nc[col].apply(lambda x: categorizza_temp(x, soglie, etichette))

    # 2. Creazione delle transazioni
    transazioni = []
    for i in range(len(df_vc)):
        localita = df_vc.iloc[i, 0]   # località contenute nella prima colonna del dataset
        # NB: .iloc permette di selezionare righe e colonne in un DataFrame usando posizioni numeriche
        for giorno in range(356):   #num giorni comuni per entrambi gli anni = 356 = 365-9 giorni tolti dai dataset 
            #Temperature minime
            min_v = df_vc.iloc[i, 3 + giorno]
            min_n = df_nc.iloc[i, 3 + giorno]
            #Temperature massime
            max_v = df_vc.iloc[i, 3 + 356 + giorno]
            max_n = df_nc.iloc[i, 3 + 356 + giorno]
            # Creazione di una lista chiamata 'transazione' con quattro stringhe formattate (f-string), ciascuna delle quali rappresenta la categoria di
            # temperatura preceduta da un prefisso che indica l'anno di appartenenza e la tipologia di misurazione
            transazione = [
                f"OLD_min_{min_v}",
                f"NEW_min_{min_n}",
                f"OLD_max_{max_v}",
                f"NEW_max_{max_n}"
            ]
            # Aggiunta alla lista 'transazioni' di un dizionario che ha due chiavi:
            # -'localita': nome della località
            # -'dati': lista 'transazione' appena creata.
            transazioni.append({'localita': localita, 'dati': transazione})
    # Creazione di due nuove liste:
    transazioni_dati = [t['dati'] for t in transazioni]    #categorie di temperatura
    transazioni_localita = [t['localita'] for t in transazioni]   #nomi delle località
    # print(len(transazioni_dati))
    # print(len(transazioni_localita))
    # 9968 = 356 * 28     OK: una transazione per ogni giorno e per ogni località

    # 3. One-hot encoding delle transazioni: trasformazione delle transazioni in un dataset binario per poter applicare l'algoritmo Aprior
    te = TransactionEncoder()
    te_array = te.fit(transazioni_dati).transform(transazioni_dati)
    # -fit() scansiona tutte le transazioni per identificare gli item univoci
    # -transform() costruisce una matrice binaria  in cui ogni riga rappresenta una transazione, mentre ogni colonnaidentifica un item; 
    # il valore è True se l’item è presente nella transazione, False altrimenti.
    # print(type(te_array)) -> <class 'numpy.ndarray'>
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    # pd.DataFrame() converte la matrice te_array in un DataFrame Pandas, assegnando come nomi delle colonne i nomi degli item (te.columns_)
    # Aggiungo al DataFrame la colonna 'localita' per sapere a quale localita si riferisce ciascuna transazione.
    # Oss:  la lista transazioni_localita ha la stessa lunghezza di transazioni_dati, quindi l'aggiunta è coerente.
    df_encoded['localita'] = transazioni_localita

    # 4. Applicazione di Apriori sul DataFrame codificato binariamente per identificare gli itemset frequenti
    frequent_itemsets = apriori(df_encoded.drop(columns=['localita']), min_support=supporto_min, use_colnames=True)
    # -viene esclusa dal dataset la colonna 'localita';
    # -min_support: supporto minimo richiesto per mantenere un itemset
    # -use_colnames: se True, mostra i nomi delle colonne invece degli indici
    # -frequent_itemsets (output): DataFrame con due colonne: 1)itemsets: itemset frequenti
    #                                                         2)support: supporto dei corrispondenti itemset frequenti

    # 5. Calcolo delle regole di associazione a partire dagli itemset frequenti appena calcolati
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=confidenza_min)
    # -frequent_itemsets: DataFrame appena creato con l'algoritmo Apriori
    # -metric: metrica da usare per selezionare le regole, in questo caso la confidenza
    # -min:treshold: valore minimo richiesto per la metrica scelta.
    
    # 6. Selezione delle regole del tipo temperatura_vecchia -> temperatura_nuova
    rules = rules[rules.apply(lambda row: is_old_to_new(row['antecedents'], row['consequents']), axis=1)]
    # lambda row: funzione anonima che a sua volta chiama su ogni riga del DataFrame rules la funzione is_old_to_new: se is_old_to_new restituisce True, quella     # riga viene tenuta, altrimenti viene scartata.
    # NB: In Pandas: df[condizione_booleana] è il modo standard per filtrare un DataFrame. L'oggetto tra parentesi quadre ([...]) è una serie di True/False,        # uno per ogni riga, e Pandas seleziona solo le righe corrispondenti ai True.
    return frequent_itemsets, rules


# Funzione che estrae la media delle metriche considerate per ogni granularità
def estrai_metriche(rules):
    return {
        'SUPPORTO': rules['support'].mean(),
        'CONFIDENZA': rules['confidence'].mean(),
        'LIFT': rules['lift'].mean(),
        'LEVERAGE': rules['leverage'].mean(),
        'CONVICTION': rules['conviction'].mean(),
        'METRICA DI ZHANG': rules['zhangs_metric'].mean()  
    }

# Funzione che plotta l'andamento delle metriche al variare della granularità
def plot_metriche(soglie1, etichette1, soglie2, etichette2, soglie3, etichette3, soglie4, etichette4, soglie5, etichette5, supporto_min, confidenza_min, 
                  df_vecchie, df_nuove, save_path):
    # Genero regole di associazione con granularità diverse
    frequent_itemsets1, regole1 = genera_regole(soglie1, etichette1, supporto_min, confidenza_min, df_vecchie, df_nuove)
    frequent_itemsets2, regole2 = genera_regole(soglie2, etichette2, supporto_min, confidenza_min, df_vecchie, df_nuove)
    frequent_itemsets3, regole3 = genera_regole(soglie3, etichette3, supporto_min, confidenza_min, df_vecchie, df_nuove)
    frequent_itemsets4, regole4 = genera_regole(soglie4, etichette4, supporto_min, confidenza_min, df_vecchie, df_nuove)
    frequent_itemsets5, regole5 = genera_regole(soglie5, etichette5, supporto_min, confidenza_min, df_vecchie, df_nuove)

    # Creo un DataFrame riassuntivo contenente le medie delle metriche per ogni granularità
    metriche = []
    granularita = ['soglie1', 'soglie2', 'soglie3', 'soglie4', 'soglie5']  #nomi delle diverse granularità
    regole_list = [regole1, regole2, regole3, regole4, regole5]  #regole di associazione generate con le granularità diverse
    for reg in regole_list:
        metriche.append(estrai_metriche(reg))
    df_metriche = pd.DataFrame(metriche, index=granularita) 
    # colonna -> metrica; riga -> granularità

    # Plot delle metriche in funzione della granularità
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))  
    axs = axs.flatten()

    for i, col in enumerate(df_metriche.columns): #i = indice colonna, col = nome metrica 
        axs[i].plot(df_metriche.index, df_metriche[col], marker='o')
        axs[i].set_title(col)
        axs[i].set_xlabel('Granularità')
        axs[i].set_ylabel(col)
        axs[i].grid(True)

    plt.tight_layout()  #regola gli spazi tra i subplot per evitare sovrapposizioni
    plt.tight_layout()
    plt.savefig(save_path)  # salva il grafico in PNG
    plt.show()