#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:47:22 2024

@author: Coraly Gervasoni & Perrine Martin 
"""
##############################################################################
########################### BIBLIOTHÈQUES ####################################
##############################################################################

# Importations pour le traitement et l'analyse de texte
from sklearn.feature_extraction.text import TfidfVectorizer  # Vectorisation des textes avec TF-IDF
from stop_words import get_stop_words  # Obtenir les stopwords pour le français

# Importations pour les opérations numériques et la manipulation de données
import pandas as pd  # Manipulation de DataFrame
import numpy as np  # Calculs numériques
import random  # Génération de nombres aléatoires
from scipy.sparse import find, csr_matrix  # Manipulation de matrices creuses
from scipy.linalg import norm  # Calcul de la norme d'un vecteur
from collections import Counter, defaultdict  # Structures de données pour compter et avec valeurs par défaut

# Importations pour le travail avec les graphes
import networkx as nx  # Manipulation et analyse de graphes
import igraph as ig  # Utilisé pour certaines méthodes de clustering avancées

# Importations pour le clustering et la détection de communautés
import community as community_louvain  # Pour le clustering Louvain
from sklearn.cluster import SpectralClustering  # Clustering spectral

# Importations pour la visualisation
import matplotlib.pyplot as plt  # Création de graphiques
import seaborn as sns  # Amélioration esthétique des graphiques
from matplotlib.colors import LinearSegmentedColormap # Personnalisation des cartes de couleurs
from matplotlib import cm  # Cartes de couleurs

# Importations pour l'analyse spectrale et le clustering
from scipy.sparse.linalg import eigsh  # Pour trouver les valeurs propres des matrices creuses
from sklearn.manifold import SpectralEmbedding  # Pour l'embedding spectral

# Importations pour le Machine Learning et l'évaluation des modèles
from sklearn.model_selection import train_test_split  # Division des données en ensembles d'entraînement et de test
from sklearn.ensemble import RandomForestClassifier  # Classifier de forêt aléatoire
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score, precision_recall_fscore_support, confusion_matrix  # Métriques d'évaluation

# Importations pour le Deep Learning avec PyTorch et PyTorch Geometric
import torch  # Calcul tensoriel et réseaux de neurones
from torch_geometric.nn import GCNConv, GATConv  # Couches de convolution pour les graphes
from torch_geometric.utils import from_networkx  # Conversion de NetworkX à PyTorch Geometric
from torch_geometric.data import Data  # Structure de données pour les graphes dans PyTorch Geometric
import torch.nn.functional as F  # Fonctions non-linéaires et utilitaires

# Autres utilitaires
import time  # Mesure du temps d'exécution
from tqdm import tqdm  # Barre de progression pour les boucles
import pickle  # Sauvegarde et chargement des objets Python

##############################################################################
############################## Import des données ############################
##############################################################################

#https://www.kaggle.com/code/mpwolke/load-pickle-file

# Import du jeu de données
with open('20240125_dataset_rscir_xxs.pickle', 'rb') as f:
    data = pickle.load(f)

# Affichage du jeu de données
print(data)

# Dimensions
data.shape

##############################################################################
######################### Textes en français uniquement ######################
##############################################################################

# Nombre de documents en fonction de la langue 
data['dcterms:language{Literal}'].value_counts()

# On garde que les documents en français (57411/59529)
data = data[data['dcterms:language{Literal}'] == 'fr']    
data.shape

##############################################################################
############################ Valeurs manquantes ##############################
##############################################################################

# Nombre de valeurs manquantes par column
valeurs_manquantes_par_column = data.isnull().sum()

# Créer un nouveau dataframe avec les column qui ont moins de 50 000 valeurs manquantes
columns_a_garder = valeurs_manquantes_par_column[valeurs_manquantes_par_column < 50000].index
nouveau_df = data[columns_a_garder] 

# Nombre de valeurs manquantes pour les columns sur le résumé des documents
columns_abstract = [col for col in valeurs_manquantes_par_column.index if 'abstract' in col]
valeurs_manquantes_par_column.loc[columns_abstract]


columns_subject = [col for col in valeurs_manquantes_par_column.index if 'subject' in col]
valeurs_manquantes_par_column.loc[columns_subject]

#Celle qui contient les moins de valeurs manquantes est dcterms:subject{Literal}[0]@en (57 123 sur 59 529, 4%) suivi de dcterms:abstract{Literal}@fr
            
#### Nombre de valeurs manquantes pour chaque column renseignant les auteurs.
# Pour 5729 documents, aucun auteur n'est renseigné

columns_marcrel_aut = [column for column in valeurs_manquantes_par_column.index if 'marcrel:aut{URIRef}' in column]
valeurs_manquantes_marcrel_aut = valeurs_manquantes_par_column[columns_marcrel_aut]
print(valeurs_manquantes_marcrel_aut)

##############################################################################
################# Suppression des documents sans auteurs renseignés ##########
##############################################################################

data = data.dropna(subset=['marcrel:aut{URIRef}[0]'])
data.shape

##############################################################################
################# Extraction de la catégorie des documents ###################
##############################################################################

# Récupération des catégories
categorie = data.index.str.extract(r'doc/(.*?)(?=_)')
# Rennomage de la column
categorie = categorie.rename(columns={0: 'Categorie'})
#Stockage des index dans l'objet index 
index= data.index
# Supression des index de data pour qu'ils aient les même index que categorie
data.reset_index(drop=True, inplace=True)
# Concaténation de la base et des catégories
data = pd.concat([data, categorie], axis=1)

##############################################################################
##############################################################################
############################# Fonction 1 #####################################
##############################################################################
##############################################################################

print("Nombre de documents : ", data.shape[0])

data['bibo:numPages{Literal}'] = pd.to_numeric(data['bibo:numPages{Literal}'], errors='coerce')
print("Nombre de page moyen par document :",  round(data['bibo:numPages{Literal}'].mean()))

all_authors = set()
for col in data.columns:
    if 'marcrel:aut{URIRef}' in col:
        all_authors.update(data[col].dropna().tolist())
print("Nombre d'auteurs :", len(all_authors))

# Nombre de documents par auteur
documents_per_author = defaultdict(int)
for col in data.columns:
    if 'marcrel:aut{URIRef}' in col:
        authors = data[col].dropna().tolist()
        for author in authors:
            documents_per_author[author] += 1


for author, num_documents in documents_per_author.items():
    print(f"Auteur : {author}, Nombre de documents : {num_documents}")

#############################################################################
#### Graphiques : distribution du nombre d'auteurs en fonction du nombre de documents qu'ils ont écrits
   
# Pour plus de visibilité (nombre d'auteurs en logarithme)
authors_per_document_count = defaultdict(int)
for num_documents in documents_per_author.values():
    authors_per_document_count[num_documents] += 1
# Extraction des données pour le graphique
document_counts = list(authors_per_document_count.keys())
author_counts = list(authors_per_document_count.values())
# Création du graphique à barres
plt.bar(document_counts, author_counts, color='skyblue')
plt.xlabel('Nombre de documents')
plt.ylabel("Nombre d'auteurs")
plt.title("Fréquence des auteurs par nombre de documents écrits")
plt.show()

# Nombre d'auteurs en logarithme pour plus de visibilité 
plt.bar(document_counts, author_counts, color='red')
plt.yscale('log')  # Utilisation d'une échelle logarithmique pour l'axe y
plt.xlabel('Nombre de documents')
plt.ylabel("Logarithme du nombre d'auteurs")
plt.title("Fréquence des auteurs (log) par nombre de documents écrits")
plt.show()

#############################################################################
#### Graphiques : Nombre de documents par date de publications

publication_dates = data['persee:dateOfPrintPublication{Literal}(xsd:gYear)'].dropna()
publication_years = pd.to_numeric(publication_dates.str[:4])
bins = range(1920, 2030, 10)
publication_years_counts = pd.cut(publication_years, bins=bins, right=False).value_counts().sort_index()
publication_years_counts.plot(kind='bar')
plt.xlabel('Tranches de 10 ans')
plt.ylabel('Fréquence')
plt.title('Distribution des documents par tranche de 10 ans des dates de publication')
plt.show()

#############################################################################
#### Nombre de documents par catégorie

categorie_counts = data['Categorie'].value_counts()
print(categorie_counts)

categorie_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Catégorie')
plt.ylabel('Nombre de documents')
plt.title('Distribution des catégories')
plt.show()

#############################################################################
#### Renommage des collections_id en discinpline 

categories = {
    'thlou': 'Religion théologie',
    'rscir': 'Religion théologie',
    'tiers': 'Science politique',
    'syria': 'Histoire',
    'slave': 'Études générales et culturelles',
    'xxs': 'Histoire',
    'rural': 'Pluri. Par essence',
    'sotra': 'Sociologie',
    'scrip': 'Littérature',
    'spgeo': 'Science environnementale',
    'rvart': 'Arts',
    'xvii': 'Études générales et culturelles',
    'topoi': 'Histoire',
    'spira': 'Sciences de l\'éducation',
    'simon': 'Littérature',
    'sosan': 'Sociologie',
    'russe': 'Littérature',
    'vita': 'Études classiques',
    'sracf': 'Études régionales',
    'sorci': 'Sociologie',
    'stice': 'Sciences de l\'éducation',
    'shmes': 'Histoire',
    'tigr': 'Géographie',
    'vilpa': 'Géographie',
    'socco': 'Sociologie',
    'tcfdi': 'Droit',
    'vibra': 'Sociologie',
    'versa': 'Histoire',
    'sfhom': 'Histoire',
    'tlgpa': 'Sciences de la Terre',
    'xvi': 'Littérature',
    'salam': 'Archéologie (Moyen Âge)'
}

categorie_counts_grouped = categorie_counts.groupby(categories).sum()

# Création d'un graphique à barres
plt.figure(figsize=(10, 6))
categorie_counts_grouped.sort_values(ascending=False).plot(kind='bar')
plt.title('Nombre de documents par catégorie')
plt.xlabel('Catégorie')
plt.ylabel('Nombre de documents')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Remplacement des collection_id par les discipline
data['Categorie'] = data['Categorie'].replace(categories)

#############################################################################
### Nombre de document écrit par auteur et catégorie : pour le clustering et la classification

# Sélectionn des colonnes contenant "marcrel:aut"
columns_author = [column for column in data.columns if 'marcrel:aut' in column]

documents_by_author_categorie = pd.melt(data, id_vars=['Categorie'], value_vars= columns_author , value_name='Auteur')
documents_by_author_categorie = documents_by_author_categorie.dropna(subset=['Auteur'])
documents_by_author_categorie = documents_by_author_categorie.pivot_table(index='Auteur', columns='Categorie', aggfunc='size', fill_value=0)

# Calcul du nombre de documents écrit pour chaque auteur
documents_by_author = documents_by_author_categorie.sum(axis=1)

# Garder les auteurs qui ont écrit au moins 5 documents
documents_by_author_categorie_5 = documents_by_author_categorie[documents_by_author > 4]

### Attribution d'une catégorie à chaque auteur 

# Trouver le nom de la column où chaque auteur atteint son maximum
Categorie_Author = documents_by_author_categorie_5.idxmax(axis=1)
Categorie_Author = pd.DataFrame({'Categorie': Categorie_Author})
print(Categorie_Author.shape)

##############################################################################

### On combine les colonnes sur les auteurs
Label_Author = pd.melt(data, id_vars=['rdfs:label{Literal}'], value_vars= columns_author , value_name='Auteur')

### Suppression des valeurs vides dans Auteur
Label_Author = Label_Author.dropna(subset=['Auteur'])

# Concaténation de tous les labels par auteur
Label_Author = Label_Author.groupby('Auteur')['rdfs:label{Literal}'].apply(lambda x: '. '.join(x)).reset_index()

# Utilisation de la variable Auteur comme index et non plus comme colonne
Label_Author = Label_Author.set_index('Auteur')

# Garder que les auteurs qui ont écrit au moins 5 documents
Label_Author = Label_Author.reindex(Categorie_Author.index)

##############################################################################
### Fusionner les deux matrices sur les catégorie et les labels par auteur 

Categorie_Label_Author = pd.merge(Categorie_Author, Label_Author, left_index=True, right_index=True)

##############################################################################
##############################################################################
############################# Fonction 2 #####################################
##############################################################################
##############################################################################

##############################################################################
################## Matrice d'adjacence avec les auteurs ######################
##############################################################################

data_subset = data#.head(1000)

# On récupère tous les auteurs uniques dans le sous-ensemble de données
all_authors_list = list(all_authors)

# MATRICE D'ADJACENCE (non pondérée)
adjacency_matrix = pd.DataFrame(0, index=all_authors_list, columns=all_authors_list)

# On parcourt chaque ligne du sous-ensemble de données
for _, row in data_subset.iterrows():
    authors = [row[col] for col in data_subset.columns if 'marcrel:aut{URIRef}' in col and not pd.isna(row[col])]
    # Pour chaque paire d'auteurs, mettre la valeur 1 dans la matrice d'adjacence
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            adjacency_matrix.loc[authors[i], authors[j]] = 1 #+= 1 si on veut les poids
            adjacency_matrix.loc[authors[j], authors[i]] = 1 #+= 1 si on veut les poids

print(adjacency_matrix)

##############################################################################
################################### Graphe ###################################
##############################################################################

#### Convertir la matrice d'adjacence en graphe
G = nx.from_pandas_adjacency(adjacency_matrix)

#########################
#### Composantes connexes
#########################

num_connected_components = nx.number_connected_components(G)
print("Nombre de graphes connexes :", num_connected_components)
    
connected_components = list(nx.connected_components(G))
component_sizes = [len(component) for component in connected_components]
print("Distribution du nombre de nœuds dans les graphes connexes :")
for size in set(component_sizes):
    count = component_sizes.count(size)
    print(f"Nombre de composantes de taille {size} :", count)

################
## Visualisation de la distribution des composantes connexes
################
    
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].hist(component_sizes, bins = 20,  color='skyblue')  
axs[0].set_title("Distribution du nombre de \n nœuds dans les graphes connexes")
axs[0].set_xlabel("Taille de la composante")
axs[0].set_ylabel("Nombre de composantes")
axs[0].grid(True)

axs[1].hist(component_sizes, bins=20, log=True, color='salmon')  
axs[1].set_title("Distribution du nombre de \n nœuds dans les graphes connexes (échelle logarithmique)")
axs[1].set_xlabel("Taille de la composante")
axs[1].set_ylabel("Nombre de composantes (log)")
axs[1].grid(True)

plt.tight_layout()
plt.show()

df_component_sizes = pd.DataFrame(component_sizes, columns=['Taille des composants'])
distribution_component_size = df_component_sizes['Taille des composants'].value_counts().reset_index()
distribution_component_size.columns = ['Taille des composants', 'Effectif']
distribution_component_size = distribution_component_size.sort_values(by='Taille des composants')

# Traçage de la courbe
plt.plot(distribution_component_size['Taille des composants'], distribution_component_size['Effectif'], marker='o', linestyle='-')

# Ajouter des étiquettes et un titre
plt.xlabel('Taille des composants')
plt.ylabel('Effectif')
plt.title('Distribution des tailles des composants')

# Affichage de la grille
plt.grid(True)

# Affichage de la courbe
plt.show()

###############################
#### Densité
###############################  
    
A = nx.to_scipy_sparse_array(G) 
A.todense()
plt.spy(A, markersize=0.5)    

################################
#### Largeur
################################

def graph_width(graph):
    # Utilisation de l'algorithme de Floyd-Warshall pour trouver tous les chemins les plus courts
    all_shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
    
    max_width = 0
    # Parcours de tous les chemins les plus courts pour trouver la largeur maximale
    for source, paths in all_shortest_paths.items():
        for target, length in paths.items():
            if source != target:
                # La largeur est la longueur maximale des chemins les plus courts
                max_width = max(max_width, length)
    
    return max_width

# Calcul de la largeur maximale des chemins les plus court du graphe G
largeur_graphe = graph_width(G)
print("La largeur du graphe est :", largeur_graphe)

################################
#### Degré
################################
    
#### Liens des auteurs qui ont écrit au moins un document ensemble (arrête)     
rows, cols = np.where(np.triu(adjacency_matrix.values) > 0)  # Utiliser np.triu pour récupérer uniquement la moitié supérieure de la matrice
edges = [(adjacency_matrix.index[i], adjacency_matrix.columns[j]) for i, j in zip(rows, cols)]

# Ajout des arêtes au graphe
G.add_edges_from(edges)

#### Informations sur le graphe
print(nx.info(G))

#### degrés des noeuds : Nombres d'auteurs avec qui a écrit chaque auteur
G.degree()
d_degree = dict(G.degree())

# Collect de tous les nœuds uniques à partir des arêtes
unique_nodes = set()
for edge in edges:
    unique_nodes.update(edge)

# Ajout des nœuds supplémentaires qui ne sont pas dans les arêtes
additional_nodes = set(d_degree.keys()) - unique_nodes
unique_nodes.update(additional_nodes)

# Compter le nombre d'occurrences de chaque arête
edge_counts = Counter(edges)

# Initialisation d'un dictionnaire pour stocker les degrés recalculés
recomputed_degree = {node: 0 for node in unique_nodes}

# Calcul des degré de chaque nœud en tenant compte du nombre d'occurrences de chaque arête
for edge, count in edge_counts.items():
    for node in edge:
        recomputed_degree[node] += count

# Récupération des valeurs des degrés recalculés
degree_values = list(recomputed_degree.values())

# Calcul du nombre de modalités différentes
num_unique_values = len(set(degree_values))

################################
# Nombre d'auteurs avec qui chaque auteur a écrit un document
################################

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].hist(recomputed_degree.values(), bins=num_unique_values)
axs[0].set_title('Distribution du nombre d\'auteurs avec \n qui ont écrit les auteurs')
axs[0].set_xlabel('Nombre d\'auteurs avec qui un auteur a écrit')
axs[0].set_ylabel('Nombre d\'auteurs')

axs[1].hist(recomputed_degree.values(), bins=num_unique_values, log=True)
axs[1].set_title('Distribution logarithmique du nombre d\'auteurs avec \n qui ont écrit les auteurs')
axs[1].set_xlabel('Nombre d\'auteurs avec qui un auteur a écrit')
axs[1].set_ylabel('Nombre d\'auteurs (échelle logarithmique)')

plt.show()

df_recomputed_degree = pd.DataFrame(recomputed_degree.values(), columns=['Degré des auteurs'])
df_recomputed_degree = df_recomputed_degree['Degré des auteurs'].value_counts().reset_index()
df_recomputed_degree.columns = ['Degré des auteurs', 'Effectif']
df_recomputed_degree = df_recomputed_degree.sort_values(by='Degré des auteurs')

# Courbe
plt.plot(df_recomputed_degree['Degré des auteurs'], df_recomputed_degree['Effectif'], marker='o', linestyle='-')
plt.xlabel('Degré des auteurs')
plt.ylabel('Effectif')
plt.title('Distribution des degré des auteurs')
plt.grid(True)
plt.show()

################################
# Afficher le graphe mais uniquement avec les auteurs qui ont écrit au moins un documents en
# commun avec un autre auteur
################################

G_filtered = G.subgraph([node for node, degree in recomputed_degree.items() if degree > 0])

# Calculer les nouvelles positions des nœuds avec la disposition à ressort
pos = nx.random_layout(G_filtered)  # Ajuster le paramètre k pour contrôler l'espacement entre les nœuds

color_map_filtered = []
for node in G_filtered:
    if recomputed_degree[node] >= 10:
        color_map_filtered.append('red')  # Mettre en rouge si le degré est >= 10
    elif recomputed_degree[node] >= 2:
        color_map_filtered.append('blue')  # Mettre en bleu si le degré est >= 2 et < 10
    else:
        color_map_filtered.append('green')  # Mettre en vert pour les autres nœuds

nx.draw(G_filtered, pos, with_labels=False, node_color=color_map_filtered, width= 0.2, node_size=10)
plt.title("Réseau de collaboration entre auteurs ayant écrit avec au moins un autre auteur")
plt.show()

################################
# Afficher le graphe mais uniquement avec les auteurs qui ont écrit avec au moins 5 auteurs
################################

G_filtered = G.subgraph([node for node, degree in recomputed_degree.items() if degree > 4])

# Calculer les nouvelles positions des nœuds avec la disposition à ressort
pos = nx.random_layout(G_filtered)  # Ajuster le paramètre k pour contrôler l'espacement entre les nœuds

color_map_filtered = []
for node in G_filtered:
    if recomputed_degree[node] >= 30:
        color_map_filtered.append('red')  # Mettre en rouge si le degré est >= 30
    elif recomputed_degree[node] >= 10:
        color_map_filtered.append('blue')  # Mettre en bleu si le degré est >= 10
    else:
        color_map_filtered.append('green')  # Mettre en vert pour les autres nœuds

nx.draw(G_filtered, pos, with_labels=False, node_color=color_map_filtered, width= 0.2, node_size = 10)
plt.title("Réseau de collaboration entre auteurs ayant écrit avec au moins 5 autres auteurs")
plt.show()

################################
# Afficher le graphe mais uniquement avec les auteurs qui ont écrit avec au moins 10 auteurs
################################

G_filtered = G.subgraph([node for node, degree in recomputed_degree.items() if degree > 9])

# Calculer les nouvelles positions des nœuds avec la disposition à ressort
pos = nx.random_layout(G_filtered)  # Ajuster le paramètre k pour contrôler l'espacement entre les nœuds

color_map_filtered = []
for node in G_filtered:
    if recomputed_degree[node] >= 50:
        color_map_filtered.append('red')  # Mettre en rouge si le degré est >= 50
    elif recomputed_degree[node] >= 20:
        color_map_filtered.append('blue')  # Mettre en bleu si le degré est >= 20
    else:
        color_map_filtered.append('green')  # Mettre en vert pour les autres nœuds

nx.draw(G_filtered, pos, with_labels=False, node_color=color_map_filtered, width= 0.2, node_size = 10)
plt.title("Réseau de collaboration entre auteurs ayant écrit avec au moins 10 autres auteurs")
plt.show()

################################
#### Graphes connexes qui ont plus de 200 noeuds
################################

plt.figure(figsize=(15, 15))  
num_subgraphs = len(connected_components)
num_cols = 4 
large_components = [component for component in connected_components if len(component) > 200]
num_large_components = len(large_components)

for i, component in enumerate(large_components, start=1):
    subgraph = G.subgraph(component)
    pos = nx.spring_layout(subgraph)
    color_map = ['red' if recomputed_degree[node] >= 10 else 'blue' if recomputed_degree[node] >= 2 else 'green' for node in subgraph]
    plt.subplot((num_large_components // num_cols) + 1, num_cols, i)  # Calculer le nombre de lignes en fonction du nombre de composantes connexes
    nx.draw(subgraph, pos, with_labels=False, node_color=color_map, width=1, node_size=10)
    plt.title(f"Composante {i} - {len(subgraph.nodes)} nœuds")

plt.tight_layout()  # On ajuster automatiquement les positions des sous-graphes pour éviter les chevauchements
plt.show()

################################
#### Graphes connexes qui ont entre 20 et 199 noeuds
################################

plt.figure(figsize=(15, 15))  
num_subgraphs = len(connected_components)
num_cols = 4 
medium_components = [component for component in connected_components if 20 <= len(component) <= 199]  # Filtrer les composantes connexes contenant entre 20 et 199 noeuds
num_medium_components = len(medium_components)

for i, component in enumerate(medium_components, start=1):
    subgraph = G.subgraph(component)
    pos = nx.spring_layout(subgraph)
    color_map = ['red' if recomputed_degree[node] >= 10 else 'blue' if recomputed_degree[node] >= 2 else 'green' for node in subgraph]
    plt.subplot((num_medium_components // num_cols) + 1, num_cols, i)  # Calculer le nombre de lignes en fonction du nombre de composantes connexes
    nx.draw(subgraph, pos, with_labels=False, node_color=color_map, width=1, node_size=10)
    plt.title(f"Composante {i} - {len(subgraph.nodes)} nœuds")

plt.tight_layout()  # On ajuste automatiquement les positions des sous-graphes pour éviter les chevauchements
plt.show()

##############################################################################
##############################################################################
############################# Fonction 3 #####################################
##############################################################################
##############################################################################

# https://pypi.org/project/stop-words/
# https://velcin.github.io/files/NA/2.1_vect1.html

##############################################################################
# 1. Moteur de recherche basé sur TF-IDF et similarité cosinus
##############################################################################

# Transformation de la colonne qui contient les titres en liste
title_list = data['dcterms:title{Literal}'].tolist()

# Obtenir les stopwords
french_stopwords = list(get_stop_words('french'))

# Supprimer les stopwords dans notre liste qui contient les titres
# Utilisation de TFxIDF : prend en compte la rareté du mot au lieu du nombre d'occurrence
tfidf_vectorizer = TfidfVectorizer(stop_words=french_stopwords)
tfidf_vectorizer.fit(title_list)

# Montrer l'intégralité du vocabulaire
tfidf_vectorizer.vocabulary_

# Obtenir la matrice Documents x termes basées sur le vocabulaire
title_list_hp = tfidf_vectorizer.transform(title_list)
features_hp = tfidf_vectorizer.get_feature_names_out()

len(features_hp) # Nombre de mots dans le vocabulaire

# Obtenir 10 mots du vocabulaire aléatoirement
random_indices = random.sample(range(len(features_hp)), 10)
random_words = [features_hp[idx] for idx in random_indices]
print(random_words)

# Fonction qui permet d'obtenir les mots qui apparaissent le plus
def print_feats(v, features, top_n = 30):
    _, ids, values = find(v)
    feats = [(ids[i], values[i], features[ids[i]]) for i in range(len(list(ids)))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[0:top_n]
    return pd.DataFrame({"word" : [t[2] for t in top_feats], "value": [t[1] for t in top_feats]})   

print(title_list)
print_feats(title_list_hp[1], features_hp, top_n=5)

title_list[1]

# Affichage des mots qui apparaissent le plus fréquemment
n_docs, n_terms = title_list_hp.shape

# Somme sur toutes les lignes pour chacun des mots
tf_sum = title_list_hp.sum(axis=0)
tf_sum = tf_sum.tolist()[0] # conversion en liste

print_feats(tf_sum, features_hp)

# Fonction qui calcule le cosinus entre deux vecteurs (= deux documents)
def cosinus(i, j):
        # numérateur : <i.j>
    num = i.dot(j.transpose())[0,0]
        # dénominateur : ||i||_2 * ||j||_2
    den = norm(i.todense()) * norm(j.todense())
    if (den>0): # on vérifie que le dénominateur n'est pas nul
        return (num/den)
    else:
        return 0

query = ['juif', 'guerre'] # Mot(s) clé(s) choisis par l'utilisateur

indexes = [np.where(features_hp == q)[0][0] for q in query if q in features_hp] # Recherche les index des termes de la requête dans les caractéristiques (features_hp) et les stocke dans une liste
print(indexes) # Affichage

query_vec = np.zeros(n_terms)

# alternative pour pouvoir mettre plus que 1 en répétant les mots-clefs
for tt in indexes:
    query_vec[tt] += 1
    
query_vec = csr_matrix(query_vec) # Conversion du vecteur de requête en une matrice creuse csr_matrix pour l'efficacité du calcul
query_vec.sum() # Calcul de la somme des éléments du vecteur de requête

def search(q, X):
    cc = {i: cosinus(X[i], q) for i in range(n_docs)} # Calcul de la similarité cosinus entre chaque document et la requête
    cc = sorted(cc.items(), key=lambda x: x[1], reverse=True) # Trie les résultats par similarité décroissante
    return cc

result = search(query_vec, title_list_hp) # Recherche les documents les plus similaires à la requête dans la liste des titres

nb_top_docs = 10 # Nombre de documents les plus similaires à afficher
top_docs = [r for (r,v) in result[0:nb_top_docs]] # Sélectionne les n_docs documents les plus similaires
print(top_docs)  # Affiche les indices des documents les plus similaires

for i, td in zip(range(nb_top_docs), top_docs):
    print("%s (%s): %s" % (i+1, td, title_list[td])) # Affiche les titres des documents les plus similaires

##############################################################################
# 2. Intégration de l'information textuelle et structurelle avec les GNNs
##############################################################################

# https://pypi.org/project/stop-words/
# https://velcin.github.io/files/NA/2.1_vect1.html

######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
######## ATTENTION CETTE PARTIE 2. EST TRÈS LOURDE À EXÉCUTER 
########(NE S'EXECUTE PAS SUR NOS MACHINES MALHEUREUSEMENT)
######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 1. Préparation des données
# Filtre les nœuds du graphe basé sur un seuil de degré pour obtenir G_filtered
degrees = dict(G.degree())
G_filtered = G.subgraph(node for node, deg in degrees.items() if deg > 12)

# Sélection aléatoire d'un sous-ensemble des données pour simplifier les calculs
data_subset = data.sample(n=100, random_state=42)

# 2. Mapping des URI d'auteur à des catégories
# Initialisation du dictionnaire pour le mapping URI-catégorie
uri_to_category = {}
# Boucle sur le sous-ensemble des données pour le mapping
for index, row in data_subset.iterrows():
    for i in range(7):  # Assumant un maximum de 7 auteurs par document
        uri_col = f'marcrel:aut{{URIRef}}[{i}]'
        if pd.notna(row[uri_col]):
            uri_to_category[row[uri_col]] = row['Categorie']

# Mapping de catégorie à des labels numériques
category_to_label = {cat: i for i, cat in enumerate(set(uri_to_category.values()))}

# 3. Assignation des étiquettes aux nœuds
# Création d'un tensor pour les labels des nœuds dans G_filtered
labels_list = [category_to_label.get(uri_to_category.get(node), -1) for node in G_filtered.nodes()]
labels_tensor = torch.tensor(labels_list, dtype=torch.long)

# 4. Préparation des features des nœuds basées sur le TF-IDF des titres
# Extraction et vectorisation des titres des documents
tfidf_vectorizer = TfidfVectorizer()
tfidf_embeddings = tfidf_vectorizer.fit_transform(data_subset['dcterms:title{Literal}'].tolist()).toarray()

# Mapping des URI d'auteur aux indices de `data_subset`
uri_to_index = {uri: index for index, uri in enumerate(data_subset['marcrel:aut{URIRef}[0]'])}

# Initialisation du tensor pour les features des nœuds
node_features = torch.zeros((len(G_filtered.nodes), tfidf_embeddings.shape[1]), dtype=torch.float)

# Assignation des features TF-IDF aux nœuds correspondants dans G_filtered
for node in G_filtered.nodes():
    idx = uri_to_index.get(node)
    if idx is not None:
        node_features[list(G_filtered.nodes()).index(node)] = torch.tensor(tfidf_embeddings[idx], dtype=torch.float)

# 5. Conversion de G_filtered pour utilisation avec PyTorch Geometric
data_filtered = from_networkx(G_filtered)
data_filtered.x = node_features
data_filtered.y = labels_tensor

# 6. Définition d'un modèle Graph Neural Network (GNN)
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 7. Initialisation et entraînement du modèle
model = GCN(num_features=node_features.size(1), num_classes=len(torch.unique(labels_tensor)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Boucle d'entraînement
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data_filtered)
    loss = criterion(out[data_filtered.y != -1], data_filtered.y[data_filtered.y != -1])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        
##############################################################################
##############################################################################
############################# Fonction 4 #####################################
##############################################################################
##############################################################################

# https://www.kaggle.com/code/aybukehamideak/clustering-text-documents-using-k-means/notebook
# https://velcin.github.io/files/NA/2.2_vect2.html
# Louvain : https://www.kaggle.com/code/lsjsj92/network-graph-with-louvain-algorithm

##############################################################################
# 1. Clustering avec l'algorithme de Louvain
##############################################################################

############# Filtrage

# graphe filtré contenant uniquement les auteurs avec un certain nombre minimal de connexions
G_filtered = G.subgraph([node for node, degree in recomputed_degree.items() if degree > 2])

############# Clustering

# Appliquer l'algorithme de Louvain pour détecter les communautés
partition = community_louvain.best_partition(G_filtered)

############# Visualisation

# Visualisation du graphe avec les communautés
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G_filtered)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G_filtered, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G_filtered, pos, alpha=0.5)
plt.show()

############ Fusionner avec les Informations de Clustering

# Convertir le dictionnaire de partition en DataFrame pour faciliter la fusion
df_partition = pd.DataFrame(list(partition.items()), columns=['Auteur_URI', 'Cluster'])

# Transformation du DataFrame pour avoir une ligne par combinaison de document-auteur
df_authors_expanded = pd.DataFrame(data.set_index(data.columns.drop('marcrel:aut{URIRef}[0]',1).tolist())
   .stack()
   .reset_index()
   .rename(columns={0:'marcrel:aut{URIRef}[0]'})
   .loc[:, data.columns]
)

# Fusionner df_authors_expanded avec df_partition sur 'Auteur_URI'
df_merged = pd.merge(df_authors_expanded, df_partition, left_on='marcrel:aut{URIRef}[0]', right_on='Auteur_URI', how='left')

############# Nombre de cluster

# Nombre de clusters
num_clusters = len(set(partition.values()))

# Calcul du nombre total d'éléments
total_elements = len(partition)

print(f"Il y a {num_clusters} clusters pour un total de {total_elements} éléments.")

############# Nombre de catégories

# Compter le nombre unique de catégories
num_categories = df_merged['Categorie'].nunique()

print(f"Il y a {num_categories} catégories uniques.")

############# Analyse Par Catégorie

# Grouper par cluster et par catégorie pour compter le nombre de documents
cluster_category_counts = df_merged.groupby(['Cluster', 'Categorie']).size().reset_index(name='Counts')

# Créer un pivot table avec les clusters en index et les catégories en colonnes
pivot_table = cluster_category_counts.pivot(index='Cluster', columns='Categorie', values='Counts')

# Remplacer les valeurs NaN par des zéros pour une meilleure lisibilité
pivot_table = pivot_table.fillna(0).astype(int)

# Réinitialiser l'index pour avoir 'Cluster' en tant que colonne
pivot_table.reset_index(inplace=True)

# Afficher le tableau complet
print(pivot_table)

##############################################################################
# 2. Block Modèle
##############################################################################

############# Filtrage

G_filtered = G.subgraph([node for node, degree in recomputed_degree.items() if degree > 2])

############# Conversion

# Conversion du graphe NetworkX en igraph
def convert_networkx_to_igraph(G):
    # On s'assure que la conversion est correcte
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    reverse_mapping = dict(zip(range(G.number_of_nodes()), G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    g = ig.Graph(edges=list(G.edges()), directed=False)
    return g, reverse_mapping

G_filtered_ig, reverse_mapping = convert_networkx_to_igraph(G_filtered)

############# SBM

# Appliquer le SBM sur le graphe igraph
sbm = G_filtered_ig.community_leiden(objective_function="modularity")
clusters = sbm.membership

############# Block model

for node_index, cluster in enumerate(clusters):
    original_node_name = reverse_mapping[node_index]  # Accéder au nom original du noeud
    G_filtered.nodes[original_node_name]['BlockModelCluster'] = cluster

df_block_model_cluster = pd.DataFrame({'Auteur': list(reverse_mapping.values()), 'Cluster': clusters})

# Générer une palette de couleurs pour les clusters basée sur les nouveaux clusters attribués
n_clusters = max(clusters) + 1  # OOn s'assure que cela reflète le nombre de clusters distincts
color_map = plt.get_cmap('viridis', n_clusters)
node_colors = [color_map(G_filtered.nodes[node]['BlockModelCluster']) for node in G_filtered.nodes()]

############# Viusalisation

# Dessiner le graphe avec les couleurs basées sur Block Model
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_filtered, seed=42)
nx.draw_networkx_edges(G_filtered, pos, alpha=0.2)
nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=50, linewidths=0.8, edgecolors='k')
plt.title('Visualisation des Clusters obtenus par Block Model')
plt.axis('off')
plt.show()

############# Analyse

# Fusionner avec les données existantes, en utilisant les étiquettes du Block Model
df_merged_block_model = pd.merge(df_authors_expanded, df_block_model_cluster, left_on='marcrel:aut{URIRef}[0]', right_on='Auteur', how='left')
df_merged_block_model['BlockModelCluster'] = df_merged_block_model['Cluster']

# Analyse par catégorie en utilisant les étiquettes du Block Model
cluster_category_counts_block_model = df_merged_block_model.groupby(['BlockModelCluster', 'Categorie']).size().reset_index(name='Counts')
pivot_table_block_model = cluster_category_counts_block_model.pivot(index='BlockModelCluster', columns='Categorie', values='Counts')
pivot_table_block_model = pivot_table_block_model.fillna(0).astype(int)
pivot_table_block_model.reset_index(inplace=True)
print(pivot_table_block_model)

##############################################################################
# 3. Spectral Clustering Embedding
##############################################################################

########### TECHNIQUE 1 : Combien de cluster ?

# Calcul de la matrice laplacienne
L = nx.laplacian_matrix(G_filtered).asfptype()

# Calcul des k plus petites valeurs propres
k = 20
eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

# Tri des valeurs propres dans l'ordre croissant
eigenvalues_sorted = np.sort(eigenvalues)

# Visualisation des valeurs propres
plt.figure(figsize=(10, 6))
plt.plot(eigenvalues_sorted, marker='o')
plt.title('Spectre de la matrice laplacienne')
plt.xlabel('Index')
plt.ylabel('Valeur propre')
plt.grid(True)
plt.show()

########### TECHNIQUE 2 : Combien de cluster ?

# Supposons que L et k sont déjà définis comme dans votre code
eigenvalues_sorted = np.sort(eigenvalues)

# Trouver le plus grand gap entre les valeurs propres consécutives
eigen_gaps = np.diff(eigenvalues_sorted)
optimal_clusters = np.argmax(eigen_gaps) + 1  # +1 car les indices commencent à 0

print('Le nombre optimal de clusters selon la méthode Eigen Gap est :', optimal_clusters)

########### Mise-en-place modèle

# Application du clustering spectral
n_clusters = 17
spectral_clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='nearest_neighbors')

# Obtention de la matrice d'adjacence du graphe G_filtered
adj_matrix = nx.adjacency_matrix(G_filtered)

# Conversion de la matrice d'adjacence en une matrice dense
adj_matrix_dense = adj_matrix.toarray()
labels = spectral_clustering.fit_predict(adj_matrix_dense)

# Utilisation de SpectralEmbedding pour transformer la matrice d'adjacence
embedding = SpectralEmbedding(n_components=n_clusters, affinity='nearest_neighbors')
adj_matrix_transformed = embedding.fit_transform(adj_matrix)

# Application du clustering spectral sur l'espace transformé
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
labels = spectral_clustering.fit_predict(adj_matrix_transformed)

# Création du DataFrame avec les auteurs et les clusters
df_spectral_cluster = pd.DataFrame(list(G_filtered.nodes()), columns=['Auteur'])
df_spectral_cluster['Cluster'] = labels

############ Visualisation des Clusters sur le Graphe

# Assigner les étiquettes de cluster comme attributs des nœuds dans G_filtered
for i, node in enumerate(G_filtered.nodes()):
    G_filtered.nodes[node]['Cluster'] = labels[i]

# Générer une palette de couleurs pour les clusters
color_map = plt.cm.get_cmap('viridis', n_clusters)
node_colors = [color_map(G_filtered.nodes[node]['Cluster']) for node in G_filtered.nodes()]

# Dessiner le graphe
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_filtered, seed=42)  # Position des nœuds utilisant l'algorithme de Spring Layout
nx.draw_networkx_edges(G_filtered, pos, alpha=0.2)
nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=50, linewidths=0.8, edgecolors='k')
plt.title('Visualisation des Clusters obtenus par Clustering Spectral')
plt.axis('off')  # Enlever les axes pour une meilleure visibilité
plt.show()

############ Fusionner avec les Données Existantes

# Fusionner df_authors_expanded (ou votre DataFrame transformé pour les besoins de cette fusion) avec df_spectral_cluster sur l'identifiant/nom de l'auteur
df_merged_spectral = pd.merge(df_authors_expanded, df_spectral_cluster, left_on='marcrel:aut{URIRef}[0]', right_on='Auteur', how='left')

############# Analyse Par Catégorie

cluster_category_counts_spectral = df_merged_spectral.groupby(['Cluster', 'Categorie']).size().reset_index(name='Counts')
pivot_table_spectral = cluster_category_counts_spectral.pivot(index='Cluster', columns='Categorie', values='Counts')
pivot_table_spectral = pivot_table_spectral.fillna(0).astype(int)
pivot_table_spectral.reset_index(inplace=True)
print(pivot_table_spectral)

##############################################################################
# 4.Comparaison des méthodes 
##############################################################################

# Nettoyage des labels pour supprimer les NaN
cleaned_df_merged = df_merged.dropna(subset=['Cluster'])
cleaned_df_merged_spectral = df_merged_spectral.dropna(subset=['Cluster'])
cleaned_df_merged_block_model = df_merged_block_model.dropna(subset=['BlockModelCluster'])

# On s'assure que les indices sont réinitialisés pour permettre un alignement correct
cleaned_df_merged.reset_index(drop=True, inplace=True)
cleaned_df_merged_spectral.reset_index(drop=True, inplace=True)
cleaned_df_merged_block_model.reset_index(drop=True, inplace=True)

# Ré-extraction des labels de cluster nettoyés
labels_louvain_clean = cleaned_df_merged['Cluster'].values
labels_spectral_clean = cleaned_df_merged_spectral['Cluster'].values
labels_block_model_clean = cleaned_df_merged_block_model['BlockModelCluster'].values

# Recalcul des scores ARI et AMI avec les données nettoyées
ari_louvain_spectral_clean = adjusted_rand_score(labels_louvain_clean, labels_spectral_clean)
ami_louvain_spectral_clean = adjusted_mutual_info_score(labels_louvain_clean, labels_spectral_clean)

ari_louvain_block_clean = adjusted_rand_score(labels_louvain_clean, labels_block_model_clean)
ami_louvain_block_clean = adjusted_mutual_info_score(labels_louvain_clean, labels_block_model_clean)

ari_spectral_block_clean = adjusted_rand_score(labels_spectral_clean, labels_block_model_clean)
ami_spectral_block_clean = adjusted_mutual_info_score(labels_spectral_clean, labels_block_model_clean)

print(f"ARI Louvain vs Spectral (clean): {ari_louvain_spectral_clean}")
print(f"AMI Louvain vs Spectral (clean): {ami_louvain_spectral_clean}")
print()
print(f"ARI Louvain vs Block Model (clean): {ari_louvain_block_clean}")
print(f"AMI Louvain vs Block Model (clean): {ami_louvain_block_clean}")
print()
print(f"ARI Spectral vs Block Model (clean): {ari_spectral_block_clean}")
print(f"AMI Spectral vs Block Model (clean): {ami_spectral_block_clean}")

##############################################################################
##############################################################################
############################# Fonction 5 #####################################
##############################################################################
##############################################################################

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://velcin.github.io/files/NA/4.3_graph3.html

###########################
## Reconstruction du graphe : On s'intéresse qu'aux auteurs qui ont écrit au moins 5 documents
###########################

# Graphe 
G_classification = nx.Graph(G.subgraph(documents_by_author_categorie_5.index))

# Ajout des arêtes au sous-graphe seulement si les nœuds associés sont déjà présents
G_classification.add_edges_from(edge for edge in edges if all(node in G_classification for node in edge))

d_degree_classification = dict(G_classification.degree()) # Affichage des degré de chaque noeud dans un dictionnaire

# Collect de tous les nœuds uniques à partir des arêtes
unique_nodes_classification = set()

# Ajout des nœuds supplémentaires qui ne sont pas dans les arêtes
additional_nodes_classification = set(d_degree_classification.keys()) - unique_nodes_classification
unique_nodes_classification.update(additional_nodes_classification)

# Initialisation d'un dictionnaire pour stocker les degrés recalculés
recomputed_degree_classification = {node: 0 for node in unique_nodes_classification}

for edge, count in edge_counts.items():
    for node in edge:
        if node not in recomputed_degree_classification:
            recomputed_degree_classification[node] = 0  # Initialiser à zéro si le nœud n'existe pas
        recomputed_degree_classification[node] += count

# Affichage du graphe
pos = nx.random_layout(G_classification)  # Ajuster le paramètre k pour contrôler l'espacement entre les nœuds

color_map_filtered_classification = []
for node in G_classification:
    if recomputed_degree_classification[node] >= 20:
        color_map_filtered_classification.append('red')  # Mettre en rouge si le degré est >= 50
    elif recomputed_degree_classification[node] >= 10:
        color_map_filtered_classification.append('blue')  # Mettre en bleu si le degré est >= 20
    else:
        color_map_filtered_classification.append('green')  # Mettre en vert pour les autres nœuds

nx.draw(G_classification, pos, with_labels=False, node_color=color_map_filtered_classification, width= 0.2, node_size = 10)
plt.title("Réseau de collaboration entre auteurs ayant écrit au moins 5 documents")
plt.show()

# Distribution des degrés des auteurs #plus de 1051 sont isolés
node_degrees = dict(G_classification.degree())
degree_counts = Counter(node_degrees.values())

## Construction d'un tensor qui donne les relations entre les noeuds
# Dictionnaire pour mapper les noms de nœuds à des indices numériques
edges_classification = list(G_classification.edges())
node_to_index = {node: i for i, node in enumerate(G_classification.nodes)}
src_nodes = []
tgt_nodes = []
for edge in edges_classification:
    src_nodes.append(node_to_index[edge[0]])
    tgt_nodes.append(node_to_index[edge[1]])
edges_index_classification = torch.tensor([src_nodes, tgt_nodes])
print(edges_index_classification)

# Encodage des labels 
Label_encoded = TfidfVectorizer().fit_transform(Categorie_Label_Author["rdfs:label{Literal}"])

# Variable cible 
y = Categorie_Label_Author["Categorie"]

# Division des données en ensemble d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(Label_encoded, y, test_size=0.2, random_state=42)

# Création de masque pour pour les bases de test et d'entrainement
positions_y_test = [y.index.get_loc(index) for index in y_test.index]
positions_y_train = [y.index.get_loc(index) for index in y_train.index]
test_mask = Categorie_Label_Author.index.isin(y_test.index)
train_mask = Categorie_Label_Author.index.isin(y_train.index)

##############################################################################
########################## RANDOM FOREST #####################################
##############################################################################

# Initialisation du Ramdom forest
model_rd = RandomForestClassifier(n_estimators=100, random_state=42) # Initialisation un classifieur de forêts aléatoires avec 100 arbres et une graine aléatoire fixée à 42
                
# Entrainement du modèle 
start_time_rd = time.time() # Enregistrement du temps de début de l'entraînement du modèle RF          
model_rd.fit(X_train, y_train) # Entraînement du modèle de forêts aléatoires sur les données d'entraînement X_train avec les étiquettes y_train
end_time_rd = time.time() # Enregistrement du temps de la fin de l'entraînement du modèle RF 
execution_time_rd = end_time_rd - start_time_rd # Calcul du temps d'exécution total de l'entraînement du modèle GF
print("Temps d'exécution :", execution_time_rd, "secondes") # Affichage du temps d'exécution total de l'entraînement du modèle RF en secondes

# Prédiction des données de test
predictions_rd = model_rd.predict(X_test)

## RESULTATS
accuracy = accuracy_score(y_test, predictions_rd) # Calcul de l'accuracy 
print("L'accuracy est de: ", np.round(accuracy, 4)*100, "%") # Affichage de l'accuracy en %

###########################
## Visualisation de la matrice de confusion
###########################

conf_matrix_rd = confusion_matrix(y_test, predictions_rd) # Création de la matrice de confusion

unique_modalities_sorted = np.sort(np.unique(y_test)) # Trie alphabétique et unique des modalités de la variable cible (y_test)

# Création d'une colormap personnalisée allant du rose au violet
colors = [(1, 1, 0), (1, 0.5, 0), (1, 0, 0)]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Visualisation de la matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_rd, annot=True, fmt="d", cmap="YlOrRd")

# Définir les labels des axes x et y avec les modalités triées
plt.xticks(ticks=np.arange(len(unique_modalities_sorted)) + 0.5, labels=unique_modalities_sorted, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(unique_modalities_sorted)) + 0.5, labels=unique_modalities_sorted, rotation=0)

plt.title("Matrice de confusion - Ramdom forest")
plt.xlabel("Valeurs prédites")
plt.ylabel("Vraies réelles")
plt.show()

###########################
## Plus de métriques : analyse des performances par catégorie
###########################

# Calculer le F1-score, le recall et la précision pour chaque classe
precision_rd, recall_rd, f1_score_rd, _ = precision_recall_fscore_support(y_test, predictions_rd, labels=unique_modalities_sorted)

# Liste pour stocker les résultats par classe
metrics_list = []

# Remplir la liste avec les métriques calculées
for i, modality in enumerate(unique_modalities_sorted):
    metrics_list.append({
        "Catégorie": modality,
        "Précision": precision_rd[i],
        "Recall": recall_rd[i],
        "F1-score": f1_score_rd[i]
    })

# Convertir la liste en DataFrame
metric_rd = pd.DataFrame(metrics_list)

# Affichage des metrics, prenant compte du déséquilibre de la variable cible
print(metric_rd)

##############################################################################
########################## GAT v1 et GCN #####################################
##############################################################################

# Sélection du périphérique
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Sélection du périphérique approprié (GPU si disponible, sinon CPU)

## GRAPH 
# Conversion de positions_y_test en un ensemble pour une recherche efficace
positions_y_test_set = set(positions_y_test)

# Filtrer les paires d'indices dans edge_index_classification
filtered_indices = []
for i in range(edges_index_classification.size(1)):
    src, dst = edges_index_classification[0][i].item(), edges_index_classification[1][i].item()
    if src < 1391 and dst < 1391:
        filtered_indices.append(i)

# Création d'un nouveau tenseur filtré
filtered_edge_index = edges_index_classification[:, filtered_indices]

## LABEL
label_to_id = {label: i for i, label in enumerate(np.unique(y))} # dictionnaire pour mapper chaque étiquette unique à un identifiant numérique
y_numeric = [label_to_id[label] for label in y] # Conversion de chaque étiquette en son identifiant numérique correspondant
y_tensor = torch.tensor(y_numeric, dtype=torch.long).to(device)  # Conversion de la liste d'identifiants numériques en tenseur PyTorch et le déplace sur le périphérique spécifié

## FEATURES
# Conversion de la matrice Label_encoded en tenseur PyTorch avec le bon type de données et la déplace sur le périphérique spécifié
Label_encoded_tensor = torch.tensor(Label_encoded.todense(), dtype=torch.float).to(device)

## DATA
data = Data(x= Label_encoded_tensor, edge_index=filtered_edge_index , y=y_tensor) # Objet de type Data avec les caractéristiques encodées, les indices des arêtes et les étiquettes
data.train_mask = train_mask # Masque d'entraînement pour les données
data.test_mask = test_mask # Masque de test pour les données

data = data.to(device) # Déplacement des données sur le périphérique sélectionné

## NOMBRE DE CATEGORIE ET DIMENSIONS 
num_classes = len(np.unique(Categorie_Label_Author.Categorie)) # Nombre de catégorie
num_features = X_train.shape[1] # Dimensions de représentations des labels

# Classe GCN 
class MyGCN(torch.nn.Module):
    def __init__(self, d, n_feat):
        super().__init__()
        self.d = d
        # Initialisation des couches de convolutions
        self.conv1 = GCNConv(n_feat, 16)        
        self.conv2 = GCNConv(16, num_classes)

    def forward(self):
        # Récupération des données (données textuelles et données structurelles)
        x, edge_index = self.d, data.edge_index
        # Premiere convolution
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # Deuxième convolution
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Softmax = Attribue probabilité à chaque catégorie
        return F.log_softmax(x, dim=1)   

# Classe GAT
class MyGAT(torch.nn.Module):
    def __init__(self, d, n_feat):
        super().__init__()
        self.d = d
        # Initialization of convolution layers
        self.conv1 = GATConv(n_feat, 16)
        self.conv2 = GATConv(16, num_classes)

    def forward(self):
        # Récupération des données
        x, edge_index = self.d, data.edge_index
        # Premiere convolution
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # Deuxième convolution
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Softmax = Attribue probabilité à chaque catégorie
        return F.log_softmax(x, dim=1)

#Class Training
class Training():
    
    def __init__(self, m, o):
        self.model = m
        self.optim = o
    
    def train(self, nb_epochs=50):
        progress_bar = tqdm(range(nb_epochs))
        for epoch in progress_bar:
            progress_bar.set_description(f'Epoch: {epoch:03d}')        
            self.model.train() ## mode "train"
            self.optim.zero_grad()
            F.nll_loss(self.model()[data.train_mask], data.y[data.train_mask]).backward()
            self.optim.step()
        
    def eval(self):
        self.model.eval()
        
    def forward(self):
        return self.model.forward()

    # retourne les prédictions sur les données d'entraînement et de test
    @torch.no_grad()
    def test(self):
        self.model.eval()         
        log_probs = self.model()
        # La catégorie pour laquelle la probabilité est la plus élevée est prédite
        pred_test = np.array(log_probs[data.test_mask].max(1)[1])
        return pred_test

##############################################################################
########################## Résultats GAT #####################################
##############################################################################

modelGATv1, data = MyGAT(data.x, num_features).to(device), data.to(device) # Chargement du modèle et des données en mémoire (VRAM ou RAM en fonction de la présence ou non d'un gpu)
optimizerGATv1 = torch.optim.Adam(modelGATv1.parameters(), lr=0.05, weight_decay=1e-3) # Initialisation d'un optimiseur Adam pour ajuster les paramètres du modèle GAT avec un taux d'apprentissage initial de 0.05 et une régularisation de 1e-3
m1 = Training(modelGATv1, optimizerGATv1) # Entraînement du modèle GAT avec l'optimiseur Adam spécifié.

start_time_GAT = time.time() # Enregistrement du temps de début de l'entraînement du modèle GAT           
m1.train(nb_epochs=200) # Entraînement du modèle GAT pour 200 epochs
end_time_GAT = time.time() # Enregistrement du temps de fin de l'entraînement du modèle GAT    
execution_time_GAT = end_time_GAT - start_time_GAT  # Calcul du temps d'exécution total de l'entraînement du modèle GAT
print("Temps d'exécution :", execution_time_GAT, "secondes") # Affichage du temps d'exécution total de l'entraînement du modèle GAT en secondes

predictions_GAT = m1.test() # Prédictions sur l'ensemble de test à l'aide du modèle entraîné m1 (GAT).

# Inversion du dictionnaire label_to_id pour obtenir id_to_label
id_to_label = {i: label for label, i in label_to_id.items()}

# Transformation des valeurs numériques en étiquettes correspondantes pour y_tensor[test_mask] et prediction_test
y_labels = [id_to_label[num.item()] for num in y_tensor[test_mask]]
predictions_GAT_label = [id_to_label[num.item()] for num in predictions_GAT]

accuracy_GAT = accuracy_score(y_labels, predictions_GAT_label) # Calcul de l'accuracy
print("L'accuracy est de: ", np.round(accuracy_GAT, 2)*100, "%") # Affichage de l'accuracy

###########################
## Visualisation de la matrice de confusion
###########################

conf_matrix_GAT = confusion_matrix(y_labels, predictions_GAT_label) # Création de la matrice de confusion des résultats

unique_modalities_sorted = np.sort(np.unique(y_labels))  # Affichage des modalités de "Categorie" par ordre alphabétique

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_GAT, annot=True, fmt="d", cmap="YlOrRd") 

# Définition des labels des axes x et y avec les modalités triées
plt.xticks(ticks=np.arange(len(unique_modalities_sorted)) + 0.5, labels=unique_modalities_sorted, rotation=45, ha='right') # Affichage des modalités sur l'axe des abcisses
plt.yticks(ticks=np.arange(len(unique_modalities_sorted)) + 0.5, labels=unique_modalities_sorted, rotation=0) # Affichage des modalités sur l'axe des ordonnées

plt.title("Matrice de confusion - GAT") # Titre
plt.xlabel("Valeurs prédites") # Label de l'abcisse
plt.ylabel("Vraies réelles") # Label de l'ordonné
plt.show() # Affichage final du plot

###########################
## Plus de métriques : analyse des performances par catégorie
###########################

# Calculer le F1-score, le recall et la précision pour chaque classe
precision_GAT, recall_GAT, f1_score_GAT, _ = precision_recall_fscore_support(y_labels, predictions_GAT_label, labels=unique_modalities_sorted)

#  Dictionnaire contenant les métriques de précision, rappel et F1-score pour chaque classe prédite par le modèle GAT, en utilisant les modalités uniques triées comme clés.
metrics_by_class_GAT = {}
for i, modality in enumerate(unique_modalities_sorted):
    metrics_by_class_GAT[modality] = {
        'Precision': precision_GAT[i],
        'Recall': recall_GAT[i],
        'F1-score': f1_score_GAT[i]
    }

# Afficher les résultats pour chaque classe
metric_GAT = pd.DataFrame(columns=["Catégorie", "Précision", "Recall", "F1-score"])

for modality, metrics_GAT in metrics_by_class_GAT.items():
    metric_GAT = metric_GAT.append({
        "Catégorie": modality,
        "Précision": metrics_GAT["Precision"],
        "Recall": metrics_GAT["Recall"],
        "F1-score": metrics_GAT["F1-score"]
    }, ignore_index=True)

# Affichage du DataFrame
print(metric_GAT)

##############################################################################
################################# Résultats GCN ##############################
##############################################################################

model_GCN1, data = MyGCN(data.x, num_features).to(device), data.to(device) # Chargement du modèle et des données en mémoire (VRAM ou RAM en fonction de la présence ou non d'un gpu)
optimizer_GCN1 = torch.optim.Adam(model_GCN1.parameters(), lr=0.05, weight_decay=1e-3)  # Initialisation d'un optimiseur Adam pour ajuster les paramètres du modèle GAT avec un taux d'apprentissage initial de 0.05 et une régularisation de 1e-3  
m2 = Training(model_GCN1, optimizer_GCN1) # Entraînement du modèle GAT avec l'optimiseur Adam spécifié.

start_time_GCN = time.time()  # Enregistrement du temps de début de l'entraînement du modèle GCN          
m2.train(nb_epochs=200) # Entraînement du modèle GAT pour 200 epochs
end_time_GCN = time.time() # Enregistrement du temps de fin de l'entraînement du modèle GAT  
execution_time_GCN = end_time_GCN - start_time_GCN # Calcul du temps d'exécution total de l'entraînement du modèle GAT
print("Temps d'exécution :", execution_time_GCN, "secondes") # Affichage du temps d'exécution total de l'entraînement du modèle GAT en secondes

predictions_GCN = m2.test() # Prédictions sur l'ensemble de test à l'aide du modèle entraîné m1 (GCN).

# Transformer les valeurs numériques en étiquettes correspondantes pour y_tensor[test_mask] et prediction_test
predictions_GCN_label = [id_to_label[num.item()] for num in predictions_GCN]

# Transformer les valeurs numériques en étiquettes correspondantes pour y_tensor[test_mask] et prediction_test
y_labels = [id_to_label[num.item()] for num in y_tensor[test_mask]]
predictions_GCN_label = [id_to_label[num.item()] for num in predictions_GCN]

accuracy_GCN = accuracy_score(y_labels, predictions_GCN_label) # Calcul de l'accuracy
print("L'accuracy est de: ", np.round(accuracy_GCN, 2)*100, "%") # Affichage de l'accuracy

###########################
## Visualisation de la matrice de confusion
###########################

conf_matrix_GCN = confusion_matrix(y_labels, predictions_GCN_label) # Création de la matrice de confusion des résultats

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_GCN, annot=True, fmt="d", cmap="YlOrRd")

# Définir les labels des axes x et y avec les modalités triées
plt.xticks(ticks=np.arange(len(unique_modalities_sorted)) + 0.5, labels=unique_modalities_sorted, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(unique_modalities_sorted)) + 0.5, labels=unique_modalities_sorted, rotation=0)

plt.title("Matrice de confusion - GCN")
plt.xlabel("Valeurs prédites")
plt.ylabel("Vraies réelles")
plt.show()

###########################
## Plus de métriques : analyse des performances par catégorie
###########################

# Calcul du F1-score, du recall et de la précision pour chaque classe
precision_GCN, recall_GCN, f1_score_GCN, _ = precision_recall_fscore_support(y_labels, predictions_GCN_label, labels=unique_modalities_sorted)
# Créer un dictionnaire pour stocker les résultats par classe
metrics_by_class_GCN = {}
for i, modality in enumerate(unique_modalities_sorted):
    metrics_by_class_GCN[modality] = {
        'Precision': precision_GCN[i],
        'Recall': recall_GCN[i],
        'F1-score': f1_score_GCN[i]
    }

# Résultats pour chaque classe
metric_GCN = pd.DataFrame(columns=["Catégorie", "Précision", "Recall", "F1-score"])

for modality, metrics_GCN in metrics_by_class_GCN.items():
    metric_GCN = metric_GCN.append({
        "Catégorie": modality,
        "Précision": metrics_GCN["Precision"],
        "Recall": metrics_GCN["Recall"],
        "F1-score": metrics_GCN["F1-score"]
    }, ignore_index=True)

# Afficher le DataFrame
print(metric_GCN)

##############################################################################
################ Comparaison des résultats des 3 modèles #####################
##############################################################################

# Accuracy et temps d'éxécution pour chaque modèle
resultats = {
    'Accuracy (en %)': [np.round(accuracy*100,2), np.round(accuracy_GAT*100, 2), np.round(accuracy_GCN*100, 2)],
    'Execution Time (sec)': [np.round(execution_time_rd,2), np.round(execution_time_GAT,2), np.round(execution_time_GCN, 2)]
}
resultats = pd.DataFrame(resultats, index=['Random Forest', 'GAT', 'GCN'])
print(resultats)

# Autres métrics
df_RF = pd.DataFrame(metric_rd)
df_GAT = pd.DataFrame(metric_GAT)
df_GCN = pd.DataFrame(metric_GCN)

# Calcul des moyennes pour Random Forest en pourcentage
precision_RF_mean = round(df_RF['Précision'].mean() * 100, 2)
recall_RF_mean = round(df_RF['Recall'].mean() * 100, 2)
f1_score_RF_mean = round(df_RF['F1-score'].mean() * 100, 2)

# Calcul des moyennes pour GAT en pourcentage
precision_GAT_mean = round(df_GAT['Précision'].mean() * 100, 2)
recall_GAT_mean = round(df_GAT['Recall'].mean() * 100, 2)
f1_score_GAT_mean = round(df_GAT['F1-score'].mean() * 100, 2)

# Calcul des moyennes pour GCN en pourcentage
precision_GCN_mean = round(df_GCN['Précision'].mean() * 100, 2)
recall_GCN_mean = round(df_GCN['Recall'].mean() * 100, 2)
f1_score_GCN_mean = round(df_GCN['F1-score'].mean() * 100, 2)

# Moyenne non pondérée de chaque métrics en fonction du modèle utilisé
mean_metrics = {
    'Modèle': ['Random Forest', 'GAT', 'GCN'],
    'Précision (en %)': [precision_RF_mean, precision_GAT_mean, precision_GCN_mean],
    'Recall (en %)': [recall_RF_mean, recall_GAT_mean, recall_GCN_mean],
    'F1-score (en %)': [f1_score_RF_mean, f1_score_GAT_mean, f1_score_GCN_mean]
}

# Création d'un DataFrame pour les moyennes
df_mean_metrics = pd.DataFrame(mean_metrics)

# Affichage du DataFrame des moyennes
print("Moyennes pour chaque modèle :")
print(df_mean_metrics)