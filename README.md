# M2 MALIA-MIASHS : Projet d'Analyse de Réseau pour la Recherche d'Informations

## Présentation

Ce projet a été réalisé dans le cadre du cours "Network Analysis for Information Retrieval", proposé par Julien Velcin à l'Université Lyon 2, Laboratoire ERIC, en Février 2024. Le but principal est de développer une solution d'analyse d'un corpus structuré intégrant plusieurs fonctionnalités clés, allant du chargement des données à la classification supervisée en passant par la visualisation et le clustering.

## Fonctionnalités

- **Chargement rapide des données** et affichage de statistiques.
- **Visualisation du corpus** pour examiner la structure des données.
- **Moteur de recherche** basé sur des mots-clés.
- **Nouvelle structuration des données** via techniques de clustering.
- **Classification supervisée** des données, combinant structure et information textuelle.

## Prérequis

Le fichier `requirements.txt` contient toutes les bibliothèques nécessaires à l'exécution du projet.

## Structure des Dossiers

- `SCRIPT_GERVASONI_MARTIN.py` : Script principal du projet.
- `RAPPORT_GERVASONI_MARTIN.pdf` : Rapport détaillant les méthodes utilisées, les résultats obtenus, et les conclusions.
- `requirements.txt` : Liste des dépendances Python nécessaires.

## Base de Données utilisée

Nous avons utilisé la base de données nommée "20240125_dataset_rscir_xxs.pickle". Étant donné que le dataset est très lourd, nous ne pouvons par le partager ici. Mais cette base de données est un ensemble structuré de métadonnées descriptives issues du corpus de documents disponibles sur le portail www.persee.fr, un site dédié à la numérisation et à la diffusion du patrimoine scientifique dans les domaines des sciences humaines et sociales. 

Le dataset spécifique a été généré à partir d'un sous-ensemble des fichiers de dumps de données liées (.rdf) du triplestore de Persée, disponibles à l'adresse https://data.persee.fr/explorer/demander-un-dump/dumps-collections/. 

### Note importante

Le moteur de recherche (Fonction 3) est particulièrement exigeant en termes de mémoire RAM. Assurez-vous de disposer de suffisamment de ressources système avant de lancer cette partie.

## Contribution

Ce projet a été réalisé en collaboration par GERVASONI Coraly et MARTIN Perrine. 
