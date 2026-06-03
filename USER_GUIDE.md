# Mode d'emploi — Moteur de Recherche d'Images (MIR)

Ce guide explique comment utiliser l'application pas à pas. Aucune connaissance technique n'est requise.

---

## Sommaire

1. [Lancer le projet en local](#lancer-le-projet-en-local)
2. [Accès à l'application](#accès-à-lapplication)
3. [Section Recherche](#section-recherche)
   - [Étape 1 — Sélectionner une image](#étape-1--sélectionner-une-image)
   - [Étape 2 — Configurer la recherche](#étape-2--configurer-la-recherche)
   - [Étape 3 — Résultats](#étape-3--résultats)
   - [Étape 4 — Métriques d'évaluation](#étape-4--métriques-dévaluation)
4. [Section Benchmark](#section-benchmark)
5. [Section CLIP — Recherche multimodale](#section-clip--recherche-multimodale)
6. [Scripts utilitaires](#scripts-utilitaires)
7. [Notebooks d'entraînement](#notebooks-dentraînement)
8. [Déconnexion](#déconnexion)
9. [Résolution de problèmes courants](#résolution-de-problèmes-courants)

---

## Lancer le projet en local

### Prérequis

| Outil | Version minimale | Utilité |
|---|---|---|
| Python | 3.11+ | Backend FastAPI |
| pip | inclus avec Python | Installation des dépendances |
| Node.js | 18+ | Uniquement si vous modifiez le frontend |

### 1. Cloner le dépôt

```bash
git clone https://github.com/gmroue01/MIR_Project.git
cd MIR_Project
```

### 2. Installer les dépendances Python

Il est recommandé d'utiliser un environnement virtuel :

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Préparer les données

Placez le dossier `dataset/` (5 012 images de voitures) à la racine du projet :

```
MIR_Project/
├── dataset/
│   ├── Acura Integra Type R 2001_0001.jpg
│   └── ...
├── indexes/
└── ...
```

Les indexes pré-calculés (`indexes/*.npz`, `indexes/metrics.json`) sont déjà inclus — **aucune ré-indexation n'est nécessaire** pour lancer l'application.

> **Index SIFT (optionnel) :** pour activer le reranking SIFT-RANSAC, générez l'index avec :
> ```bash
> python scripts/generate_sift_index.py
> ```
> La réduction PCA (32D) est appliquée **automatiquement** à la suite — aucune étape supplémentaire n'est nécessaire.

> **Index CLIP (optionnel) :** pour activer la recherche CLIP/Flickr8K, générez les index avec :
> ```bash
> python scripts/generate_faiss_index.py
> ```
> Nécessite le dossier `Flickr8k/Images/` et `Flickr8k/captions.txt`. Durée estimée : ~5 min CPU, ~1 min GPU.

### 4. Variables d'environnement (optionnel)

Sans ces variables, l'application démarre sans authentification (accès libre — pratique en développement).

| Variable | Utilité |
|---|---|
| `APP_PASSWORD` | Active la page de connexion avec ce mot de passe |
| `JWT_SECRET` | Clé de signature des tokens JWT (obligatoire si `APP_PASSWORD` est défini) |

Sous Windows (PowerShell) :
```powershell
$env:APP_PASSWORD = "monmotdepasse"
$env:JWT_SECRET   = "une-chaine-aleatoire-longue"
```

Sous macOS / Linux :
```bash
export APP_PASSWORD="monmotdepasse"
export JWT_SECRET="une-chaine-aleatoire-longue"
```

### 5. Lancer le serveur

```bash
python run.py
```

Le serveur démarre sur **http://localhost:8000**. Ouvrez cette URL dans votre navigateur.

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 6. (Optionnel) Reconstruire le frontend

Le frontend compilé est déjà inclus dans `app/static/`. Cette étape n'est nécessaire que si vous modifiez le code React dans `frontend/src/`.

```bash
cd frontend
npm install
npm run build
```

Le build est automatiquement copié dans `app/static/` (configuré dans `vite.config.js`).

---

### Récapitulatif rapide

```bash
# 1. Environnement
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Placer dataset/ à la racine

# 3. Lancer
python run.py
# → http://localhost:8000
```

---

## Accès à l'application

Ouvrez l'URL de l'application dans votre navigateur. Une page de connexion s'affiche.

> ### 🔑 Mot de passe par défaut
>
> ```
> admin
> ```
>
> Ce mot de passe est configuré sur le déploiement Railway. En local sans variable `APP_PASSWORD`, l'application démarre sans authentification.

Entrez le mot de passe, puis cliquez sur **Connexion →** (ou appuyez sur Entrée).

> Le bouton ●/○ à droite du champ permet d'afficher ou masquer le mot de passe.

Une fois connecté, vous arrivez sur la page principale. La barre de navigation en haut donne accès aux trois sections :

| Onglet | Contenu |
|---|---|
| **Recherche** | Trouver des images similaires à une image requête |
| **Benchmark** | Comparer les performances des descripteurs |
| **CLIP** | Recherche par description textuelle (Flickr8K) |

---

## Section Recherche

La recherche se déroule en **4 étapes** numérotées. La barre de points sur la droite de l'écran indique où vous en êtes et permet de naviguer entre les étapes.

> Les étapes 3 et 4 ne sont accessibles qu'après avoir lancé une recherche.

---

### Étape 1 — Sélectionner une image

La grille affiche les 5 000 images du dataset (voitures).

**Pour filtrer par marque/modèle :**
Utilisez le menu déroulant en haut à droite pour n'afficher qu'une classe (ex. *BMW X3*, *Ford Fiesta*…).

**Pour naviguer entre les pages :**
Utilisez les boutons **‹** et **›** en haut à droite. Chaque page affiche 48 images.

**Pour sélectionner une image :**
Cliquez dessus. Elle se marque d'une coche ✓. La miniature sélectionnée apparaît en bas de l'écran avec le nom de sa classe.

**Pour continuer :**
Cliquez sur **Continuer →** en bas à droite.

---

### Étape 2 — Configurer la recherche

Cette étape permet de choisir **comment** l'application va chercher des images similaires.

#### Descripteurs

Un descripteur est la façon dont l'image est représentée numériquement pour la comparaison. Chaque descripteur capte un aspect différent de l'image.

| Descripteur | Ce qu'il capture |
|---|---|
| **Histo. Couleur** | La distribution des couleurs dans l'image |
| **MobileNet ArcFace** | Les caractéristiques visuelles profondes (modèle entraîné) |
| **MobileNet ZeroShot** | Caractéristiques MobileNet sans entraînement spécifique |
| **ResNet50 ZeroShot** | Caractéristiques ResNet50 sans entraînement spécifique |
| **ViT-B/16 ZeroShot** | Vision Transformer (analyse globale de la scène) |
| **DinoV2 SupCon** | DINOv2 avec apprentissage supervisé contrastif |
| **DinoV2 ZeroShot** | DINOv2 sans entraînement spécifique |

Vous pouvez **cocher plusieurs descripteurs** à la fois. Ils seront alors combinés (concaténés) pour la recherche — la mention **concaténés ×N** apparaît à côté du label.

#### Mesure de similarité

La mesure définit comment la ressemblance entre deux images est calculée.

| Mesure | Comportement |
|---|---|
| **Euclidienne** | Distance directe entre les vecteurs. Polyvalente, bon point de départ. |
| **Cosinus** | Compare l'orientation des vecteurs, insensible à l'échelle. |

#### Nombre de résultats

Choisissez **Top 20** ou **Top 50** selon le nombre d'images que vous souhaitez voir retournées.

#### Lancer la recherche

Cliquez sur **Lancer la recherche →**. Un indicateur de chargement apparaît le temps du calcul (de quelques millisecondes à ~2 secondes selon le descripteur). L'application passe automatiquement à l'étape 3 dès que les résultats sont prêts.

---

### Étape 3 — Résultats

La grille affiche les images les plus similaires à votre image requête, classées du plus proche au plus lointain.

**Lecture de la grille :**
- **Bordure verte** = image pertinente (même modèle de voiture que la requête)
- **Bordure rouge** = image non pertinente (modèle différent)
- **#N** en haut à gauche = rang dans les résultats
- En passant la souris sur une image : son nom de classe et sa distance s'affichent

En haut, un compteur indique le nombre d'images pertinentes retrouvées sur le total retourné.

#### Reranking SIFT-RANSAC

Le panneau en bas de page propose d'**affiner les résultats** avec une vérification géométrique. Cette étape est optionnelle et plus lente.

**Comment ça fonctionne :** après la recherche initiale, l'algorithme SIFT détecte des points-clés sur l'image requête et les compare avec les images candidates. Seules les images dont les correspondances géométriques sont cohérentes (vérification RANSAC) remontent dans le classement.

**Le slider "Pool de candidats" :** contrôle combien d'images supplémentaires sont analysées au-delà du top-K.
- Exemple : Top-50 + 25% = 63 images analysées, les 50 meilleures sont retournées.
- Plus le pool est grand, plus le résultat est précis, mais plus le calcul est long.

Cliquez sur **⬡ Lancer le reranking**. Quand le reranking est appliqué, un badge **✦ SIFT-RANSAC** apparaît dans l'en-tête, et chaque image affiche son nombre d'**inliers** (correspondances géométriques validées).

**Pour lancer une nouvelle recherche :** cliquez sur **↺ Nouvelle** en haut à droite.

**Pour voir les métriques :** cliquez sur **Voir les métriques →** en bas à droite.

---

### Étape 4 — Métriques d'évaluation

Cette étape affiche des indicateurs chiffrés de la qualité de la recherche.

#### Métriques par requête

Quatre métriques sont calculées pour Top 20 et Top 50 :

| Métrique | Ce qu'elle mesure |
|---|---|
| **Precision** | Proportion d'images pertinentes parmi les N résultats retournés |
| **Recall** | Proportion des images pertinentes de la base qui ont été retrouvées |
| **Average Precision** | Précision moyenne pondérée par le rang des résultats pertinents |
| **R-Precision** | Précision aux R premiers résultats, où R = nombre total de pertinents |

Une valeur de **100%** est parfaite, **0%** signifie qu'aucun résultat pertinent n'a été trouvé.

#### Calcul de la MAP (Mean Average Precision)

Le bouton **Calculer MAP** lance une évaluation sur l'ensemble des 46 classes : l'application effectue une recherche pour une image représentative de chaque classe et calcule la moyenne des Average Precision pour **Top 20, Top 50 et Top 100 simultanément**.

> Ce calcul prend environ 30 à 90 secondes selon le descripteur choisi (3 passes).

Le résultat affiche :
- Les **trois scores MAP** côte à côte : **MAP@20**, **MAP@50**, **MAP@100**
- Un **classement par classe** avec, pour chaque classe, trois barres colorées (@20, @50, @100) :
  - 🟢 Vert : AP > 50% — la classe est bien retrouvée
  - 🟡 Orange : AP entre 20% et 50% — résultats moyens
  - 🔴 Rouge : AP < 20% — la classe est difficile à retrouver

---

## Section Benchmark

Cette page compare les descripteurs sur des critères techniques.

Pour chaque descripteur, les indicateurs suivants sont affichés :

| Indicateur | Signification |
|---|---|
| **Dimension** | Taille du vecteur (ex. `1280 → 49` = réduit par PCA de 1280 à 49) |
| **Indexation** | Temps de calcul des vecteurs pour les 5 000 images |
| **Taille index** | Espace mémoire occupé par les vecteurs pré-calculés |
| **Temps recherche / img** | Durée d'une recherche en millisecondes |
| **Source** | *Colab* = entraîné sur GPU, *Local* = calculé en local |

Les barres horizontales permettent de comparer visuellement les descripteurs entre eux.

Un tableau récapitulatif en bas de page consolide toutes les métriques.

---

## Section CLIP — Recherche multimodale

CLIP permet de rechercher des images du dataset **Flickr8K** (8 091 photos de scènes naturelles) par description en langage naturel, et inversement.

> Les requêtes doivent être rédigées **en anglais** pour de meilleurs résultats.

---

### 01 — Texte → Image

Entrez une description dans le champ texte, choisissez le nombre de résultats (Top 5 / 10 / 20) et cliquez sur **Rechercher** (ou appuyez sur Entrée).

L'application retourne les images Flickr8K dont le contenu correspond le mieux à votre description. Chaque image affiche son **score de similarité** en pourcentage.

**Exemples de requêtes :**
```
a dog running on the beach
two children playing in the snow
a red car on a mountain road
a group of people at a concert
```

---

### 02 — Image → Texte (Inverse Search)

Parcourez la grille Flickr8K et **cliquez sur une image** pour la sélectionner (elle s'entoure d'une coche ✓). Choisissez le nombre de résultats et cliquez sur **Rechercher**.

L'application retourne les descriptions textuelles du dataset qui correspondent le mieux à l'image sélectionnée, avec leur score de similarité.

---

### 03 — Évaluation

Cette section permet de mesurer les performances du moteur CLIP sur un petit corpus personnalisé.

**Corpus Images (max 3) :**
Cliquez sur des images dans la grille pour les ajouter aux 3 emplacements. Cliquez à nouveau ou sur le **×** pour en retirer une.

**Corpus Textes (max 3) :**
Saisissez jusqu'à 3 descriptions dans les zones de texte. Pour des métriques précises, utilisez des captions issues du dataset Flickr8K (les textes officiellement associés aux images).

Choisissez le Top K et cliquez sur **Évaluer →**.

Les résultats affichent pour chaque direction (Texte→Image et Image→Texte) :
- La **MAP** globale du corpus
- Un tableau avec **P@K**, **R@K** et **AP** par requête

---

## Scripts utilitaires

Le dossier `scripts/` contient cinq scripts Python indépendants. Ils ne sont **pas nécessaires pour faire tourner l'application** (les indexes sont pré-calculés), mais permettent de régénérer ou recalibrer les données si besoin.

---

### `compute_metrics.py` — Ré-indexation complète

**Rôle :** recalcule tous les indexes de descripteurs à partir de zéro et met à jour `indexes/metrics.json` (temps d'indexation, taille, temps de recherche).

- **Descripteurs classiques** (Histogramme Couleur, SIFT) : extrait les vecteurs directement depuis les images du `dataset/`.
- **Descripteurs deep learning** : charge les embeddings pré-calculés depuis `modelsV2/*.pth`, les aligne sur le dataset, applique la réduction PCA + whitening + normalisation L2, et sauvegarde les indexes réduits.

```bash
python scripts/compute_metrics.py
```

> Nécessite `modelsV2/` (fichiers `.pth`) et `dataset/` (images).

---

### `apply_pca.py` — Réduction PCA des indexes DL

**Rôle :** applique la réduction dimensionnelle PCA + whitening + L2-norm aux indexes deep learning existants (`indexes/*.npz`), sans avoir à recharger les modèles `.pth`.

Dimensions cibles (95 % de variance expliquée) :

| Descripteur | Dimension originale | Dimension réduite |
|---|---|---|
| DinoV2 SupCon / ZeroShot | 256 | 13 |
| MobileNet ArcFace | 256 | 49 |
| ViT-B/16 ZeroShot | 768 | 230 |
| MobileNet ZeroShot | 1 280 | 257 |
| ResNet50 ZeroShot | 2 048 | 256 |

Produit également les fichiers `indexes/pca_*.npz` (modèles PCA) utilisés pour transformer les vecteurs requête au moment de la recherche.

```bash
python scripts/apply_pca.py
```

---

### `generate_sift_index.py` — Index SIFT pour le reranking

**Rôle :** extrait les descripteurs SIFT de chaque image du dataset et sauvegarde l'index dans `indexes/sift_ransac.npz`. Cet index est requis pour activer le reranking SIFT-RANSAC dans l'interface.

La réduction PCA (32D) est **appliquée automatiquement** à la fin du script via `apply_pca_sift.py`. Le résultat final est un index réduit (~146 Mo, -27%) avec une qualité de matching améliorée (+32% d'inliers RANSAC).

```bash
python scripts/generate_sift_index.py
```

> Prend plusieurs minutes selon la machine. À lancer une seule fois.

---

### `apply_pca_sift.py` — Réduction PCA de l'index SIFT

**Rôle :** applique une réduction PCA (32 dimensions, whitening + L2-normalisation) à un index `sift_ransac.npz` existant (uint8, 128D). Produit également le modèle PCA dans `indexes/pca_sift_ransac.npz`.

> Ce script est appelé automatiquement par `generate_sift_index.py`. Ne l'exécutez manuellement que si vous disposez déjà d'un `sift_ransac.npz` original non réduit.

```bash
python scripts/apply_pca_sift.py
```

| Avant | Après |
|---|---|
| 128D, uint8, ~200 Mo | 32D, float16, ~146 Mo (-27%) |
| 4,3 inliers RANSAC moy. | 5,7 inliers RANSAC moy. (+32%) |

---

### `generate_faiss_index.py` — Index FAISS pour la recherche CLIP

**Rôle :** encode toutes les images et captions du dataset Flickr8K avec le modèle CLIP (ViT-B/32) et sauvegarde les index FAISS dans `indexes_faiss/`. Ces fichiers sont requis pour activer la section CLIP de l'interface.

```bash
python scripts/generate_faiss_index.py
```

> Nécessite `Flickr8k/Images/` (8 091 images) et `Flickr8k/captions.txt`. Durée estimée : ~5 min CPU, ~1 min GPU.

Produit :
- `indexes_faiss/index_images.faiss` — 8 091 vecteurs image (512D)
- `indexes_faiss/index_captions.faiss` — 40 455 vecteurs caption (512D)

---

## Notebooks d'entraînement

Le dossier `notebook/` contient les notebooks Jupyter utilisés pour **entraîner les modèles deep learning** sur Google Colab (GPU). Ils documentent les choix d'architecture, les fonctions de perte et les résultats obtenus.

> Ces notebooks sont fournis à titre de documentation. Ils ne sont pas nécessaires pour faire tourner l'application — les poids entraînés (`modelsV2/*.pth`) sont déjà disponibles.

---

### `MIR_DinoV2_training.ipynb`

Fine-tuning de **DINOv2** (ViT-S/14, Meta AI) sur le dataset de voitures. Deux variantes sont entraînées :
- **SupCon** — avec une fonction de perte contrastive supervisée (SupConLoss), qui rapproche les embeddings d'images de même classe.
- **ZeroShot** — sans fine-tuning supplémentaire, les embeddings bruts de DINOv2 sont utilisés directement.

---

### `MIR_MobileNetV2_ArcFace_training.ipynb`

Fine-tuning de **MobileNetV2** avec la fonction de perte **ArcFace**, qui maximise la marge angulaire entre les classes. Adapté à la reconnaissance fine-grained (plusieurs modèles de voitures visuellement proches).

---

### `MIR_ResNet50_training.ipynb`

Fine-tuning de **ResNet-50** (pré-entraîné ImageNet) en mode zero-shot puis avec apprentissage supervisé. Sert de baseline solide pour comparer les architectures plus récentes.

---

### `MIR_Vitbase16_training.ipynb`

Fine-tuning de **ViT-B/16** (Vision Transformer, Google). Le patch-based attention permet une analyse globale de la scène, complémentaire aux CNN comme ResNet ou MobileNet.

---

### `MIR_SIFT_reranking.ipynb`

Exploration et prototypage de l'algorithme de **reranking géométrique SIFT-RANSAC** :
- Extraction de points-clés SIFT sur les images candidates
- Estimation de la transformation géométrique (homographie) par RANSAC
- Tri des candidats par nombre d'inliers (correspondances géométriques validées)

Ce notebook a servi à valider l'approche avant son intégration dans le backend.

---

## Déconnexion

Cliquez sur le bouton **⏻** en haut à droite de la barre de navigation pour vous déconnecter. Le token d'authentification est supprimé et vous retournez à la page de connexion.

---

## Résolution de problèmes courants

| Problème | Solution |
|---|---|
| La page met longtemps à charger | Le premier accès charge les indexes en mémoire (~2-5s). Les requêtes suivantes sont plus rapides. |
| La recherche CLIP texte est lente | Le modèle CLIP (~175 Mo) est chargé à la première requête texte. Les suivantes sont quasi-instantanées. |
| "Index SIFT non disponible" | Lancez `python scripts/generate_sift_index.py` (la réduction PCA s'applique automatiquement). |
| "Index CLIP non disponible" | Lancez `python scripts/generate_faiss_index.py` (nécessite le dossier `Flickr8k/`). |
| "Mot de passe incorrect" | Après 5 tentatives erronées, l'accès est bloqué 5 minutes. |
| Les images ne s'affichent pas | Vérifiez votre connexion. Les images sont servies depuis le même serveur que l'API. |
