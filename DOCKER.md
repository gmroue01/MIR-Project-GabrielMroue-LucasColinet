# L'image Docker — conception et fonctionnement

Ce document explique comment l'image Docker du projet est construite, ce qu'elle contient, et ce qui se passe depuis la commande `docker build` jusqu'au premier appel API.

---

## Pourquoi Docker ?

Sans Docker, lancer l'application sur une nouvelle machine demande d'installer Python, Node.js, OpenCV, PyTorch, toutes les dépendances pip, builder le frontend, et configurer les chemins. Docker empaquette tout ça dans une seule unité autonome : l'**image**. On peut ensuite la déployer sur n'importe quelle machine Linux (un serveur cloud, Railway.app, une VM) avec exactement le même comportement.

---

## Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│  docker build                                            │
│                                                          │
│  Stage 1 : node:20-alpine          Stage 2 : python:3.11 │
│  ┌─────────────────────┐           ┌──────────────────┐  │
│  │ npm ci              │           │ apt-get (OpenCV) │  │
│  │ npm run build       │  ──────►  │ pip install      │  │
│  │ → /frontend/dist/   │  copie    │ code applicatif  │  │
│  └─────────────────────┘           │ frontend buildé  │  │
│                                    └──────────────────┘  │
│                                            │             │
│                                     Image finale (~2 Go)  │
└──────────────────────────────────────────────────────────┘
```

L'image est construite en **deux étapes** (multi-stage build). Cette technique est courante : elle permet de garder l'image finale légère en n'embarquant que ce qui est nécessaire à l'exécution, pas à la compilation.

---

## Stage 1 — Compilation du frontend (node:20-alpine)

```dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build
```

**Ce qui se passe :**
1. On part d'une image Node.js minimaliste (Alpine Linux, ~50 Mo).
2. On copie uniquement `package.json` et `package-lock.json` en premier — si ces fichiers n'ont pas changé, Docker réutilise le cache de l'étape `npm ci` lors des builds suivants (optimisation importante).
3. `npm ci` installe exactement les dépendances déclarées dans `package-lock.json` (reproducible build).
4. `npm run build` compile le code React + Vite en fichiers statiques (`index.html`, `assets/*.js`, `assets/*.css`).

À la fin de ce stage, tout Node.js et les `node_modules` (plusieurs centaines de Mo) sont utilisés puis **abandonnés**. Seul le dossier `dist/` compilé (~500 Ko) sera récupéré.

---

## Stage 2 — Runtime Python (python:3.11-slim)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
```

C'est l'image qui sera réellement déployée. On part d'une base Python légère sans les outils de développement.

### 2.1 Dépendances système

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

OpenCV (bibliothèque de vision par ordinateur) a besoin de deux bibliothèques système Linux. On les installe avec `apt-get` puis on supprime le cache `apt` immédiatement pour ne pas alourdir l'image. Le flag `--no-install-recommends` évite d'installer des paquets suggérés mais non nécessaires.

### 2.2 PyTorch CPU

```dockerfile
RUN pip install --no-cache-dir \
        torch==2.6.0+cpu \
        torchvision==0.21.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu
```

PyTorch est installé séparément, avant les autres dépendances, pour deux raisons :
- La version `+cpu` (sans CUDA) est disponible sur un index PyPI distinct (`download.pytorch.org`).
- PyTorch est très lourd (~700 Mo). L'isoler dans une couche Docker dédiée permet de ne pas le réinstaller si seul `requirements.txt` change.

### 2.3 Dépendances Python

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub
```

FastAPI, Uvicorn, NumPy, scikit-learn, OpenCV, CLIP, FAISS, etc. sont installés ici. `huggingface_hub` est ajouté explicitement car il est nécessaire au script de démarrage mais pas listé dans `requirements.txt` (il ne fait pas partie du runtime normal).

`--no-cache-dir` évite que pip conserve les archives `.whl` téléchargées, ce qui réduirait la taille de l'image.

### 2.4 Code applicatif

```dockerfile
COPY app/ ./app/
COPY indexes/ ./indexes_image/
COPY run.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
```

Le code Python est copié dans `/app/app/`. Les indexes pré-calculés (vecteurs de features des 5000 voitures) sont copiés dans `/app/indexes_image/` — notez le nom différent : ce n'est pas le dossier live, c'est une **source de référence** que le script de démarrage copie vers le bon endroit.

### 2.5 Récupération du frontend compilé

```dockerfile
COPY --from=frontend-builder /app/static ./app/static/
```

Cette ligne est le lien entre les deux stages. Elle récupère uniquement les fichiers compilés produits par Stage 1, sans embarquer Node.js ni les `node_modules`.

---

## Ce que contient l'image finale

```
/app/
├── app/                    ← code Python (FastAPI, descripteurs, searcher…)
│   ├── main.py
│   ├── auth.py
│   ├── searcher.py
│   ├── clip_searcher.py
│   ├── static/             ← frontend React compilé (HTML + JS + CSS)
│   └── …
├── indexes_image/          ← indexes de référence (embarqués dans l'image)
│   ├── mobilenet_arcface.npz
│   ├── dinov2_supcon.npz
│   ├── filenames.npy
│   ├── classes.npy
│   └── …
├── run.py                  ← point d'entrée Uvicorn
└── entrypoint.sh           ← script de démarrage
```

**Ce qui n'est PAS dans l'image** (voir `.dockerignore`) :
- `dataset/` — 5000 images voitures (~300 Mo)
- `indexes_faiss/` — index FAISS pour CLIP (~100 Mo)
- `Flickr8K/` — images et captions (~1 Go)
- `venv/`, `node_modules/`, caches, notebooks

Ces données volumineuses ne sont **pas** dans l'image. Elles vivent sur un **volume persistant Railway** qui a été alimenté une seule fois depuis le PC via le CLI Railway.

---

## Le démarrage — entrypoint.sh

Quand un conteneur démarre à partir de l'image, Docker exécute `./entrypoint.sh`. Ce script fait quatre choses dans l'ordre.

### Étape 1 — Vérification des données (volume Railway)

```sh
SENTINEL="$DATA_DIR/.download_complete"
if [ ! -f "$SENTINEL" ]; then
    python -c "snapshot_download('GabrielMroue/MirProject', ...)"
    touch .download_complete
fi
```

Le script vérifie si un fichier sentinelle `.download_complete` est présent dans `/app/data/`. Ce dossier correspond au **volume Railway** monté dans le conteneur.

**En production (Railway)** : le volume a été peuplé une seule fois en uploadant les données directement depuis le PC via la commande Railway CLI :
```bash
railway volume push ./dataset      /app/data/dataset
railway volume push ./indexes_faiss /app/data/indexes_faiss
railway volume push ./Flickr8K     /app/data/Flickr8k
```
Le sentinelle étant présent, le script saute le téléchargement et démarre immédiatement.

**Fallback (volume vide ou nouveau déploiement)** : si le sentinelle est absent, le script tente de télécharger les données depuis HuggingFace (`GabrielMroue/MirProject`). Il réessaie 5 fois avec un timeout de 10 minutes par tentative.

### Étape 2 — Création de liens symboliques

```sh
ln -sfn /app/data/dataset        /app/dataset
ln -sfn /app/data/indexes_faiss  /app/indexes_faiss
ln -sfn /app/data/Flickr8k       /app/Flickr8K
```

Le code Python référence des chemins fixes (`/app/dataset`, `/app/indexes_faiss`…). Les liens symboliques font pointer ces chemins vers les données du volume sans modifier le code.

### Étape 3 — Synchronisation des indexes

```sh
for f in /app/indexes_image/*; do
    # Copie les .npz manquants vers /app/data/indexes/
    # Écrase toujours les .json et .npy (métadonnées)
done
ln -sfn /app/data/indexes /app/indexes
```

Les indexes de features (`.npz`) sont embarqués dans l'image comme source de référence. Ce loop les copie vers le volume persistant `/app/data/indexes/` si ils n'y sont pas déjà. Les fichiers de métadonnées (`.npy`, `.json`) sont toujours écrasés pour prendre en compte une mise à jour de l'image. Après la copie, `/app/indexes` pointe vers ce dossier live.

**Pourquoi cette complexité ?** Séparer l'image (immutable) du stockage (mutable) permet de mettre à jour l'image sans perdre les indexes calculés localement (ex. `sift_ransac.npz` généré par un script).

### Étape 4 — Lancement du serveur

```sh
exec python run.py
```

`run.py` démarre Uvicorn sur le port défini par la variable `PORT` (défaut : 8000). `exec` remplace le processus shell par le processus Python, ce qui fait que Python devient le PID 1 du conteneur et reçoit correctement les signaux d'arrêt de Docker.

---

## Stratégie de stockage des données selon l'environnement

Les données volumineuses (~1,3 Go au total) ne peuvent pas être embarquées dans l'image. Elles sont gérées différemment selon l'environnement.

### Production — volume Railway (upload CLI)

Railway fournit un volume persistant attaché au service. Les données ont été uploadées **une seule fois** depuis le PC de développement :

```bash
railway volume push ./dataset       /app/data/dataset
railway volume push ./indexes_faiss /app/data/indexes_faiss
railway volume push ./Flickr8K      /app/data/Flickr8k
```

Le volume survit aux redéploiements : mettre à jour l'image (nouveau `git push` → Railway rebuild) ne supprime pas les données. Le conteneur redémarre, trouve le sentinelle, et saute le téléchargement.

### Développement local — volumes Docker Compose

```yaml
volumes:
  - ./indexes_faiss:/app/indexes_faiss:ro
  - ./dataset:/app/dataset:ro
  - ./Flickr8K:/app/Flickr8K:ro
```

Les données sont déjà présentes sur la machine hôte. `docker-compose` les monte directement dans le conteneur en lecture seule (`:ro`), court-circuitant le script de téléchargement. Zéro transfert réseau, démarrage immédiat.

---

## Cycle de vie complet

```
docker build .
    │
    ├─ Stage 1: npm ci + npm run build → dist/
    └─ Stage 2: apt + pip + COPY code + COPY dist/ → image finale

Railway déploie l'image (volume déjà peuplé via CLI)
    │
    └─ entrypoint.sh
        ├─ Sentinelle présent → données OK, skip téléchargement
        │  (fallback : snapshot_download HuggingFace si volume vide)
        ├─ Liens symboliques dataset / indexes_faiss / Flickr8K
        ├─ Sync indexes embarqués → volume persistant
        └─ exec python run.py → Uvicorn sur :8000
                │
                ├─ GET /            → index.html (SPA React)
                ├─ GET /assets/*    → JS/CSS compilé
                ├─ GET /images/*    → photos voitures (protégé par auth)
                ├─ GET /api/*       → endpoints FastAPI (protégé par auth)
                └─ GET /flickr8k/*  → images Flickr8K (protégé par auth)
```

---

## Optimisations de taille et de build time

| Technique | Gain |
|---|---|
| Multi-stage build | Node.js (~300 Mo) absent de l'image finale |
| `COPY package*.json` avant `COPY frontend/` | Cache `npm ci` réutilisé si les dépendances n'ont pas changé |
| PyTorch dans une couche séparée | Pas réinstallé si seul `requirements.txt` change |
| `--no-cache-dir` pip | Pas d'archives `.whl` stockées dans l'image |
| `rm -rf /var/lib/apt/lists/*` | Cache apt supprimé après installation |
| `.dockerignore` | `venv/`, `node_modules/`, notebooks, dataset exclus du contexte de build |
| Volume Railway (upload CLI) | ~1,3 Go de données hors image, persistant entre redéploiements |
