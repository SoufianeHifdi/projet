# Projet Pres Surveillance Vidéo

Dans ce fichier on retrouve les fichiers qui ont été developés dans le cadre du projet PRES, surveillance caméra.
Dans ce projet on considére qu'on a 4 caméras.

Pour pouvoir éxécuter le code des deux fichiers object_trakcer.py et object_reid.py, il faut suivre les étapes suivantes:

## Etape 1 : Installer Conda 
Suivre les étapes du lien suivant : https://developers.google.com/earth-engine/guides/python_install-cond

## Etape 2 : Installer Yolov4
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu

# Cloner le répértoire
git clone https://github.com/SoufianeHifdi/yolov4-deepsort.git

# Télécharger les poids 
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/

# Installer les dépendances
pip3 install -r requirements-gpu.txt
python3 save_model.py --model yolov4
```
Si erreur ou autre, suivre les étapes du lien suivant : https://github.com/theAIGuysCode/yolov4-custom-functions#readme

## Etape 3 : Installer TorchReid
```bash
# Installer condacolab
pip3 install -q condacolab

# Cloner le reperotire de torchreid
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/

# Créer environnement d'execution
conda create --name torchreid python=3.7
conda activate torchreid

# Installer les dependances
pip3 install -r requirements.txt
```
Si erreur ou autre, suivre les étapes du lien suivant : https://github.com/KaiyangZhou/deep-person-reid

## Etape 4 : Exécuter le code
   1) Placer le fichier object_reid.py dans le dossier deep-person-reid
   2) Placer le fichier object_tracker.py dans le dossier yolov4-deepsort
   3) Place le fichier Executor.sh dans le dossier racine du projet (au meme niveau que deep-person-reid et de yolov4-deepsort)
   4) Éxecuter le script bash Executor.sh
```bash
bash executor.sh
```

Les résultats de l'exécution des trackings avec yolo (la ou sont stockes les crops) se trouvent dans yolov4-deepsort/Camera{#camera}/id{#id}

Les résultats du reid, et de l'assosciation d'id de plusieurs caméra se trouvent dans le dossier reid_results
