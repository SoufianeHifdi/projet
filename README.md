# projet Pres Surveillance Video

Dans ce fichier on retrouve les fichiers qui ont ete developes dans le cadre du projet PRES, surveillance camera.
Dans ce projet on considere qu'on a 4 cameras.

Pour pouvoir executer le code des deux fichiers object_trakcer.py et object_reid.py, il faut suivre les etapes suivantes:

## Etape 1 : Installer Conda 
Suivre les etapes du lien suivant : https://developers.google.com/earth-engine/guides/python_install-cond

## Etape 2 : Installer Yolov4
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu

# Cloner le repertoire
git clone https://github.com/SoufianeHifdi/yolov4-deepsort.git

# Telecharger les poids 
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/

# Installer les dependances
pip3 install -r requirements-gpu.txt
python3 save_model.py --model yolov4
```
Si erreur ou autre, suivre les etapes du lien suivant : https://github.com/theAIGuysCode/yolov4-custom-functions#readme

## Etape 3 : Installer TorchReid
```bash
# Installer condacolab
pip3 install -q condacolab

# Cloner le reperotire de torchreid
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/

# Creer environnement d'execution
conda create --name torchreid python=3.7
conda activate torchreid

# Installer les dependances
pip3 install -r requirements.txt
```
Si erreur ou autre, suivre les etapes du lien suivant : https://github.com/KaiyangZhou/deep-person-reid

## Etape 4 : Executer le code
   1) Placer le fichier object_reid.py dans le dossier deep-person-reid
   2) Placer le fichier object_tracker.py dans le dossier yolov4-deepsort
   3) Executer le script bash Executor.sh
```bash
bash executor.sh
```
