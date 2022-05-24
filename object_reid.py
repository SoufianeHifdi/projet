from cmath import inf, nan
import shutil
import torchreid
import torch
import numpy as np    
import collections
import os
from shutil import copyfile
import gc
gc.collect()
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, compute_model_complexity,FeatureExtractor
)

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='model.pth.tar',
    device='cpu'
)

# Construire les paths vers les ids
path_cam1 = "../yolov4-deepsort/camera1"
path_cam2 = "../yolov4-deepsort/camera2"
path_cam3 = "../yolov4-deepsort/camera3"
path_cam4 = "../yolov4-deepsort/camera4"


Camera_to_ids_to_imgPaths={"camera1":{},"camera2":{},"camera3":{},"camera4":{}}
Cam_to_ids_to_savePaths = {"camera1":{},"camera2":{},"camera3":{},"camera4":{}}
path_cams = [path_cam1,path_cam2,path_cam3,path_cam4]


for path_cam in path_cams:
    ids = os.listdir(path_cam)
    for id in ids:
        if id.startswith('id'):
            id_path = path_cam+"/"+ id #sort tous les fichiers contenue dans path_cam1
            Camera_to_ids_to_imgPaths["camera"+path_cam[-1:]][id]=[]
            Cam_to_ids_to_savePaths["camera"+path_cam[-1:]][id]=None
#Construire la liste des paths vers les images        
        images = os.listdir(id_path)
        for image in images:
            if image.startswith("crop"):
                img_paths= id_path+"/"+image
                Camera_to_ids_to_imgPaths["camera"+path_cam[-1:]][id].append(img_paths)
    

global_id = 0
print("\n\n\t Assosciations proches: \n")
for level in range(len(path_cams)-1):
    
    current_layer = "camera"+str(level+1)
    next_layer = "camera"+str(level+1+1)
    print("Traitement",current_layer,"et",next_layer)
    
    metrics_matrix= []
    nb_columns =0
    for current_id in Camera_to_ids_to_imgPaths[current_layer]:
        nb_columns += 1
        
        metrics_list=[]
        for compared_id in Camera_to_ids_to_imgPaths[next_layer]:
            print("\tComparaison", current_layer, current_id, "avec",next_layer,compared_id)
            
            # Calcul des vecteurs caracteristiques
            feature_vec_Current_id = extractor(Camera_to_ids_to_imgPaths[current_layer][current_id])
            feature_vec_compared_id = extractor(Camera_to_ids_to_imgPaths[next_layer][compared_id])

            # Calcul la matrices des distances
            Distance_matrix = torchreid.metrics.distance.cosine_distance(feature_vec_Current_id, feature_vec_compared_id)

            # Calcul la metrique, moyenne de tous les elements de la matrice
            metrique = torch.mean(Distance_matrix)
            
            # Sauvegarder la metrique dans la matrice de layer
            metrics_list.append(metrique)  
        
        metrics_matrix.append(metrics_list)
    
    # Cree la matrice et transpose pour que que axe x reprensente id de 
    #la camera de la couche superieur (ids de cam1 dans l'axe des x, ids de cam2 en y )
    metrics_matrix= np.matrix(metrics_matrix).transpose() 
    
    # Tant que la matrice n est pas vide et que le min est superieur au seuil, 
    # alors on a au moins une assosciation valide.
    seuil = 0.42
    
    print("\tMatrice des comparaison:",current_layer,"VS",next_layer,"est", metrics_matrix)
    while np.amin(metrics_matrix)!=inf and np.amin(metrics_matrix)<seuil : #!TODO il faut ajouter la conditions si rien ne corespons ; and min > metriques
        #trouve le minimum dans la matrice et retourne les coordonnees
        result = np.where(metrics_matrix == np.amin(metrics_matrix))
        # zip the 2 arrays to get the exact coordinates
        listOfCordinates = list(zip(result[0], result[1]))
        
        x = listOfCordinates[0][0] # ligne dans la matrice -> ids de la cameras inferieur
        y = listOfCordinates[0][1] # colonne -> ids de la camera superieure
        
        #Convertit les index du min dans la matrice en ids (cles):
        noeud_couche_sup = list(Cam_to_ids_to_savePaths[current_layer].keys())[y]
        noeud_couche_inf = list(Cam_to_ids_to_savePaths[next_layer].keys())[x]
        
        #Associe les ids
        print("\n\tAssosciation Trouvee !\n\t ",current_layer, noeud_couche_sup,"---->", next_layer,noeud_couche_inf,"\n")
        
        #si premier neoud d'une chaine(racine), affecte un identifiant global
        # cree le dossier de l'id global : 
        if Cam_to_ids_to_savePaths[current_layer][noeud_couche_sup] ==None:
            global_id +=1
            global_id_path = "../reid_results/global_id"+str(global_id)
            try:
                    os.makedirs(global_id_path)
            except:
                pass
            
            Cam_to_ids_to_savePaths[current_layer][noeud_couche_sup]= global_id_path

            #copie background de la racine du dossier local au dossier global
            source_file = "../yolov4-deepsort/"+current_layer+"/"+noeud_couche_sup+"/"+"tracklet_"+current_layer+".jpg"
            destination_file= global_id_path+"/"+"tracklet_"+current_layer+".jpg"
            shutil.copyfile(source_file, destination_file)


        Cam_to_ids_to_savePaths[next_layer][noeud_couche_inf] = Cam_to_ids_to_savePaths[current_layer][noeud_couche_sup]
        global_id_path = Cam_to_ids_to_savePaths[next_layer][noeud_couche_inf]
        
        #copie tracklet de la couche inferieure dans le repertoire de l'id global
        source_file = "../yolov4-deepsort/"+next_layer+"/"+noeud_couche_inf+"/"+"tracklet_"+next_layer+".jpg"
        destination_file= global_id_path+"/"+"tracklet_"+next_layer+".jpg"
        shutil.copyfile(source_file, destination_file)
        

        #Envleve les ids assoscies de la matrice (reduit la matrice)
        metrics_matrix[:,y] = inf
        metrics_matrix[x,:] = inf





# Les assosciations prioritaires ont ete faites, dans ce bloc on va traiter le cas ou une personne
# passer par deux cameras distantes, cad, par la couche/camera 1 et la couche/camera 3 
# Assosciations avec couches distantes restantes:
print("\n\n\t Assosications Distantes: \n")
for distance in range(2,4): #produit 2,3
    for camera in range(1,3): # produit 1 et 2
        current_layer = "camera" + str(camera) #peut etre couche 1 ou couche 2

        if distance + camera<=4 : #si sup  a 4 break -> couche distante valide
            couche_distante = "camera" + str(camera + distance) #peut etre couche 3 ou couche 4
            print("\n\nCurrent_layer est",current_layer,"couche_distante est",couche_distante)
            
            metrics_matrix=[]

            for current_id in Camera_to_ids_to_imgPaths[current_layer]:
                metrics_list = []
                linked= False
                if Cam_to_ids_to_savePaths[current_layer][current_id] is not None:
                    path_to_tracklet = Cam_to_ids_to_savePaths[current_layer][current_id] + "/"+"tracklet_camera"+str(camera+1)+".jpg"
                    if os.path.exists(path_to_tracklet):
                        print("\tLe noeud",current_layer, current_id,"est lie a la camera",str(camera+1), ",on reboucle car la couche inferieure aura une meilleure decision")
                        linked = True
                    #if Cam_to_ids_to_savePaths[current_layer][current_id] contient "tracklet_Cam"+str(camera+1)
                    # Va break si noeud de couche 1 est liee a la couche suivante alors couche suivante a une meilleur decision 
                    # dans tous les cas
                if not linked :
                    print("\tNoeud",current_layer, current_id, "n'est pas lie a sa couche inferieure, on peut donc voir si lie a sa couche distante")
                
                    for distant_id in Camera_to_ids_to_imgPaths[couche_distante]:
                        # comparer current_id avec distant_id 
                        # garde les comparaisons en memoire pour toute la couche current_layer
                        print("\tComparaison", current_layer, current_id, "avec",couche_distante,distant_id)
                        
                        # Calcul des vecteurs caracteristiques
                        feature_vec_Current_id = extractor(Camera_to_ids_to_imgPaths[current_layer][current_id])
                        feature_vec_compared_id = extractor(Camera_to_ids_to_imgPaths[couche_distante][distant_id])

                        # Calcul la matrices des distances
                        Distance_matrix = torchreid.metrics.distance.cosine_distance(feature_vec_Current_id, feature_vec_compared_id)

                        # Calcul la metrique, moyenne de tous les elements de la matrice
                        metrique = torch.mean(Distance_matrix)
                        
                        # Sauvegarder la metrique dans la matrice de layer
                        metrics_list.append(metrique)  
                    metrics_matrix.append(metrics_list)
            
            #boucle sur les couches
            if metrics_matrix==[]:
                print("\tAucun noeud de la couche ne peut prendre de decision valide ")
            else:
                print("\tMatrice comparaison",current_layer,"VS", couche_distante,metrics_matrix)
                metrics_matrix= np.matrix(metrics_matrix).transpose() 
                
                # Tant que la matrice n est pas vide et que le min est superieur au seuil, 
                # alors on a au moins une assosciation valide.
                seuil = 0.42
                while np.amin(metrics_matrix)!=inf and np.amin(metrics_matrix)<seuil : #!TODO il faut ajouter la conditions si rien ne corespons ; and min > metriques
                    #trouve le minimum dans la matrice et retourne les coordonnees
                    result = np.where(metrics_matrix == np.amin(metrics_matrix))
                    # zip the 2 arrays to get the exact coordinates
                    listOfCordinates = list(zip(result[0], result[1]))
                    
                    x = listOfCordinates[0][0] # ligne dans la matrice -> ids de la cameras inferieur
                    y = listOfCordinates[0][1] # colonne -> ids de la camera superieure
                    
                    #Convertit les index du min dans la matrice en ids (cles):
                    noeud_couche_sup = list(Cam_to_ids_to_savePaths[current_layer].keys())[y]
                    noeud_couche_inf = list(Cam_to_ids_to_savePaths[couche_distante].keys())[x]
                    
                    #Associe les ids
                    print("\t\n\tAssosciation possible !!: ",current_layer, noeud_couche_sup, "--->", couche_distante,noeud_couche_inf)

                    # ----> Ici on a donc une assosciation entre deux noeuds distants
                    
                    if Cam_to_ids_to_savePaths[couche_distante][noeud_couche_inf] is not None:
                        path_to_tracklet = Cam_to_ids_to_savePaths[couche_distante][noeud_couche_inf]+"/"+"tracklet_camera"+str(camera+distance-1)+".jpg"
                        print(path_to_tracklet)
                    # if Cam_to_ids_to_savePaths[couche_distante][distant_id] contient "tracklet_Cam"+str(camera+distance-1)
                    # break si noeud de couche 3 est lie a couche 2 a une decision prioritaire
                        if os.path.exists(path_to_tracklet):
                            print("\tAssosciation non valide",current_layer, noeud_couche_sup,"--//-->",couche_distante,noeud_couche_inf)
                            #Envleve les ids assoscies de la matrice (reduit la matrice)
                            metrics_matrix[:,y] = inf
                            metrics_matrix[x,:] = inf
                            continue
                    # ----> couche 3 pas lie a couche 2, couche 1 a une decision valide
                    
                    if Cam_to_ids_to_savePaths[couche_distante][noeud_couche_inf]==None:
                        # creer id global et affecter a couche 1 et 3 par ex
                        global_id +=1
                        print("\tNouvelle liaison distante cree avec id_global=", global_id)
                        global_id_path = "../reid_results/global_id"+str(global_id)
                        try:
                                os.makedirs(global_id_path)
                        except:
                            pass
                        Cam_to_ids_to_savePaths[current_layer][noeud_couche_sup]= global_id_path
                        Cam_to_ids_to_savePaths[couche_distante][noeud_couche_inf] = global_id_path
                        
                        #copie tracklet couche 1 
                        source_file = "../yolov4-deepsort/"+current_layer+"/"+noeud_couche_sup+"/"+"tracklet_"+current_layer+".jpg"
                        destination_file= global_id_path+"/"+"tracklet_"+current_layer+".jpg"
                        shutil.copyfile(source_file, destination_file)
                        
                        #copie tracklet couche 3
                        source_file = "../yolov4-deepsort/"+couche_distante+"/"+noeud_couche_inf+"/"+"tracklet_"+couche_distante+".jpg"
                        destination_file= global_id_path+"/"+"tracklet_"+couche_distante+".jpg"
                        shutil.copyfile(source_file, destination_file)

                    else :
                        #prendre id global de couche 3 et l affecter a la couche 1
                        Cam_to_ids_to_savePaths[current_layer][noeud_couche_sup] = Cam_to_ids_to_savePaths[couche_distante][noeud_couche_inf]
                        
                        #copier tracklet du neoud de couche 1 dans le dossier de la couche 3
                        source_file = "../yolov4-deepsort/"+current_layer+"/"+noeud_couche_sup+"/"+"tracklet_"+current_layer+".jpg"
                        destination_file= global_id_path+"/"+"tracklet_"+current_layer+".jpg"
                        shutil.copyfile(source_file, destination_file)
                    
                    #Envleve les ids assoscies de la matrice (reduit la matrice)
                    metrics_matrix[:,y] = inf
                    metrics_matrix[x,:] = inf







 #On a fait toutes les assosciation, pour les identifiants pendants (aucune assosciation)
 #On leur cree leur ids globaux\
print("\n\n\t Traitement des noeuds pendants:\n")
for camera in range(len(path_cams)-1):
    current_layer = "camera"+str(camera+1)
    for current_id in Cam_to_ids_to_savePaths[current_layer]:
        # A la fin, si aucun id global, creer l'id, affecter et copier le trackelt dans le bon dossier
        if Cam_to_ids_to_savePaths[current_layer][current_id] ==None:
            global_id +=1
            print("Le noeud",current_layer, current_id,"est pendant","\n on cree id_global=",global_id)
            global_id_path = "../reid_results/global_id"+str(global_id)
            try:
                    os.makedirs(global_id_path)
            except:
                pass
            Cam_to_ids_to_savePaths[current_layer][current_id]= global_id_path
            
            #Copie du tracklet dans le dossier d'id global
            source_file = "../yolov4-deepsort/"+current_layer+"/"+current_id+"/"+"tracklet_"+current_layer+".jpg"
            destination_file= global_id_path+"/"+"tracklet_"+current_layer+".jpg"
            shutil.copyfile(source_file, destination_file)

print("\n\n",Cam_to_ids_to_savePaths)                 


# imagesPerson1 = ["Cam1/ID1/1.png","Cam1/ID1/2.png","Cam1/ID1/2.png"]
# imagesPerson2 = ["Cam1/ID1/2.png","Cam1/ID1/1.png","Cam1/ID2/3.png"]
# imagesPerson3 = ["Cam1/ID3/1.png","Cam1/ID3/2.png","Cam1/ID2/2.png"]

# featureVecPerson1 = extractor(imagesPerson1)
# featureVecPerson2 = extractor(imagesPerson2)
# featureVecPerson3 = extractor(imagesPerson3)


# # # Verification difference entre meme image et differentes images
# Comp11 = torchreid.metrics.distance.cosine_distance(featureVecPerson1, featureVecPerson1)
# Comp12 = torchreid.metrics.distance.cosine_distance(featureVecPerson1, featureVecPerson2)
# Comp13 = torchreid.metrics.distance.cosine_distance(featureVecPerson1, featureVecPerson3)

# print("meme image comparaison :  ",Comp11, type(Comp11))
# print("images differentes :  ", Comp12)
# print("images differentes :  ", Comp13)

# # ici on decide de la metrique commme etant la somme du tenseur torch de comparaison
# metrique11 = torch.mean(Comp11)
# metrique12 = torch.mean(Comp12)
# metrique13 = torch.mean(Comp13)
# print("Metrique pour comparer la meme personne",metrique11)
# print("Metrique entre id1 et id2",metrique12)
# print("Metrique entre id1 et id3",metrique13)
