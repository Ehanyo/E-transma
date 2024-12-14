import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import face_recognition
import socket
import requests

# Dossier contenant les images à comparer
images_folder = 'images'

# Fonction pour nettoyer le dossier d'images
def clean_images_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Supprimé: {file_path}")
        except Exception as e:
            print(f"Erreur lors de la suppression de {file_path}: {e}")

# Charger le modèle pré-entraîné
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Initialiser la caméra OpenCV
cap = cv2.VideoCapture(0)

# Vérifie si la caméra s'est bien ouverte
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra")
    exit()

# Définit la largeur et la hauteur de la vidéo en Full HD
cap.set(3, 1920)  # 1920 pixels de large (Full HD)
cap.set(4, 1080)  # 1080 pixels de haut (Full HD)

# Créer une seule fenêtre pour l'affichage
cv2.namedWindow("Détection de personnes et de visages", cv2.WINDOW_NORMAL)

# Fichier JSON pour enregistrer les données de comptage de personnes
json_file = 'person_count.json'
api_url = 'http://16.171.141.132:8000/vehicles/receive-camera-data/'

# Créer le dossier pour les images s'il n'existe pas
if not os.path.exists(images_folder):
    os.makedirs(images_folder)
else:
    # Nettoyer le dossier d'images
    clean_images_folder(images_folder)

# Vérifier si le fichier JSON existe, sinon initialiser le contenu
if not os.path.exists(json_file):
    with open(json_file, 'w') as file:
        json.dump([], file)  # Initialiser avec une liste vide

# Lire le contenu du fichier JSON
def read_json(file_path):
    if os.path.getsize(file_path) == 0:  # Vérifie si le fichier est vide
        return []
    with open(file_path, 'r') as file:
        return json.load(file)

# Écrire dans le fichier JSON
def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Envoyer des données à l'API
def send_data_to_api(data):
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()  # Vérifie si la requête a réussi
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'envoi des données à l'API: {e}")

# Initialiser le nombre de personnes précédemment détectées
previous_person_count = 0
start_time = time.time()

# Charger et encoder les visages des images dans le dossier
def load_and_encode_faces(folder_path):
    encodings = []
    names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])
                names.append(filename)  # Utilise le nom du fichier comme identifiant
    return encodings, names

# Comparer les visages dans une nouvelle image avec les visages connus
def compare_faces(image_path, known_encodings):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    results = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            results.append((True, known_face_names[first_match_index]))
        else:
            results.append((False, None))
    return results

# Initialiser les encodages des visages connus
known_face_encodings, known_face_names = load_and_encode_faces(images_folder)

# Variable pour stocker le nom du fichier de l'image précédente
previous_image_filename = None

while True:
    # Lire une image depuis la caméra OpenCV
    ret, frame = cap.read()

    # Si la capture OpenCV a réussi
    if not ret:
        print("Erreur lors de la capture vidéo")
        break

    # Obtenir les dimensions de l'image
    (h, w) = frame.shape[:2]

    # Prétraiter l'image pour le modèle
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Initialiser le compteur de personnes
    current_person_count = 0

    # Parcourir les détections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filtrer les détections avec une confiance suffisante
        if confidence > 0.7:
            idx = int(detections[0, 0, i, 1])
            
            # Vérifier si la détection correspond à une personne (classe 15 pour MobileNet SSD)
            if idx == 15:
                current_person_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Dessiner une boîte autour de la personne détectée
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"Person: {confidence:.2f}"
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Vérifier si 10 secondes se sont écoulées
    if time.time() - start_time >= 10:
        # Sauvegarder une photo dans le dossier 'images'
        new_image_filename = f"{images_folder}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(new_image_filename, frame)

        if previous_image_filename:
            # Comparer la nouvelle image avec la précédente
            results = compare_faces(new_image_filename, known_face_encodings)

            # Compter les nouveaux visages qui ne correspondent pas aux visages précédemment détectés
            new_faces_detected = 0
            for result in results:
                if not result[0]:  # Si le visage est nouveau
                    new_faces_detected += 1
                    print("Nouveau visage détecté")

            if new_faces_detected > 0:
                # Lire les données JSON existantes
                data = read_json(json_file)
                # Mettre à jour le fichier JSON avec les nouveaux visages détectés
                for i in range(new_faces_detected):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_entry = {
                        "vehicle": 1,
                        "camera_id": socket.gethostname(),
                        "passengers_count": 1,
                        "timestamp": timestamp
                    }
                    data.append(new_entry)

                write_json(data, json_file)
                # Envoyer les données à l'API
                send_data_to_api(new_entry)

                # Mettre à jour les encodages des visages connus
                known_face_encodings, known_face_names = load_and_encode_faces(images_folder)

            # Supprimer l'ancienne image
            if os.path.exists(previous_image_filename):
                os.remove(previous_image_filename)
        
        # Mettre à jour l'image précédente
        previous_image_filename = new_image_filename
        start_time = time.time()

    # Afficher le nombre de personnes détectées sur l'image
    cv2.putText(frame, f"Personnes trouvées: {current_person_count}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Afficher l'image avec les détections
    cv2.imshow("Détection de personnes et de visages", frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
