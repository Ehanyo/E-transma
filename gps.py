import gpsd
import time
import requests
import json

# Connexion à GPSD
gpsd.connect()

# URL du serveur
pi_url = 'http://16.171.141.132:8000/vehicles/receive-camera-data/'

try:
    # Boucle pour récupérer les données GPS toutes les 5 secondes
    while True:
        # Lecture des données GPS
        packet = gpsd.get_current()

        # Vérifie si le paquet contient des données de position
        if packet.mode >= 2:  # mode 2 = données 2D disponibles (latitude, longitude)
            latitude = packet.lat
            longitude = packet.lon
            print(f"Latitude: {latitude}, Longitude: {longitude}")

            # Créer le dictionnaire avec les données à envoyer
            data = {
                'latitude': latitude,
                'longitude': longitude
            }

            # Convertir le dictionnaire en JSON
            json_data = json.dumps(data)

            # Envoyer les données au serveur
            response = requests.post(pi_url, json=data)

            # Vérifier la réponse du serveur
            if response.status_code == 200:
                print("Données envoyées avec succès")
            else:
                print(f"Erreur lors de l'envoi des données: {response.status_code}")

        else:
            print("Données de position non disponibles")

        # Attendre 5 secondes avant de vérifier à nouveau
        time.sleep(5)

except KeyboardInterrupt:
    print("\nProgramme arrêté.")
