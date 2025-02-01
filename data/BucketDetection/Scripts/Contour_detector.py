import cv2
import matplotlib.pyplot as plt
import numpy as np

# TODO Rename pour une seule langue -> En ou FR

class ContoursDetector:
    def __init__(self):
        pass

    contour_type = ["cercle", "box"] # TODO : à utiliser pour se focus sur un type de contour

    # TODO : Méthode pour prétraiter les images
    def pretraiter_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # TODO : Ajouter d'autre prétraitement pour détecter les contours au besoin

        # Appliquer un flou gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        return blurred

    # Méthode pour faire la détection des contours
    # TODO : Ajouter en paramètre le type de contour qu'on veut détecter
    def detect_contours(self, image):
        # Prétraitement
        image_pretraitee = self.pretraiter_image(image)

        # Détection des contours avec Canny
        edges = cv2.Canny(image_pretraitee, threshold1=50, threshold2=150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Seuil d'aire à ajuster
        edges_filtered = np.zeros_like(edges)
        cv2.drawContours(edges_filtered, filtered_contours, -1, 255, thickness=cv2.FILLED)

        # Image vide pour dessiner les contours des cercles uniquement
        edges_filtered = np.zeros_like(edges)
        # Créer une image vide pour dessiner les contours
        contours_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # TODO : Trier les contours pour ne détecter que les cercles
        # Utilisation de HoughCircles pour détecter que les cercle
        circles = cv2.HoughCircles(
            image_pretraitee,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=150,
            param1=50,
            param2=60,
            minRadius=20,
            maxRadius=150
        )

        # Si des cercles sont détectés, les dessiner
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Dessiner uniquement les cercles sur l'image filtrée
                cv2.circle(edges_filtered, (circle[0], circle[1]), circle[2], 255, 2)  # Contour blanc sur fond noir
                cv2.circle(contours_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)  # Contour du cercle
                cv2.circle(contours_image, (circle[0], circle[1]), 2, (0, 0, 255), 3)  # Centre du cercle
        else :
            circles = []

        return edges_filtered, contours_image, circles



    # Méthode pour récupérer l'image si elle existe
    def get_image(self, chemin_image):
        try:
            image = cv2.imread(chemin_image)
            if image is None:
                raise FileNotFoundError(f"L'image '{chemin_image}' n'a pas été trouvée.")
            return image
        except Exception as e:
            print(f"Erreur lors du chargement de l'image : {e}")
            return None

    # Méthode pour afficher les différentes images déjà annotées avec les contours
    def afficher_images(self, image, edges):
        # Affichage des résultats
        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.title("Image originale")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title("Contours détectés")
        plt.imshow(edges, cmap='gray')
        plt.show()

    # Méthode pour ajouter un carré autour du cercle détecter
    # TODO : Mieux détecter les cercles pour les encadrer avec le box
    def bounding_box(self, image):
        # Obtenir les contours et l'image avec contours
        edges, image_contours, circles = self.detect_contours(image)

        # TODO : Ajouter une liste pour stocker les différentes carré détecté
        box_list =[]

        # TODO : Ajouter les bounding box autour des cercles détectées
        # Si des cercles sont détectés
        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Dessiner une boîte autour de chaque cercle détecté
            for circle in circles[0, :]:
                # Coordonnées du cercle
                x, y, radius = circle

                # Dessiner le carré autour du cercle
                top_left = (x - radius, y - radius)  # Coin supérieur gauche
                bottom_right = (x + radius, y + radius)  # Coin inférieur droit
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0),
                              2)  # Dessiner la boîte en vert avec une épaisseur de 2

                # Optionnel : dessiner le cercle aussi (pour visualiser les cercles)
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)  # Contour du cercle
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # Centre du cercle

        return image, image_contours

        return image, image_contours

    # Fonction principale pour la webcam
    def webcam_detection(self):
        detector = ContoursDetector()
        cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

        if not cap.isOpened():
            print("Erreur : Impossible d'accéder à la webcam.")
            return

        print("Appuyez sur 'q' pour quitter.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur : Impossible de lire la frame depuis la webcam.")
                break

            # Détecter les contours sur la frame
            edges, contours_image = detector.detect_contours(frame)  # On récupère les deux images

            # Afficher la frame avec les cercles détectés
            cv2.imshow("Contours détectés", contours_image)  # Afficher contours_image, pas un tuple

            # Quitter la boucle avec la touche 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # TODO : Méthode pour aligner le centre au caré détecter
    def center_bucket(self, image):
        pass



# TODO : Filtrer les contours trouver si besoin
#  (Concentrer sur les cercles si c'est le contour qu'on aura)

# TODO : Ajouter un centre à l'image -> Aidera pour l'ajustement par la suite

# TODO : Créer un nouveau répertoire avec les images avec les contours détecter

# TODO : Ajouter les images avec les contours dans le répertoire

# --------------------------------------
# TODO : Étape 1 : Ajouter la possibilité d'analyser une vidéo
# TODO : Étape 2 : Ajouter la possibilité de le faire en temps réel


# TODO : Ajouter une méthode pour centrer le carré du milieu à celui autour de l'ouverture du bucket

# TODO : Faire en sorte que ça fonctionne avec la webcam