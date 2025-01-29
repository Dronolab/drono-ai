import cv2
import matplotlib.pyplot as plt

# TODO Rename pour une seule langue -> En ou FR

class ContoursDetector:
    def __init__(self):
        pass

    # TODO : Méthode pour prétraiter les images
    def pretraiter_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # TODO : Ajouter d'autre prétraitement pour détecter les contours au besoin

        # Appliquer un flou gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    # TODO : Créer une méthode qui sera par la suite appelée dans main
    # Méthode pour faire la détection des contours
    def detect_contours(self, image):
        # Prétraitement
        image_pretraitee = self.pretraiter_image(image)

        # Détection des contours avec Canny
        edges = cv2.Canny(image_pretraitee, threshold1=50, threshold2=150)

        # Créer une image vide pour dessiner les contours
        contours_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges, contours_image



    # Méthode pour récupérer l'image
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
    def bounding_box(self, image):
        # Obtenir les contours et l'image avec contours
        edges, image_contours = self.detect_contours(image)

        max_radius = 0  # Stocker le plus grand rayon
        max_circle = None  # Stocker les informations du plus grand cercle

        # Parcourir chaque pixel pour détecter les zones de contours
        # Cela suppose que vous avez une image binaire avec des pixels blancs pour les contours
        contours = cv2.connectedComponentsWithStats(edges, connectivity=8)[2]

        for stats in contours:
            x, y, w, h, area = stats  # Coordonnées du rectangle englobant, largeur, hauteur et aire
            radius = min(w, h) / 2  # Approximer un rayon basé sur la taille du rectangle englobant

            # Vérifier si c'est le cercle le plus grand détecté
            if radius > max_radius:
                max_radius = radius
                max_circle = (x + w // 2, y + h // 2, radius)  # Centre et rayon estimés

        # Dessiner le rectangle englobant autour du plus grand cercle
        if max_circle:
            x, y, radius = max_circle
            x, y, radius = int(x), int(y), int(radius)
            cv2.rectangle(
                image,
                (x - radius, y - radius),  # Coin supérieur gauche
                (x + radius, y + radius),  # Coin inférieur droit
                (255, 0, 0),  # Couleur : bleu
                2,  # Épaisseur
            )

        return image, image_contours





# TODO : Filtrer les contours trouver si besoin
#  (Concentrer sur les cercles si c'est le contour qu'on aura)

# TODO : Ajouter un centre à l'image -> Aidera pour l'ajustement par la suite

# TODO : Créer un nouveau répertoire avec les images avec les contours détecter

# TODO : Ajouter les images avec les contours dans le répertoire