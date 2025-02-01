# Imports
from Contour_detector import ContoursDetector

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def run_detector():
    contour_detector = ContoursDetector()
    contour_detector.webcam_detection()
    # TODO Faire en sorte que ça fonctionne avec la webcam
    #chemin_image_test = '../Data/Poubelle.jpg' # TODO : Faire avec un prompt
    # ... ou parcourir le répertoire et le faire pour toutes les images qui sont dedans

    # Charger l'image
    #image = contour_detector.get_image(chemin_image_test)
    #if image is None:
    #    return

    # Détecter les contours
    # edges = contour_detector.detect_contours(image)

    # Afficher les résultats
    # contour_detector.afficher_images(image, edges)

    # Ajouter les bounding boxes autour des cercles détectés
    #image_with_boxes, edges = contour_detector.bounding_box(image)
    #contour_detector.afficher_images(image_with_boxes, edges)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
