# Imports
from Contour_detector import ContoursDetector

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def web_cam_run():
    contour_detector = ContoursDetector()
    contour_detector.webcam_detection()

def run_detector():
    contour_detector = ContoursDetector()

    # TODO Faire en sorte que ça fonctionne avec la webcam
    chemin_image_test = '../Data/Sceau_bleu.jpg' # TODO : Faire avec un prompt
    # ... ou parcourir le répertoire et le faire pour toutes les images qui sont dedans

    # Charger l'image
    image = contour_detector.get_image(chemin_image_test)
    if image is None:
        return

    # Détection et ajout bounding_box autour des cercles
    image_with_boxes, edges = contour_detector.bounding_box(image)
    # Afficher les images avec les contours et les cercles détectées
    contour_detector.afficher_images(image_with_boxes, edges)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_detector()
    # web_cam_run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
