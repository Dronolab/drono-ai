from ultralytics import YOLO
import cv2
import sys

def main():
    # Vérifiez qu'un chemin d'image est fourni en argument
    if len(sys.argv) < 2:
        print("Usage: python inference.py <chemin_de_l_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Charger le modèle YOLOv8 (.pt)
    # Remplacez "votre_modele.pt" par le chemin vers votre modèle fine-tuné
    model = YOLO(r"C:\Users\Hinnovis\OneDrive\Desktop\Dronolab ets\drono-ai\models\best1.pt")
    
    # Effectuer l'inférence sur l'image
    results = model(image_path)
    
    # On peut accéder aux prédictions (pour la première image, results[0])
    # Par exemple, pour afficher les coordonnées des boîtes englobantes :
    # Filtrer les boîtes avec une confiance inférieure à 0.80
    filtered_boxes = [box for box in results[0].boxes if float(box.conf.cpu().numpy()) >= 0.80]

# Remplacer la liste originale par celle filtrée
    results[0].boxes = filtered_boxes

# Affichage des boîtes retenues
    for box in results[0].boxes:
        print("Boîte :", box.xyxy.cpu().numpy(), "Confiance :", box.conf.cpu().numpy())

    
    # Visualiser le résultat annoté
    annotated_img = results[0].plot()  # Renvoie l'image annotée
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print("Inférence terminée. Résultat sauvegardé dans", output_path)

if __name__ == "__main__":
    main()
