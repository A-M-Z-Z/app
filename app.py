from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import face_recognition
import pickle
import os
import csv
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import uuid
import numpy as np
from pathlib import Path
from functools import wraps
from typing import List, Dict, Tuple, Optional, Any, Union
from PIL import Image
import threading

# Configuration de l'application
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or str(uuid.uuid4())
    UPLOAD_FOLDER = Path('face_images')
    DATA_FOLDER = Path('data')
    ENCODING_FILE = DATA_FOLDER / 'encodings.pickle'
    PRESENCE_FILE = DATA_FOLDER / 'presence.csv'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    FACE_MATCH_TOLERANCE = 0.45
    PRESENCE_TIMEOUT = 60  # secondes
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    # Paramètres pour améliorer les performances
    MAX_IMAGE_WIDTH = 640  # Largeur maximale pour le traitement
    FACE_DETECTION_MODEL = "hog"  # "hog" (rapide) ou "cnn" (plus précis mais plus lent)
    THREADING_ENABLED = True  # Utiliser le threading pour les opérations longues

# Initialisation de l'application
app = Flask(__name__)
app.config.from_object(Config)

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG if app.config['DEBUG'] else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Création des dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Modèle de données
class FaceDatabase:
    def __init__(self, encoding_file: Path):
        self.encoding_file = encoding_file
        self.data: Dict[str, List] = {"encodings": [], "names": []}
        self.load()
    
    def load(self) -> None:
        """Charge les encodages depuis le fichier pickle"""
        if self.encoding_file.exists() and self.encoding_file.stat().st_size > 0:
            try:
                with open(self.encoding_file, "rb") as f:
                    self.data = pickle.load(f)
                logger.info(f"Base de données chargée avec {len(self.data['names'])} visages")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la base de données: {e}")
    
    def save(self) -> None:
        """Sauvegarde les encodages dans le fichier pickle"""
        try:
            with open(self.encoding_file, "wb") as f:
                pickle.dump(self.data, f)
            logger.info(f"Base de données sauvegardée avec {len(self.data['names'])} visages")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la base de données: {e}")
    
    def add_face(self, name: str, encoding: np.ndarray) -> None:
        """Ajoute un visage à la base de données"""
        self.data["encodings"].append(encoding)
        self.data["names"].append(name)
        self.save()
    
    def delete_face(self, name: str) -> bool:
        """Supprime un visage de la base de données"""
        if name in self.data["names"]:
            indices = [i for i, n in enumerate(self.data["names"]) if n == name]
            for index in sorted(indices, reverse=True):
                self.data["names"].pop(index)
                self.data["encodings"].pop(index)
            self.save()
            return True
        return False
    
    def compare_face(self, encoding: np.ndarray, tolerance: float = 0.45) -> Optional[str]:
        """Compare un encodage avec la base de données et retourne le nom si trouvé"""
        if not self.data["encodings"]:
            return None
        
        matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=tolerance)
        
        if True in matches:
            matched_index = matches.index(True)
            return self.data["names"][matched_index]
        return None
    
    def get_all_faces(self) -> List[str]:
        """Retourne la liste de tous les visages"""
        return sorted(set(self.data["names"]))

class PresenceTracker:
    def __init__(self, presence_file: Path):
        self.presence_file = presence_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """S'assure que le fichier de présence existe avec un en-tête"""
        if not self.presence_file.exists():
            with open(self.presence_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Nom", "Date", "Heure"])
    
    def log_presence(self, name: str) -> None:
        """Enregistre une présence dans le fichier CSV"""
        now = datetime.now()
        with open(self.presence_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        logger.info(f"Présence enregistrée pour {name}")
    
    def recent_presence_exists(self, name: str, timeout_seconds: int = 60) -> bool:
        """Vérifie si une présence récente existe pour le nom donné"""
        if not self.presence_file.exists():
            return False
        
        now = datetime.now()
        with open(self.presence_file, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 3 and row[0] == name:
                    try:
                        date_str, time_str = row[1], row[2]
                        last_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        if (now - last_time).total_seconds() < timeout_seconds:
                            return True
                    except Exception as e:
                        logger.error(f"Erreur lors du parsing de la date: {e}")
        return False
    
    def get_all_presences(self) -> List[List[str]]:
        """Retourne toutes les présences"""
        presences = []
        if self.presence_file.exists():
            with open(self.presence_file, newline="") as f:
                reader = csv.reader(f)
                presences = list(reader)
        return presences

# Utilitaires
def allowed_file(filename: str) -> bool:
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(file) -> Tuple[bool, Union[np.ndarray, str]]:
    """Traite une image et retourne l'image chargée ou un message d'erreur"""
    if file.filename == '':
        return False, "Aucun fichier sélectionné"
    
    if not allowed_file(file.filename):
        return False, "Format de fichier non autorisé. Utilisez JPG, JPEG ou PNG."
    
    try:
        image = face_recognition.load_image_file(file)
        return True, image
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {e}")
        return False, f"Erreur lors du traitement de l'image: {str(e)}"

def save_uploaded_face(file, fullname: str) -> str:
    """Sauvegarde l'image du visage téléchargée"""
    filename = secure_filename(f"{fullname.replace(' ', '_')}_{uuid.uuid4().hex}.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.seek(0)  # Réinitialiser le pointeur du fichier
    file.save(filepath)
    return filepath

# Initialisation des services
face_db = FaceDatabase(app.config['ENCODING_FILE'])
presence_tracker = PresenceTracker(app.config['PRESENCE_FILE'])

# Routes
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Validation des entrées
        name = request.form.get('name', '').strip()
        surname = request.form.get('surname', '').strip()
        
        if not name or not surname:
            flash("Veuillez fournir un nom et un prénom", "danger")
            return render_template('register.html', error="Veuillez fournir un nom et un prénom")
        
        fullname = f"{name} {surname}"
        
        if 'image' not in request.files:
            flash("Aucun fichier image reçu", "danger")
            return render_template('register.html', error="Aucun fichier image reçu")
        
        file = request.files['image']
        if file.filename == '':
            flash("Aucun fichier sélectionné", "danger")
            return render_template('register.html', error="Aucun fichier sélectionné")
            
        try:
            success, result = process_image(file)
            if not success:
                flash(result, "danger")
                return render_template('register.html', error=result)
                
            image = result
            
            # Redimensionner l'image pour améliorer les performances
            height, width = image.shape[:2]
            if width > app.config['MAX_IMAGE_WIDTH']:
                scale = app.config['MAX_IMAGE_WIDTH'] / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_small = np.array(Image.fromarray(image).resize((new_width, new_height)))
            else:
                image_small = image
                
            # Extraction des caractéristiques du visage avec modèle "hog" plus rapide
            face_locations = face_recognition.face_locations(image_small, model="hog")
            
            if not face_locations:
                flash("Aucun visage détecté dans l'image", "warning")
                return render_template('register.html', error="Aucun visage détecté")
            
            if len(face_locations) > 1:
                flash("Plusieurs visages détectés. Veuillez soumettre une image avec un seul visage", "warning")
                return render_template('register.html', error="Plusieurs visages détectés")
            
            # Si l'image a été redimensionnée, ajuster les coordonnées
            if width > app.config['MAX_IMAGE_WIDTH']:
                face_locations = [(int(top/scale), int(right/scale), 
                                int(bottom/scale), int(left/scale)) 
                                for top, right, bottom, left in face_locations]
            
            # Encodage et sauvegarde
            face_encodings = face_recognition.face_encodings(image, face_locations)
            face_db.add_face(fullname, face_encodings[0])
            
            # Sauvegarde de l'image
            file.seek(0)  # Réinitialiser le pointeur du fichier
            save_uploaded_face(file, fullname)
            
            flash(f"Visage de {fullname} enregistré avec succès", "success")
            return redirect(url_for('list_faces'))
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement: {e}")
            flash(f"Erreur lors de l'enregistrement: {str(e)}", "danger")
            return render_template('register.html', error=str(e))
    
    return render_template('register.html')
    
    return render_template('register.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image reçue"}), 400
        
        file = request.files['image']
        success, result = process_image(file)
        
        if not success:
            return jsonify({"error": result}), 400
        
        image = result
        
        # Redimensionner l'image pour améliorer les performances
        # Cette étape peut considérablement accélérer la détection
        height, width = image.shape[:2]
        if width > 640:
            # Redimensionner l'image tout en conservant les proportions
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_small = np.array(Image.fromarray(image).resize((new_width, new_height)))
        else:
            image_small = image
        
        # Détection des visages avec model="hog" pour plus de rapidité
        # Le modèle "hog" est moins précis mais beaucoup plus rapide que "cnn"
        face_locations = face_recognition.face_locations(image_small, model="hog")
        if not face_locations:
            return jsonify({"warning": "Aucun visage détecté", "names": [], "boxes": []}), 200
        
        # Si l'image a été redimensionnée, ajuster les coordonnées
        if width > 640:
            face_locations = [(int(top/scale), int(right/scale), 
                              int(bottom/scale), int(left/scale)) 
                              for top, right, bottom, left in face_locations]
        
        # Reconnaissance des visages
        # Utiliser l'image originale pour l'encodage afin de maintenir la précision
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        names = []
        for encoding in face_encodings:
            name = face_db.compare_face(encoding, app.config['FACE_MATCH_TOLERANCE']) or "Inconnu"
            names.append(name)
            
            # Enregistrement de la présence si la personne est reconnue
            if name != "Inconnu" and not presence_tracker.recent_presence_exists(name, app.config['PRESENCE_TIMEOUT']):
                presence_tracker.log_presence(name)
        
        # Conversion des coordonnées pour l'affichage
        face_locations_json = [
            {"top": top, "right": right, "bottom": bottom, "left": left}
            for top, right, bottom, left in face_locations
        ]
        
        return jsonify({"names": names, "boxes": face_locations_json})
    
    return render_template('recognize.html')

@app.route('/faces')
def list_faces():
    faces = face_db.get_all_faces()
    return render_template("faces.html", faces=faces)

@app.route('/delete_face', methods=['POST'])
def delete_face():
    name = request.form.get('name', '')
    if not name:
        flash("Nom manquant", "danger")
        return redirect(url_for('list_faces'))
    
    if face_db.delete_face(name):
        # Suppression des images associées
        for img_file in app.config['UPLOAD_FOLDER'].glob(f"{name.replace(' ', '_')}*.jpg"):
            try:
                os.remove(img_file)
            except Exception as e:
                logger.error(f"Erreur lors de la suppression de l'image {img_file}: {e}")
        
        flash(f"Visage {name} supprimé avec succès", "success")
    else:
        flash("Nom introuvable", "danger")
    
    return redirect(url_for('list_faces'))

@app.route('/presences')
def view_presences():
    presences = presence_tracker.get_all_presences()
    return render_template("presences.html", presences=presences)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Créer des pages d'erreur personnalisées
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page non trouvée", code=404), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Erreur serveur", code=500), 500

# Point d'entrée de l'application
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000, threaded=True)