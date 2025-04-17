import io
import logging
import torch
import numpy as np
import pytesseract
import re
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights

logger = logging.getLogger(__name__)

# ImageNet class mapping
# We'll load this dynamically from the model

# Definieren von Produktkategorien basierend auf erkanntem Text
PRODUCT_TEXT_MAPPING = {
    # Mehl-Kategorien
    "Weizenmehl": {"name": "Weizenmehl", "description": "Weizenmehl", "confidence_boost": 0.8},
    "Mehl": {"name": "Mehl", "description": "Mehl", "confidence_boost": 0.6},
    "TYP 405": {"name": "Weizenmehl_405", "description": "Weizenmehl Type 405", "confidence_boost": 0.7},
    
    # Milchprodukte
    "Milch": {"name": "Milch", "description": "Milch", "confidence_boost": 0.7},
    "Vollmilch": {"name": "Vollmilch", "description": "Vollmilch", "confidence_boost": 0.8},
    "Joghurt": {"name": "Joghurt", "description": "Joghurt", "confidence_boost": 0.7},
    
    # Getränke
    "Cola": {"name": "Cola", "description": "Cola", "confidence_boost": 0.7},
    "Fanta": {"name": "Fanta", "description": "Fanta", "confidence_boost": 0.7},
    "Wasser": {"name": "Wasser", "description": "Wasser", "confidence_boost": 0.6},
    
    # Snacks
    "Schokolade": {"name": "Schokolade", "description": "Schokolade", "confidence_boost": 0.7},
    "Chips": {"name": "Chips", "description": "Chips", "confidence_boost": 0.7},
    
    # Konserven
    "Tomaten": {"name": "Tomaten", "description": "Tomaten", "confidence_boost": 0.6},
    "Mais": {"name": "Mais", "description": "Mais", "confidence_boost": 0.6},
    
    # Allgemeine Lebensmittelkategorien
    "Bio": {"name": "Bio_Produkt", "description": "Bio-Produkt", "confidence_boost": 0.5},
    "Zucker": {"name": "Zucker", "description": "Zucker", "confidence_boost": 0.7},
    "Salz": {"name": "Salz", "description": "Salz", "confidence_boost": 0.7}
}

class ImageRecognizer:
    def __init__(self):
        """Initialize the image recognition model"""
        try:
            logger.info("Loading ResNet50 model...")
            # Load pre-trained ResNet50 model
            self.weights = ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=self.weights)
            self.model.eval()  # Set model to evaluation mode
            
            # Get the preprocessing transform from the weights
            self.preprocess = self.weights.transforms()
            
            # Get class names from the weights metadata
            self.categories = self.weights.meta["categories"]
            
            logger.info("Model loaded successfully with %d categories", len(self.categories))
            
            # Configure pytesseract
            self.tesseract_config = '--psm 11 --oem 3'  # Page segmentation mode and OCR Engine mode
            logger.info("OCR text recognition initialized with text-based product mapping")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def extract_text(self, img):
        """
        Extract text from the image using OCR with improved processing
        for better recognition of expiry dates and product information
        
        Args:
            img: PIL Image object
            
        Returns:
            Extracted text as string
        """
        try:
            # Erstelle mehrere Bildvarianten für bessere Texterkennung
            results = []
            
            # Original grayscale conversion - Standard-Ansatz
            gray_img = img.convert('L')
            text1 = pytesseract.image_to_string(gray_img, config=self.tesseract_config)
            if text1.strip():
                results.append(text1)
            
            # Versuch mit angepasstem Kontrast für kleine Texte wie Ablaufdaten
            try:
                from PIL import ImageEnhance
                # Kontrast erhöhen für bessere Erkennung kleiner Texte (wie MHD)
                enhancer = ImageEnhance.Contrast(gray_img)
                high_contrast = enhancer.enhance(2.0)  # Kontrast verdoppeln
                text2 = pytesseract.image_to_string(high_contrast, config='--psm 6 --oem 3')  # Einzelne Textzeile
                if text2.strip():
                    results.append(text2)
            except Exception as contrast_error:
                logger.debug(f"Error during contrast enhancement: {contrast_error}")
            
            # Weitere Verarbeitung für MHD-spezifische Texterkennung
            try:
                # MHD-spezifische OCR-Konfiguration für Datumserkennung
                text3 = pytesseract.image_to_string(gray_img, config='--psm 3 -c tessedit_char_whitelist="0123456789MHD./-: "')
                if text3.strip():
                    results.append(text3)
            except Exception as mhd_error:
                logger.debug(f"Error during MHD-specific recognition: {mhd_error}")
            
            # Kombiniere alle Ergebnisse
            all_text = '\n'.join(results)
            
            # Filter out empty lines and clean up the text
            text_lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            clean_text = ' '.join(text_lines)
            
            logger.info(f"Extracted text: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}")
            return clean_text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

    def is_meaningful_text(self, text):
        """
        Prüft, ob der erkannte Text sinnvoll ist (nicht nur zufällige Zeichen)
        Verwendet sehr strenge Regeln, um nur wirklich sinnvollen Text zu erkennen
        
        Args:
            text: Extrahierter Text
            
        Returns:
            Boolean, ob der Text sinnvoll erscheint
        """
        if not text or len(text.strip()) < 6:  # Mindestens 6 Zeichen (strenger)
            return False
        
        # Wenn zu viele Sonderzeichen vorhanden sind, sofort ablehnen
        special_chars_count = sum(not c.isalnum() and not c.isspace() for c in text)
        if special_chars_count / (len(text) + 0.001) > 0.3:  # Mehr als 30% Sonderzeichen
            logger.info(f"Rejected text with too many special characters: {text}")
            return False
            
        # Entferne Sonderzeichen für die Analyse
        cleaned_text = re.sub(r'[^\w\säöüÄÖÜß]', '', text)
        cleaned_text = cleaned_text.strip()
        
        # Mindestlänge nach Bereinigung - sehr streng
        if len(cleaned_text) < 6:
            return False
        
        # Entferne kleine sonderbare Wörter (oft OCR-Fehler)
        words = cleaned_text.split()
        
        # Extrahiere nur Wörter mit mindestens 5 Buchstaben (noch strenger)
        meaningful_words = [w for w in words if len(w) >= 5 and w.isalpha()]
        
        # Wenn keine aussagekräftigen Wörter gefunden wurden
        if len(meaningful_words) < 1:
            return False
        
        # Prüfe auf bekannte Wörter und Muster (deutsches Wörterbuch oder häufige Wörter)
        # Diese Liste könnte durch ein vollständiges Wörterbuch ersetzt werden
        common_german_words = ["und", "der", "die", "das", "mit", "für", "von", "bei", "aus", "Buch", 
                             "Text", "Wort", "Preis", "Euro", "Kauf", "Geld", "Zeit", "Jahr", "Leben",
                             "Mensch", "Mann", "Frau", "Kind", "Stadt", "Land", "Haus", "Auto", "Wasser",
                             "Essen", "Milch", "Brot", "Zucker", "Salz", "Butter", "Käse", "Fleisch", 
                             "Obst", "Gemüse", "Wein", "Bier", "Hotel", "Restaurant", "Schule", "Arbeit",
                             "Schrift", "Brief", "Post", "Zeitung", "Markt", "Straße", "Zimmer", "Tisch",
                             "Stuhl", "Bett", "Küche", "Bad", "Wohnung", "Fenster", "Tür", "Dach", "Wand",
                             "Mehl", "Weizen", "Weißmehl", "Weizenmehl", "Vollkorn", "Vollkornmehl", "Teig", 
                             "Teigwaren", "Pasta", "Spaghetti", "Nudel", "Nudeln", "Reis", "Kartoffel", 
                             "Kartoffeln", "Getränk", "Getränke", "Cola", "Limonade", "Saft", "Apfel",
                             "Banane", "Orange", "Birne", "Erdbeere", "Himbeere", "Brombeere", "Heidelbeere"]
        
        # Zähle, wie viele erkannte Wörter in der "bekannten Wörter"-Liste vorkommen
        known_words = [w.lower() for w in words if w.lower() in common_german_words]
        
        # Sehr strenge Regeln für Wortgröße und Bekanntheitsgrad
        if len(known_words) >= 2 or (len(known_words) >= 1 and len(meaningful_words) >= 2):
            # Prüfe zusätzlich, ob das Verhältnis von Buchstaben zu Gesamtzeichen sehr hoch ist
            letters = sum(c.isalpha() for c in cleaned_text)
            if letters / (len(cleaned_text) + 0.001) > 0.7:  # Mindestens 70% Buchstaben (noch strenger)
                logger.info(f"Meaningful text validation passed with words: {', '.join(meaningful_words)}")
                
                # Zusätzliche spezifische Prüfung auf "Wortbildung"
                # Zähle, wie viele aufeinanderfolgende Wörter es gibt
                consecutive_words = 0
                for i in range(len(words) - 1):
                    if words[i].isalpha() and words[i+1].isalpha():
                        consecutive_words += 1
                
                if consecutive_words > 0:
                    logger.info(f"Found {consecutive_words} consecutive words - valid text")
                    return True
                else:
                    # Wenn es keine aufeinanderfolgenden Wörter gibt, muss es zumindest 
                    # ein längeres, bekanntes Wort sein
                    longer_known_words = [w for w in known_words if len(w) >= 6]
                    if longer_known_words:
                        logger.info(f"Found longer known word: {longer_known_words[0]}")
                        return True
                    else:
                        return False
                
        # Spezifische Produktmuster erkennen
        # Weizenmehl + Typ
        if re.search(r'(weizen|mehl|weizenmehl|vollkorn).*?typ\s*\d+', cleaned_text.lower()):
            logger.info("Found specific product pattern: Mehl + Typ")
            return True
            
        # Special patterns - erkenne spezifische Textmuster
        if "www." in text.lower() and ".de" in text.lower():
            logger.info("Found website URL")
            return True  # Webseiten-URL
        
        if re.search(r'\d+[,.]\d+\s*€', text):
            logger.info("Found price information")
            return True  # Preisangabe mit Dezimalstellen
        
        # ISBN Muster (Bücher)    
        if re.search(r'ISBN.*?\d[-\s]?\d', text):
            logger.info("Found ISBN number")
            return True
            
        return False

    def get_best_text_for_title(self, text):
        """
        Extrahiert den besten Text für die Verwendung als Produktname
        Verbesserte Version mit strengerer Filterung
        
        Args:
            text: Extrahierter Text aus dem Bild
            
        Returns:
            Aufbereiteter Text als String, der als Produktname verwendet werden kann
        """
        if not text:
            return "Text"
            
        # Entferne Sonderzeichen und überflüssige Leerzeichen
        cleaned_text = re.sub(r'[^\w\säöüÄÖÜß]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Wenn kein sinnvoller Text übrig bleibt
        if not cleaned_text or len(cleaned_text) < 3:
            return "Text"
        
        # Liste von Wörtern, die aus einem Produktnamen herausgefiltert werden sollten
        filter_words = ["typ", "type", "ml", "l", "g", "kg", "m", "cm", "mm", "the", "und"]
        
        # Teile den Text in Wörter auf
        words = cleaned_text.split()
        
        # Filtere kurze und unwichtige Wörter
        meaningful_words = []
        for word in words:
            word_lower = word.lower()
            
            # Überspringen von kurzen oder unwichtigen Wörtern
            if len(word) < 3 or word_lower in filter_words:
                continue
                
            # Prüfen, ob das Wort hauptsächlich aus Buchstaben besteht (nicht nur Zahlen/Symbole)
            letter_count = sum(c.isalpha() for c in word)
            if letter_count / (len(word) + 0.001) > 0.5:
                meaningful_words.append(word)
        
        # Wenn keine bedeutungsvollen Wörter gefunden wurden
        if not meaningful_words:
            # Suche nach speziellen Produktmustern wie "TYP 405"
            pattern_match = re.search(r'(Typ|TYP|Type)\s*\d+', cleaned_text)
            if pattern_match:
                return pattern_match.group(0).capitalize()
            return "Text"
        
        # Prüfe nach bekannten Produktbegriffen
        common_product_words = ["Mehl", "Weizenmehl", "Zucker", "Salz", "Milch", "Brot", "Wasser", 
                              "Butter", "Käse", "Buch", "ISBN", "Roman", "Kaffee", "Tee", "Wein"]
        
        for product_word in common_product_words:
            if product_word.lower() in cleaned_text.lower():
                return product_word
        
        # Finde das längste Wort oder die ersten paar Wörter
        if len(meaningful_words) >= 1:
            # Wenn es nur ein Wort gibt oder ein Wort deutlich länger ist als die anderen
            longest_word = max(meaningful_words, key=len)
            if len(longest_word) >= 5:
                # Wenn es ein langes, dominantes Wort gibt, verwende es
                return longest_word.capitalize()
            
            # Ansonsten nimm die ersten 1-2 bedeutungsvollen Wörter
            if len(meaningful_words) == 1:
                return meaningful_words[0].capitalize()
            else:
                # Bei mehr als 1 Wort, begrenze auf maximal 2 oder bis zu 20 Zeichen
                title = " ".join(meaningful_words[:2])
                if len(title) > 20:
                    # Kürze zu lange Titel und füge "..." hinzu
                    title = title[:17] + "..."
                return title.capitalize()
        
        # Wenn nichts davon zutrifft, verwende generischen Begriff
        return "Objekt"

    def identify_product_from_text(self, text):
        """
        Identify a product from extracted text
        
        Args:
            text: Extracted text from image
            
        Returns:
            Dictionary with product information if identified, None otherwise
        """
        if not text or text.strip() == "":
            return None
        
        # Definiere bekannte Produktmappings (aus Textfragmenten zu Produktbeschreibungen)
        PRODUCT_TEXT_MAPPING = {
            "Mehl": {"name": "Mehl", "description": "Weizenmehl", "confidence_boost": 1.2},
            "Weizenmehl": {"name": "Weizenmehl", "description": "Weizenmehl", "confidence_boost": 1.5},
            "Zucker": {"name": "Zucker", "description": "Zucker", "confidence_boost": 1.2},
            "Raffinadezucker": {"name": "Raffinadezucker", "description": "Feiner Zucker", "confidence_boost": 1.5},
            "Milch": {"name": "Milch", "description": "Frische Milch", "confidence_boost": 1.2},
            "Vollmilch": {"name": "Vollmilch", "description": "Vollmilch", "confidence_boost": 1.5},
            "Brot": {"name": "Brot", "description": "Brot", "confidence_boost": 1.2},
            "Vollkornbrot": {"name": "Vollkornbrot", "description": "Vollkornbrot", "confidence_boost": 1.5},
            "Wasser": {"name": "Wasser", "description": "Mineralwasser", "confidence_boost": 1.2},
            "Mineralwasser": {"name": "Mineralwasser", "description": "Mineralwasser", "confidence_boost": 1.5},
            "Butter": {"name": "Butter", "description": "Butter", "confidence_boost": 1.5},
            "Käse": {"name": "Käse", "description": "Käse", "confidence_boost": 1.2},
            "Gouda": {"name": "Gouda", "description": "Gouda Käse", "confidence_boost": 1.5},
            "Buch": {"name": "Buch", "description": "Buch", "confidence_boost": 1.2},
            "ISBN": {"name": "Buch", "description": "Buch mit ISBN", "confidence_boost": 1.5},
            "Roman": {"name": "Roman", "description": "Romanwerk", "confidence_boost": 1.3},
            "Kaffee": {"name": "Kaffee", "description": "Kaffee", "confidence_boost": 1.3},
            "Tee": {"name": "Tee", "description": "Tee", "confidence_boost": 1.3},
            "Zeitung": {"name": "Zeitung", "description": "Tageszeitung", "confidence_boost": 1.3},
            "Magazin": {"name": "Magazin", "description": "Zeitschrift", "confidence_boost": 1.3},
            "Wein": {"name": "Wein", "description": "Weinflasche", "confidence_boost": 1.3},
            "Rotwein": {"name": "Rotwein", "description": "Rotwein", "confidence_boost": 1.5},
            "Weißwein": {"name": "Weißwein", "description": "Weißwein", "confidence_boost": 1.5}
        }
        
        text_lower = text.lower()
        best_match = None
        best_score = 0
        
        # Durchsuche den Text nach bekannten Produktbegriffen
        for key, product in PRODUCT_TEXT_MAPPING.items():
            # Prüfe auf genaue Übereinstimmung (ohne Berücksichtigung von Groß-/Kleinschreibung)
            if key.lower() in text_lower:
                # Berechne einen Score basierend auf der Länge des gefundenen Begriffs
                # Längere Übereinstimmungen sind oft spezifischer (z.B. "Weizenmehl" vs. "Mehl")
                score = len(key) * product["confidence_boost"]
                if score > best_score:
                    best_score = score
                    best_match = product
        
        # Wenn kein spezielles Produkt gefunden wurde, aber sinnvoller Text vorhanden ist,
        # identifiziere den Gegenstand mit dem erkannten Text selbst - aber nur bei wirklich
        # sinnvollem Text
        if not best_match and self.is_meaningful_text(text):
            # Extrahiere sinnvollen Text für den Namen
            cleaned_text = self.get_best_text_for_title(text)
            logger.info(f"Meaningful text detected, using as product name: '{cleaned_text}'")
            
            # Gib dem Text-basierten Produkt eine niedrigere Konfidenz als bekannten Produkten
            # So wird die Bilderkennung bevorzugt, wenn sie sehr sicher ist
            best_match = {
                "name": cleaned_text,
                "description": cleaned_text,
                "confidence_boost": 0.85  # Niedriger als bei bekannten Produktbegriffen
            }
        
        return best_match

    def preprocess_image(self, image_file):
        """
        Preprocess the image for the model
        
        Args:
            image_file: File object containing the image
            
        Returns:
            Tuple of (preprocessed image tensor, original PIL image)
        """
        try:
            # Read image file
            image_bytes = image_file.read()
            
            # Open image with PIL
            img = Image.open(io.BytesIO(image_bytes))
            
            # Store original image for text extraction
            original_img = img.copy()
            
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply preprocessing transform
            img_tensor = self.preprocess(img)
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor, original_img
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image_file):
        """
        Make predictions on the image and extract text
        
        Args:
            image_file: File object containing the image
            
        Returns:
            List of top 5 predictions with class names, confidence scores, and extracted text
        """
        try:
            # Preprocess the image
            img_tensor, original_img = self.preprocess_image(image_file)
            
            # Disable gradient calculation for inference - erst die Bilderkennung durchführen
            with torch.no_grad():
                # Forward pass through the model
                output = self.model(img_tensor)
                
                # Apply softmax to convert to probabilities
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # Get top 5 predictions
                top5_prob, top5_indices = torch.topk(probabilities, 5)
                
                # Convert to Python lists
                top5_prob = top5_prob.numpy().tolist()
                top5_indices = top5_indices.numpy().tolist()
            
            # Extrahiere Text nur, wenn die Bilderkennung nicht sehr sicher ist
            top_image_confidence = top5_prob[0]
            
            # Format the results
            results = []
            
            # Wenn die Bilderkennung sehr sicher ist (>75%), verwende nur diese
            if top_image_confidence > 0.75:
                logger.info(f"Image recognition has very high confidence ({top_image_confidence:.2f}), skipping text recognition")
                extracted_text = ""
                text_based_product = None
            else:
                # Ansonsten führe auch Texterkennung durch
                # Extract text from the image
                extracted_text = self.extract_text(original_img)
                logger.info(f"Extracted text: {extracted_text}")
                
                # Überprüfen, ob der Text wirklich bedeutungsvoll ist - SEHR STRENGE Regeln
                text_based_product = None
                if self.is_meaningful_text(extracted_text):
                    # Nur wenn der Text wirklich sinnvoll ist, identifiziere ein Produkt
                    text_based_product = self.identify_product_from_text(extracted_text)
                    
                    # Zusätzliche Überprüfung: Ist das führende Bild-Erkennungsergebnis zumindest mäßig sicher?
                    # Wenn ja, müssen wir die Texterkennung sehr spezifisch haben
                    if top_image_confidence > 0.4:
                        # Bei mittlerer Bildsicherheit muss der Text sehr spezifisch sein (bekanntes Produkt)
                        if not text_based_product or ('confidence_boost' in text_based_product 
                                and text_based_product['confidence_boost'] < 1.2):
                            logger.info(f"Image recognition confidence is reasonable ({top_image_confidence:.2f}), text not specific enough")
                            text_based_product = None
            
            # Wenn ein Produkt aus dem Text erkannt wurde, füge es an erster Stelle hinzu
            if text_based_product:
                logger.info(f"Text-based product identified: {text_based_product['description']}")
                
                # Erstelle ein Ergebnisobjekt für das textbasierte Produkt
                text_result = {
                    'class_id': -1,  # Spezielle ID für textbasierte Erkennung
                    'class_name': text_based_product['name'],
                    'class_description': text_based_product['description'],
                    'confidence': 0.95 * text_based_product.get('confidence_boost', 0.9),  # Anpassung der Konfidenz je nach Qualität
                    'extracted_text': extracted_text,
                    'identified_by_text': True  # Markiere, dass dies durch Text erkannt wurde
                }
                
                results.append(text_result)
            
            # Füge die Bilderkennungsergebnisse hinzu
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_prob)):
                class_name = self.categories[idx]
                # Create URL-friendly version of class name for CSS classes
                class_name_slug = class_name.lower().replace(' ', '_').replace("'", '').replace(',', '')
                
                result = {
                    'class_id': int(idx),
                    'class_name': class_name_slug,
                    'class_description': class_name,
                    'confidence': float(prob)
                }
                
                # Add extracted text to the first (top) model result only if no text-based product was found
                if i == 0 and extracted_text and not text_based_product:
                    result['extracted_text'] = extracted_text
                
                results.append(result)
            
            # Begrenze auf maximal 5 Ergebnisse, selbst wenn ein textbasiertes Produkt erkannt wurde
            return results[:5]
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise