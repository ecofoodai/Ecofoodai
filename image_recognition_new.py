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
        Extract text from the image using OCR
        
        Args:
            img: PIL Image object
            
        Returns:
            Extracted text as string
        """
        try:
            # Convert image to grayscale for better OCR performance
            gray_img = img.convert('L')
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(gray_img, config=self.tesseract_config)
            
            # Filter out empty lines and clean up the text
            text_lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = ' '.join(text_lines)
            
            logger.info(f"Extracted text: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}")
            return clean_text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

    def is_meaningful_text(self, text):
        """
        Prüft, ob der erkannte Text sinnvoll ist (nicht nur zufällige Zeichen)
        
        Args:
            text: Extrahierter Text
            
        Returns:
            Boolean, ob der Text sinnvoll erscheint
        """
        if not text or len(text.strip()) < 3:
            return False
            
        # Entferne Sonderzeichen für die Analyse
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        cleaned_text = cleaned_text.strip()
        
        # Mindestlänge nach Bereinigung
        if len(cleaned_text) < 3:
            return False
            
        # Prüfen auf Wörter (mindestens 3 Buchstaben)
        words = cleaned_text.split()
        meaningful_words = [w for w in words if len(w) >= 3]
        
        # Wenn mindestens ein sinnvolles Wort gefunden wurde
        if len(meaningful_words) >= 1:
            # Verhältnis von Buchstaben zu Gesamtzeichen prüfen
            letters = sum(c.isalpha() for c in cleaned_text)
            if letters / (len(cleaned_text) + 0.001) > 0.5:  # Mindestens 50% Buchstaben
                return True
                
        return False

    def get_best_text_for_title(self, text):
        """
        Extrahiert den besten Text für die Verwendung als Produktname
        
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
        
        # Teile den Text in Wörter auf
        words = cleaned_text.split()
        
        # Finde das längste Wort oder die ersten paar Wörter
        if len(words) >= 1:
            # Wenn es nur ein Wort gibt oder ein Wort deutlich länger ist als die anderen
            longest_word = max(words, key=len)
            if len(longest_word) >= 5 and len(longest_word) >= len(cleaned_text) * 0.3:
                # Wenn es ein langes, dominantes Wort gibt, verwende es
                return longest_word.capitalize()
            
            # Ansonsten nimm die ersten 1-3 Wörter, abhängig von der Länge
            if len(words) == 1:
                return words[0].capitalize()
            elif len(words) == 2:
                return " ".join(words).capitalize()
            else:
                # Bei mehr als 2 Wörtern, begrenze auf maximal 3 oder bis zu 20 Zeichen
                title = " ".join(words[:3])
                if len(title) > 20:
                    # Kürze zu lange Titel und füge "..." hinzu
                    title = title[:17] + "..."
                return title.capitalize()
        
        # Fallback: Nimm die ersten 20 Zeichen, wenn keine klare Wortstruktur erkennbar ist
        if len(cleaned_text) > 20:
            return cleaned_text[:17].capitalize() + "..."
        
        return cleaned_text.capitalize()

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
        
        # NEUE FUNKTION: Wenn kein spezielles Produkt gefunden wurde, aber sinnvoller Text vorhanden ist,
        # identifiziere den Gegenstand mit dem erkannten Text selbst
        if not best_match and self.is_meaningful_text(text):
            # Extrahiere sinnvollen Text für den Namen
            cleaned_text = self.get_best_text_for_title(text)
            logger.info(f"Meaningful text detected, using as product name: '{cleaned_text}'")
            best_match = {
                "name": cleaned_text,
                "description": cleaned_text,
                "confidence_boost": 0.9
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
            
            # Extract text from the image
            extracted_text = self.extract_text(original_img)
            
            # Identify product from text
            text_based_product = self.identify_product_from_text(extracted_text)
            
            # Disable gradient calculation for inference
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
            
            # Format the results
            results = []
            
            # Wenn ein Produkt aus dem Text erkannt wurde, füge es an erster Stelle hinzu
            if text_based_product:
                logger.info(f"Text-based product identified: {text_based_product['description']}")
                
                # Erstelle ein Ergebnisobjekt für das textbasierte Produkt
                text_result = {
                    'class_id': -1,  # Spezielle ID für textbasierte Erkennung
                    'class_name': text_based_product['name'],
                    'class_description': text_based_product['description'],
                    'confidence': 0.95,  # Hohe Konfidenz für textbasierte Erkennung
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