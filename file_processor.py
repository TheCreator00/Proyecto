"""
Módulo para procesar archivos
"""
import os
import uuid
import cv2
import numpy as np
import logging
from datetime import datetime

# Configurar el sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('file_processor')

class FileProcessor:
    def __init__(self):
        """
        Inicializa el procesador de archivos.
        """
        # Crear directorios necesarios
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        os.makedirs('static/previews', exist_ok=True)
    
    def process_file(self, file_path):
        """
        Procesa un archivo subido, lo guarda en el sistema y genera una vista previa.
        
        Args:
            file_path (str): Ruta al archivo original
            
        Returns:
            dict: Información del archivo procesado
        """
        try:
            # Obtener información básica del archivo
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Generar un ID único para el archivo
            file_id = str(uuid.uuid4())
            
            # Determinar tipo de archivo
            file_type = self.determine_file_type(file_ext)
            
            # Generar vista previa
            preview_path = self.create_preview(file_path, file_id)
            
            # Crear estructura de datos con la información del archivo
            file_info = {
                'id': file_id,
                'name': filename,
                'path': file_path,
                'size': file_size,
                'type': file_type,
                'extension': file_ext,
                'upload_date': datetime.now().isoformat(),
                'preview_path': preview_path
            }
            
            logger.info(f"Archivo procesado: {filename}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error al procesar archivo {file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def create_preview(self, file_path, unique_id):
        """
        Crea una imagen de vista previa para el archivo.
        
        Args:
            file_path (str): Ruta al archivo
            unique_id (str): Identificador único
            
        Returns:
            str: Ruta a la imagen de vista previa
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            preview_path = f"static/previews/preview_{unique_id}.jpg"
            
            # Generar vista previa según el tipo de archivo
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Procesamiento para imágenes
                image = cv2.imread(file_path)
                if image is None:
                    logger.error(f"No se pudo cargar la imagen: {file_path}")
                    return "static/no_preview.svg"
                
                # Redimensionar para vista previa
                max_size = 400
                height, width = image.shape[:2]
                
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Guardar vista previa
                cv2.imwrite(preview_path, image)
                
            elif file_ext == '.pdf':
                # Crear vista previa para PDF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    
                    if doc.page_count > 0:
                        # Tomar la primera página
                        page = doc.load_page(0)
                        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                        pix.save(preview_path)
                    else:
                        logger.warning(f"PDF sin páginas: {file_path}")
                        return "static/no_preview.svg"
                except ImportError:
                    logger.error("PyMuPDF (fitz) no está instalado. No se puede generar vista previa de PDF.")
                    return "static/no_preview.svg"
                except Exception as e:
                    logger.error(f"Error al generar vista previa del PDF: {e}")
                    return "static/no_preview.svg"
            else:
                # Otros tipos de archivo sin vista previa
                return "static/no_preview.svg"
            
            logger.info(f"Vista previa generada: {preview_path}")
            return preview_path
            
        except Exception as e:
            logger.error(f"Error al generar vista previa: {e}")
            return "static/no_preview.svg"

    def determine_file_type(self, extension):
        """
        Determina el tipo de archivo basado en su extensión.
        
        Args:
            extension (str): Extensión del archivo
            
        Returns:
            str: Tipo de archivo ('document', 'image', 'unknown')
        """
        extension = extension.lower()
        
        if extension in ['.pdf']:
            return 'document'
        elif extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return 'image'
        else:
            return 'unknown'


if __name__ == "__main__":
    # Código de prueba
    processor = FileProcessor()
    test_file = "test.jpg"
    
    if os.path.exists(test_file):
        info = processor.process_file(test_file)
        print(f"Archivo procesado: {info['name']}")
        print(f"Vista previa generada: {info['preview_path']}")
    else:
        print(f"No se encontró el archivo de prueba: {test_file}")