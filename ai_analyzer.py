"""
Módulo de análisis de componentes electrónicos mediante IA
"""
import os
import json
import time
import logging
import importlib
from datetime import datetime

# Importación condicional de TensorFlow
def import_tensorflow():
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow importado correctamente, versión: {tf.__version__}")
        return tf
    except ImportError:
        logger.warning("No se pudo importar TensorFlow")
        return None
    except Exception as e:
        logger.error(f"Error al importar TensorFlow: {e}")
        return None

# Importación condicional de OpenCV y NumPy
def import_computer_vision():
    try:
        import cv2
        import numpy as np
        logger.info(f"OpenCV importado correctamente, versión: {cv2.__version__}")
        return cv2, np
    except ImportError:
        logger.warning("No se pudo importar OpenCV o NumPy")
        return None, None
    except Exception as e:
        logger.error(f"Error al importar OpenCV o NumPy: {e}")
        return None, None

# Configurar el sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_analyzer')

class AIAnalyzer:
    def __init__(self):
        """
        Inicializa el analizador de IA para componentes electrónicos.
        """
        self.models = {}
        self.config = self._load_config()
        self.component_classes = [
            'resistor', 'capacitor', 'inductor', 'diode', 
            'transistor', 'led', 'ic', 'connector', 'switch',
            'microcontroller', 'sensor', 'relay', 'crystal', 'transformer'
        ]
        
        # Inicializar sistema de IA
        self.ai_system_ready = False
        self.using_advanced_ai = False
        
        # Definir parámetros de componentes para reconocimiento
        self.component_params = {
            'resistor': {
                'aspect_ratio_range': (2.5, 6.0),
                'color_ranges': [
                    ((0, 0, 100), (80, 80, 255)),  # Marrón/naranja/rojo
                    ((20, 100, 100), (35, 255, 255)),  # Amarillo
                    ((90, 80, 80), (120, 255, 255)),  # Azul
                ],
                'shape': 'rectangle'
            },
            'capacitor': {
                'aspect_ratio_range': (0.8, 1.5),
                'color_ranges': [
                    ((0, 0, 100), (180, 30, 255)),  # Blanco/gris
                    ((0, 0, 0), (180, 255, 60)),    # Negro
                ],
                'shape': 'circle'
            },
            'diode': {
                'aspect_ratio_range': (1.5, 4.0),
                'color_ranges': [
                    ((0, 0, 0), (180, 255, 100)),   # Negro/Gris oscuro
                ],
                'shape': 'rectangle'
            },
            'transistor': {
                'aspect_ratio_range': (0.7, 1.5),
                'color_ranges': [
                    ((0, 0, 0), (180, 255, 80)),    # Negro
                ],
                'shape': 'polygon'
            },
            'ic': {
                'aspect_ratio_range': (0.8, 2.0),
                'color_ranges': [
                    ((0, 0, 0), (180, 100, 100)),   # Negro/Gris oscuro
                ],
                'shape': 'rectangle'
            },
            'led': {
                'aspect_ratio_range': (0.8, 1.2),
                'color_ranges': [
                    ((0, 100, 100), (10, 255, 255)),   # Rojo
                    ((35, 100, 100), (85, 255, 255)),  # Verde
                    ((100, 100, 100), (130, 255, 255)),# Azul
                    ((20, 100, 100), (30, 255, 255)),  # Amarillo
                ],
                'shape': 'circle'
            }
        }
        
        self.load_model()
        logger.info("Modelo cargado correctamente")

    def _load_config(self):
        """
        Carga la configuración para el análisis de componentes.
        """
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'ai_config.json')
        default_config = {
            "detection_threshold": 0.5,
            "confidence_boost": 1.2,
            "min_component_size": 20,
            "max_component_size": 500,
            "use_edge_detection": True,
            "use_color_filtering": True,
            "use_contour_analysis": True
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Asegurar que existe el directorio config
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                # Guardar config por defecto
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {e}")
            return default_config

    def load_model(self):
        """
        Carga los modelos para análisis de componentes.
        
        Esta implementación utiliza dos enfoques:
        1. Detección basada en visión por computadora usando técnicas de procesamiento de imágenes
        2. Detección avanzada usando modelos de IA (TensorFlow/ONNX) cuando estén disponibles
        """
        # Cargar clasificadores de OpenCV para formas básicas (sistema base)
        try:
            # Crear modelos de detección basados en OpenCV
            self.models['edge_detector'] = cv2.Canny
            self.models['contour_finder'] = cv2.findContours
            self.models['shape_analyzer'] = self._analyze_shape
            self.ai_system_ready = True
            logger.info("Modelos de visión por computadora inicializados correctamente")
            
            # Intentar cargar modelo avanzado de IA
            try:
                self._load_advanced_ai_model()
            except:
                logger.info("Usando solo el sistema de visión por computadora")
                
        except Exception as e:
            logger.error(f"Error al cargar los modelos de detección: {e}")
            # Fallback a modelos más simples si los principales fallan
            self.models['basic_detector'] = self._basic_shape_detector
            self.ai_system_ready = False
    
    def _load_advanced_ai_model(self):
        """
        Carga el modelo avanzado de IA para detección de componentes electrónicos.
        Si el modelo no existe, intenta descargarlo o crearlo.
        
        Nota: Para evitar problemas de compatibilidad con NumPy 2.x, se desactiva el uso
        de TensorFlow y ONNX temporalmente.
        """
        try:
            # Deshabilitar modelos avanzados para evitar problemas de compatibilidad 
            # entre NumPy 2.x y TensorFlow/ONNX
            logger.warning("Usando solo el sistema de visión por computadora y xAI para compatibilidad")
            self.using_advanced_ai = False
            
            # Configuramos para usar el modelo xAI cuando esté disponible
            self.models['advanced_detector'] = None
            return False
            
        except Exception as e:
            logger.error(f"Error al cargar modelo de IA avanzado: {e}")
            self.using_advanced_ai = False
            return False
            
    def _create_empty_model(self, model_path):
        """
        Crea un modelo de IA simple que podrá ser entrenado con datos recolectados.
        """
        try:
            # Intentar importar TensorFlow
            tf = import_tensorflow()
            if tf is None:
                logger.error("No se pudo importar TensorFlow para crear el modelo")
                return None
            
            # Crear un modelo simple de CNN para clasificación de componentes
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.component_classes), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Guardar el modelo inicial
            model.save(model_path)
            logger.info(f"Modelo base creado y guardado en {model_path}")
            
            # Almacenar el modelo
            self.models['advanced_detector'] = model
            
        except Exception as e:
            logger.error(f"Error al crear modelo base: {e}")
            return None

    def _analyze_shape(self, contour):
        """
        Analiza la forma de un contorno para determinar el tipo de componente.
        
        Args:
            contour: Contorno a analizar
            
        Returns:
            dict: Información sobre la forma incluyendo tipo, aproximación y propiedades
        """
        # Calcular el perímetro y área
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        # Evitar división por cero
        if perimeter == 0:
            return {'type': 'unknown', 'confidence': 0.1}
        
        # Aproximar contorno a una forma más simple
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calcular rectángulo y elipse de ajuste
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calcular compacidad (relación entre área y perímetro)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calcular factor de forma
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # Determinar la forma basado en estos factores
        shape_info = {
            'vertices': len(approx),
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'extent': extent,
            'area': area,
            'perimeter': perimeter
        }
        
        # Clasificar forma
        if 0.8 < compactness <= 1.0 and len(approx) > 6:
            shape_type = 'circle'
        elif len(approx) == 4 and 0.8 <= extent <= 1.0:
            shape_type = 'square' if 0.8 <= aspect_ratio <= 1.2 else 'rectangle'
        elif len(approx) == 3:
            shape_type = 'triangle'
        elif 4 <= len(approx) <= 6:
            shape_type = 'polygon'
        else:
            shape_type = 'complex'
        
        shape_info['type'] = shape_type
        return shape_info

    def _basic_shape_detector(self, image):
        """
        Detector básico de formas como fallback.
        
        Args:
            image: Imagen a analizar
            
        Returns:
            list: Lista de contornos detectados
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Aplicar umbral
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def process_image(self, image_path):
        """
        Procesa una imagen para análisis.
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            numpy.ndarray: Imagen procesada lista para análisis
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            # Redimensionar para procesamiento uniforme (manteniendo la relación de aspecto)
            height, width = image.shape[:2]
            max_dim = 1024
            
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
            return image
        
        except Exception as e:
            logger.error(f"Error al procesar la imagen: {e}")
            return None

    def detect_components(self, image):
        """
        Detecta componentes electrónicos en la imagen usando una combinación de visión por
        computadora y modelos de IA avanzados cuando están disponibles.
        
        Args:
            image: Imagen procesada
            
        Returns:
            list: Lista de componentes detectados con sus propiedades
        """
        detected_components = []
        component_id = 1
        
        # Convertir a diferentes espacios de color para análisis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Crear imagen de depuración
        debug_image = image.copy()
        
        # Intentar usar el modelo de IA avanzado si está disponible
        ai_detections = []
        if self.using_advanced_ai and 'advanced_detector' in self.models:
            try:
                logger.info("Utilizando modelo de IA avanzado para detección de componentes")
                ai_detections = self._detect_with_ai_model(image)
                
                # Si la IA detectó componentes, usarlos prioritariamente
                if ai_detections:
                    logger.info(f"Modelo de IA detectó {len(ai_detections)} componentes")
                    for ai_comp in ai_detections:
                        x, y, w, h = ai_comp['bbox']
                        confidence = ai_comp['confidence']
                        component_type = ai_comp['type']
                        
                        # Dibujar en la imagen de depuración
                        color = (0, 0, 255)  # Rojo para detecciones de IA
                        cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(debug_image, f"AI: {component_type} ({confidence:.2f})", 
                                   (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                logger.error(f"Error en detección con IA: {e}")
                logger.info("Usando detección basada en visión por computadora como fallback")
        
        # Si el modelo de IA no detectó suficientes componentes, usar visión por computadora
        if len(ai_detections) < 2:
            logger.info("Usando detección basada en visión por computadora")
            
            # 1. Detección de bordes para encontrar estructuras
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. Dilatación para conectar bordes cercanos
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # 3. Encontrar contornos en la imagen
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos y clasificar componentes
            for contour in contours:
                # Filtrar por tamaño para eliminar ruido
                area = cv2.contourArea(contour)
                if area < self.config['min_component_size'] or area > self.config['max_component_size']:
                    continue
                    
                # Obtener rectángulo delimitador
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Extraer la región para análisis detallado
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:  # Asegurarse de que el ROI sea válido
                    continue
                    
                # Analizar forma del contorno
                shape_info = self._analyze_shape(contour)
                
                # Analizar colores predominantes en el ROI
                color_info = self._analyze_colors(roi, hsv[y:y+h, x:x+w])
                
                # Clasificar componente según características
                component_type, confidence = self._classify_component(
                    shape_info, 
                    color_info, 
                    aspect_ratio
                )
                
                # Dibujar para depuración
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.drawContours(debug_image, [contour], 0, color, 2)
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_image, f"{component_type} ({confidence:.2f})", 
                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Si la confianza es suficiente, agregamos el componente a la lista
                if confidence > self.config['detection_threshold']:
                    # Crear especificaciones según tipo de componente
                    specifications = self._generate_specifications(component_type, roi, shape_info)
                    
                    ai_detections.append({
                        'id': component_id,
                        'type': component_type,
                        'confidence': confidence,
                        'bbox': [int(x), int(y), int(x+w), int(y+h)],
                        'dimensions': [int(w), int(h)],
                        'area': float(area),
                        'shape': shape_info['type'],
                        'specifications': specifications,
                        'detection_method': 'computer_vision'
                    })
                    component_id += 1
        
        # Combinar los resultados en una única lista
        for comp in ai_detections:
            # Añadir método de detección si no existe
            if 'detection_method' not in comp:
                comp['detection_method'] = 'ai_model'
            detected_components.append(comp)
                
        # Guardar imagen de depuración
        os.makedirs('static', exist_ok=True)
        debug_path = 'static/debug_detection.jpg'
        cv2.imwrite(debug_path, debug_image)
        
        return detected_components
        
    def _detect_with_ai_model(self, image):
        """
        Detecta componentes usando el modelo de IA avanzado.
        
        Args:
            image: Imagen procesada
            
        Returns:
            list: Lista de componentes detectados por el modelo de IA
        """
        components = []
        component_id = 1
        
        try:
            # Verificar si estamos usando TensorFlow o ONNX
            if 'tensorflow' in str(type(self.models['advanced_detector'])):
                # Importar TensorFlow
                tf = import_tensorflow()
                if tf is None:
                    logger.error("No se pudo importar TensorFlow para detección con IA")
                    return []
                
                # Preprocesar la imagen para el modelo
                img_tensor = self._preprocess_for_tf(image)
                if img_tensor is None:
                    logger.error("Error al preprocesar imagen para TensorFlow")
                    return []
                
                # Realizar la predicción
                predictions = self.models['advanced_detector'].predict(img_tensor)
                
                # Procesar las predicciones
                # (Esto dependerá de la estructura exacta del modelo y sus salidas)
                # Este es un ejemplo genérico
                for i, confidence in enumerate(predictions[0]):
                    if confidence > 0.6:  # Si la confianza es mayor al 60%
                        component_type = self.component_classes[i]
                        
                        # Crear un componente (con coordenadas aproximadas basadas en la imagen completa)
                        h, w = image.shape[:2]
                        components.append({
                            'id': component_id,
                            'type': component_type,
                            'confidence': float(confidence),
                            'bbox': [w//4, h//4, w//2, h//2],  # Coordenadas aproximadas
                            'dimensions': [w//2, h//2],
                            'detection_method': 'ai_model',
                            'specifications': self._generate_specifications(component_type, image, {})
                        })
                        component_id += 1
                        
            elif 'onnxruntime' in str(type(self.models['advanced_detector'])):
                # Importar ONNX Runtime y NumPy
                try:
                    import onnxruntime as ort
                    import numpy as np
                    logger.info("ONNX Runtime importado correctamente")
                except ImportError:
                    logger.error("No se pudo importar ONNX Runtime o NumPy")
                    return []
                except Exception as e:
                    logger.error(f"Error al importar ONNX Runtime: {e}")
                    return []
                
                # Preprocesar imagen para ONNX
                onnx_input = self._preprocess_for_onnx(image)
                if onnx_input is None:
                    logger.error("Error al preprocesar imagen para ONNX")
                    return []
                
                # Realizar la predicción
                input_name = self.models['advanced_detector'].get_inputs()[0].name
                output_name = self.models['advanced_detector'].get_outputs()[0].name
                results = self.models['advanced_detector'].run([output_name], {input_name: onnx_input})
                
                # Procesar los resultados
                predictions = results[0][0]
                for i, confidence in enumerate(predictions):
                    if confidence > 0.6:
                        component_type = self.component_classes[i]
                        
                        # Crear un componente
                        h, w = image.shape[:2]
                        components.append({
                            'id': component_id,
                            'type': component_type,
                            'confidence': float(confidence),
                            'bbox': [w//4, h//4, w//2, h//2],  # Coordenadas aproximadas
                            'dimensions': [w//2, h//2],
                            'detection_method': 'ai_model',
                            'specifications': self._generate_specifications(component_type, image, {})
                        })
                        component_id += 1
                        
        except Exception as e:
            logger.error(f"Error en detección con modelo de IA: {e}")
            
        return components
        
    def _preprocess_for_tf(self, image):
        """
        Preprocesa una imagen para el modelo de TensorFlow.
        
        Args:
            image: Imagen a preprocesar
            
        Returns:
            tensor: Tensor listo para TensorFlow
        """
        # Importar TensorFlow
        tf = import_tensorflow()
        if tf is None:
            logger.error("No se pudo importar TensorFlow para preprocesar la imagen")
            return None
            
        # Importar OpenCV
        cv2, np = import_computer_vision()
        if cv2 is None:
            logger.error("No se pudo importar OpenCV para preprocesar la imagen")
            return None
            
        # Redimensionar a 224x224 (o la entrada que espere el modelo)
        resized = cv2.resize(image, (224, 224))
        
        # Normalizar a valores entre 0 y 1
        normalized = resized / 255.0
        
        # Convertir a tensor y añadir dimensión de batch
        tensor = tf.convert_to_tensor(normalized, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        
        return tensor
        
    def _preprocess_for_onnx(self, image):
        """
        Preprocesa una imagen para el modelo ONNX.
        
        Args:
            image: Imagen a preprocesar
            
        Returns:
            numpy.ndarray: Array listo para ONNX
        """
        # Importar OpenCV y NumPy
        cv2, np = import_computer_vision()
        if cv2 is None or np is None:
            logger.error("No se pudo importar OpenCV o NumPy para preprocesar la imagen")
            return None
            
        # Redimensionar a 224x224 (o la entrada que espere el modelo)
        resized = cv2.resize(image, (224, 224))
        
        # Normalizar a valores entre 0 y 1
        normalized = resized.astype(np.float32) / 255.0
        
        # Convertir a NCHW (batch, channels, height, width) si es necesario
        # Algunos modelos ONNX esperan este formato
        transposed = np.transpose(normalized, (2, 0, 1))  # De HWC a CHW
        
        # Añadir dimensión de batch
        batch = np.expand_dims(transposed, 0)
        
        return batch

    def _analyze_colors(self, roi, hsv_roi):
        """
        Analiza los colores predominantes en una región de interés.
        
        Args:
            roi: Región de interés en formato BGR
            hsv_roi: La misma región en formato HSV
            
        Returns:
            dict: Información sobre colores predominantes
        """
        # Importar OpenCV y NumPy
        cv2, np = import_computer_vision()
        if cv2 is None or np is None:
            logger.error("No se pudo importar OpenCV o NumPy para analizar colores")
            return {'dominant_colors': [], 'color_percentages': {}}
            
        color_info = {
            'dominant_colors': [],
            'color_percentages': {}
        }
        
        # Si la región es demasiado pequeña, devolver info vacía
        if roi.size == 0:
            return color_info
            
        # Convertir colores a histograma HSV y encontrar predominantes
        h_hist = cv2.calcHist([hsv_roi], [0], None, [30], [0, 180])
        s_hist = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv_roi], [2], None, [32], [0, 256])
        
        # Normalizar histogramas
        total_pixels = roi.shape[0] * roi.shape[1]
        h_hist = h_hist / total_pixels
        s_hist = s_hist / total_pixels
        v_hist = v_hist / total_pixels
        
        # Detectar rangos de color importantes
        hue_ranges = [
            (0, 10, "red"), (160, 180, "red"),
            (11, 25, "orange"), (26, 35, "yellow"),
            (36, 70, "green"), (71, 85, "cyan"),
            (86, 130, "blue"), (131, 155, "purple")
        ]
        
        for start, end, color_name in hue_ranges:
            # Calcular la suma de frecuencias en este rango
            if start < end:
                range_sum = np.sum(h_hist[start:end+1])
            else:  # para el rojo que cruza el 0
                range_sum = np.sum(h_hist[start:]) + np.sum(h_hist[:end+1])
                
            # Si más del 15% de píxeles están en este rango y la saturación es alta
            if range_sum > 0.15 and np.sum(s_hist[16:]) > 0.3:  # Saturación > 128
                color_info['dominant_colors'].append(color_name)
                color_info['color_percentages'][color_name] = float(range_sum)
                
        # Detectar si es mayormente blanco, negro o gris
        if len(color_info['dominant_colors']) == 0:
            v_low = np.sum(v_hist[:8])  # Valor < 64 (oscuro/negro)
            v_high = np.sum(v_hist[24:])  # Valor > 192 (claro/blanco)
            s_low = np.sum(s_hist[:8])  # Saturación < 64 (gris)
            
            if v_low > 0.5:  # Predominantemente oscuro
                color_info['dominant_colors'].append("black")
                color_info['color_percentages']["black"] = float(v_low)
            elif v_high > 0.5 and s_low > 0.5:  # Claro y baja saturación
                color_info['dominant_colors'].append("white")
                color_info['color_percentages']["white"] = float(v_high)
            elif s_low > 0.5:  # Baja saturación (gris)
                color_info['dominant_colors'].append("gray")
                color_info['color_percentages']["gray"] = float(s_low)
                
        return color_info

    def _classify_component(self, shape_info, color_info, aspect_ratio):
        """
        Clasifica un componente según sus características de forma y color.
        
        Args:
            shape_info: Información sobre la forma
            color_info: Información sobre colores predominantes
            aspect_ratio: Relación de aspecto del componente
            
        Returns:
            tuple: (tipo_componente, confianza)
        """
        # Puntuaciones para cada tipo de componente
        scores = {component: 0.0 for component in self.component_classes}
        
        # Clasificación basada en la forma
        shape_type = shape_info['type']
        vertices = shape_info['vertices']
        compactness = shape_info['compactness']
        extent = shape_info['extent']
        
        # Colores predominantes
        dominant_colors = color_info['dominant_colors']
        
        # Evaluación de resistencias
        if shape_type == 'rectangle' and 2.0 < aspect_ratio < 6.0:
            scores['resistor'] += 0.5
            if any(c in dominant_colors for c in ['brown', 'red', 'orange', 'yellow', 'green', 'blue']):
                scores['resistor'] += 0.3
                
        # Evaluación de capacitores
        if shape_type in ['circle', 'rectangle'] and 0.6 < aspect_ratio < 2.0:
            scores['capacitor'] += 0.4
            if any(c in dominant_colors for c in ['blue', 'black', 'gray', 'white']):
                scores['capacitor'] += 0.3
                
        # Evaluación de diodos
        if shape_type == 'rectangle' and 1.5 < aspect_ratio < 4.0:
            scores['diode'] += 0.4
            if 'black' in dominant_colors:
                scores['diode'] += 0.3
                
        # Evaluación de transistores
        if vertices >= 6 and 0.8 < aspect_ratio < 1.5:
            scores['transistor'] += 0.4
            if 'black' in dominant_colors:
                scores['transistor'] += 0.3
                
        # Evaluación de circuitos integrados
        if shape_type == 'rectangle' and vertices <= 8 and 0.8 < aspect_ratio < 2.5:
            scores['ic'] += 0.5
            if 'black' in dominant_colors:
                scores['ic'] += 0.3
                
        # Evaluación de LEDs
        if shape_type == 'circle' and compactness > 0.8:
            scores['led'] += 0.5
            if any(c in dominant_colors for c in ['red', 'green', 'blue', 'yellow']):
                scores['led'] += 0.3
                
        # Evaluación de inductores
        if shape_type == 'circle' and 0.8 < aspect_ratio < 1.2:
            scores['inductor'] += 0.3
            if any(c in dominant_colors for c in ['gray', 'blue', 'black']):
                scores['inductor'] += 0.2
                
        # Evaluación de conectores
        if shape_type in ['rectangle', 'polygon'] and vertices >= 4:
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:  # Muy alargado o muy achatado
                scores['connector'] += 0.4
                
        # Evaluación de interruptores
        if shape_type in ['rectangle', 'square'] and 1.5 < aspect_ratio < 4.0:
            scores['switch'] += 0.3
            if any(c in dominant_colors for c in ['black', 'red']):
                scores['switch'] += 0.2
                
        # Encontrar el componente con mayor puntuación
        best_component = max(scores, key=scores.get)
        confidence = scores[best_component]
        
        # Si no hay suficiente confianza, considerar desconocido
        if confidence < 0.3:
            return 'unknown', confidence
            
        return best_component, min(confidence, 0.99)  # Limitar a 0.99 como máximo

    def _generate_specifications(self, component_type, roi, shape_info):
        """
        Genera especificaciones para un componente basado en su tipo.
        
        Args:
            component_type: Tipo de componente
            roi: Región de interés de la imagen
            shape_info: Información de forma
            
        Returns:
            dict: Especificaciones específicas del componente
        """
        specs = {}
        
        # Detectar bandas en resistencias
        if component_type == 'resistor':
            # Implementación básica para detectar bandas de colores
            # En una implementación completa, aquí procesaríamos las bandas de colores
            # para calcular el valor de la resistencia
            specs['aproximate_resistance'] = "10 kΩ ± 5%"
            specs['power_rating'] = "0.25 W"
            
        # Especificaciones de capacitores
        elif component_type == 'capacitor':
            # Determinar tipo por forma y color
            if 'compactness' in shape_info and shape_info['compactness'] > 0.8:  # Circular, probablemente electrolítico
                specs['type'] = "Electrolytic"
                specs['approximate_capacitance'] = "100 μF"
                specs['voltage_rating'] = "16V"
            else:
                specs['type'] = "Ceramic"
                specs['approximate_capacitance'] = "0.1 μF"
                
        # Especificaciones de diodos
        elif component_type == 'diode':
            # Detectar si es LED o diodo normal
            specs['type'] = "General Purpose"
            specs['forward_voltage'] = "0.7V"
            
        # Especificaciones de transistores
        elif component_type == 'transistor':
            specs['type'] = "NPN/BJT"
            specs['package'] = "TO-92"
            
        # Especificaciones de circuitos integrados
        elif component_type == 'ic':
            try:
                # Contar pines (aproximación visual)
                w, h = roi.shape[1], roi.shape[0]
                pin_count = max(int(2 * (w + h) / 20), 8)
                specs['approximate_pin_count'] = pin_count
                specs['package'] = "DIP" if 'aspect_ratio' in shape_info and shape_info['aspect_ratio'] > 1.5 else "SMD"
            except (AttributeError, IndexError):
                specs['approximate_pin_count'] = 8
                specs['package'] = "Unknown"
            
        # Especificaciones de LEDs
        elif component_type == 'led':
            try:
                # Importar OpenCV y NumPy
                cv2, np = import_computer_vision()
                if cv2 is not None and np is not None:
                    # Detectar color del LED
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    h_mean = np.mean(hsv_roi[:, :, 0])
                    
                    if 0 <= h_mean <= 10 or 170 <= h_mean <= 180:
                        specs['color'] = "Red"
                    elif 35 <= h_mean <= 85:
                        specs['color'] = "Green"
                    elif 85 <= h_mean <= 130:
                        specs['color'] = "Blue"
                    elif 10 <= h_mean <= 35:
                        specs['color'] = "Yellow/Orange"
                    else:
                        specs['color'] = "Unknown"
                    
                    specs['diameter'] = f"{min(roi.shape[0], roi.shape[1])} px"
                else:
                    specs['color'] = "Unknown"
                    specs['diameter'] = "Unknown"
            except Exception as e:
                logger.error(f"Error al analizar LED: {e}")
                specs['color'] = "Unknown"
                specs['diameter'] = "Unknown"
            
        return specs

    def detect_connections(self, image, components):
        """
        Detecta conexiones entre componentes.
        
        Args:
            image: Imagen procesada
            components: Lista de componentes detectados
            
        Returns:
            list: Lista de conexiones entre componentes
        """
        if not components or len(components) < 2:
            return []
            
        # Importar OpenCV y NumPy
        cv2, np = import_computer_vision()
        if cv2 is None or np is None:
            logger.error("No se pudo importar OpenCV o NumPy para detectar conexiones")
            return []
            
        connections = []
        
        try:
            # Convertir a escala de grises y aplicar detección de bordes
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral adaptativo para detectar líneas de conexión
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Operaciones morfológicas para limpiar la imagen
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Detectar líneas con transformada de Hough
            lines = cv2.HoughLinesP(
                morph, 1, np.pi/180, threshold=50, 
                minLineLength=30, maxLineGap=20
            )
            
            # Crear lista de componentes con bounding boxes para detección de conexiones
            component_boxes = [(c['id'], tuple(c['bbox'])) for c in components]
            
            # Verificar líneas que conectan componentes
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Verificar qué componentes se conectan con esta línea
                    connected_components = []
                    
                    for comp_id, bbox in component_boxes:
                        bx1, by1, bx2, by2 = bbox
                        
                        # Verificar si algún extremo de la línea está dentro del bounding box
                        if (bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or (bx1 <= x2 <= bx2 and by1 <= y2 <= by2):
                            connected_components.append(comp_id)
                            
                    # Si la línea conecta dos componentes, registrar la conexión
                    if len(connected_components) >= 2:
                        for i in range(len(connected_components) - 1):
                            for j in range(i + 1, len(connected_components)):
                                # Determinar el tipo de conexión (simplificado)
                                connection_type = "signal"  # Por defecto
                                
                                # Añadir la conexión
                                connections.append({
                                    'from_component': connected_components[i],
                                    'to_component': connected_components[j],
                                    'type': connection_type
                                })
            
            # Eliminar conexiones duplicadas
            unique_connections = []
            connection_pairs = set()
            
            for conn in connections:
                pair = tuple(sorted([conn['from_component'], conn['to_component']]))
                if pair not in connection_pairs:
                    connection_pairs.add(pair)
                    unique_connections.append(conn)
        
        except Exception as e:
            logger.error(f"Error al detectar conexiones: {e}")
            return []
                    
        return unique_connections

    def analyze_circuit(self, components, connections):
        """
        Analiza el circuito completo para determinar su tipo y características.
        
        Args:
            components: Lista de componentes detectados
            connections: Lista de conexiones entre componentes
            
        Returns:
            dict: Análisis del circuito
        """
        if not components:
            return None
            
        # Contar componentes por tipo
        component_counts = {}
        for comp in components:
            comp_type = comp['type']
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
            
        # Determinar el tipo de circuito
        circuit_type = "Unknown"
        complexity = "Simple"
        estimated_power = "Low"
        
        # Analizar la composición del circuito
        has_ic = component_counts.get('ic', 0) > 0
        resistor_count = component_counts.get('resistor', 0)
        capacitor_count = component_counts.get('capacitor', 0)
        transistor_count = component_counts.get('transistor', 0)
        diode_count = component_counts.get('diode', 0)
        led_count = component_counts.get('led', 0)
        
        # Determinar tipo de circuito basado en la composición
        if has_ic:
            circuit_type = "Digital/Integrated"
            complexity = "Complex" if len(components) > 10 else "Medium"
        elif transistor_count > 0 and resistor_count > 0:
            circuit_type = "Analog/Amplifier"
            complexity = "Medium"
        elif diode_count > 0 and capacitor_count > 0:
            circuit_type = "Power/Rectifier"
        elif led_count > 0 and resistor_count > 0:
            circuit_type = "Indicator/LED"
        elif resistor_count > 0 and capacitor_count > 0:
            circuit_type = "Filter/Timing"
            
        # Determinar complejidad basada en número de componentes y conexiones
        total_components = len(components)
        if total_components > 15:
            complexity = "Complex"
        elif total_components > 5:
            complexity = "Medium"
            
        # Estimación de potencia (muy básica)
        if any(c['type'] == 'ic' for c in components):
            estimated_power = "Medium"
        elif any(c['type'] in ['transistor', 'diode'] for c in components):
            estimated_power = "Low-Medium"
            
        return {
            'type': circuit_type,
            'complexity': complexity,
            'estimated_power': estimated_power,
            'component_distribution': component_counts
        }

    def analyze_file(self, file_path):
        """
        Analiza un archivo electrónico.
        
        Args:
            file_path (str): Ruta al archivo a analizar
            
        Returns:
            dict: Resultados del análisis
        """
        # Si es un PDF, extraer y analizar la primera página como imagen
        if file_path.lower().endswith('.pdf'):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                page = doc.load_page(0)
                pix = page.get_pixmap()
                temp_img_path = os.path.join(os.path.dirname(file_path), f"temp_{os.path.basename(file_path)}.png")
                pix.save(temp_img_path)
                result = self.analyze_image(temp_img_path)
                # Limpiar el archivo temporal
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                return result
            except ImportError:
                logger.error("PyMuPDF (fitz) no está instalado. No se puede analizar PDF.")
                return {"error": "No se puede analizar PDF: PyMuPDF no está instalado"}
            except Exception as e:
                logger.error(f"Error al analizar PDF: {e}")
                return {"error": f"Error al analizar PDF: {str(e)}"}
        
        # Si es una imagen, analizarla directamente
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return self.analyze_image(file_path)
        
        else:
            return {"error": "Formato de archivo no soportado"}

    def analyze_image(self, image_path):
        """
        Analiza una imagen para detectar componentes electrónicos.
        Utiliza una combinación de:
        1. Visión por computadora con OpenCV
        2. Modelos avanzados de IA (TensorFlow/ONNX) si están disponibles
        3. API de xAI para análisis con modelos Grok si la API está configurada
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            dict: Resultados del análisis
        """
        logger.info(f"Analizando imagen: {image_path}")
        start_time = time.time()
        
        # Procesar imagen
        image = self.process_image(image_path)
        if image is None:
            return {"error": "No se pudo procesar la imagen"}
        
        try:
            # Intentar usar xAI para analizar la imagen
            # Primero verificamos si el módulo está disponible e inicializado
            xai_results = None
            try:
                # Importar xAI de forma dinámica
                xai_module = importlib.import_module('xai_integration')
                xai_processor = xai_module.XAIProcessor()
                
                if xai_processor.is_available():
                    logger.info("Usando xAI para análisis avanzado de componentes")
                    xai_results = xai_processor.analyze_image(image_path)
                    
                    # Si xAI retornó resultados exitosamente
                    if xai_results.get("success", True) and not xai_results.get("error") and xai_results.get("components"):
                        logger.info(f"Análisis con xAI exitoso: {len(xai_results.get('components', []))} componentes detectados")
            except Exception as e:
                logger.error(f"Error al utilizar xAI: {e}")
                logger.info("Continuando con métodos de análisis locales")
            
            # Detectar componentes con métodos locales
            components = self.detect_components(image)
            logger.info(f"Componentes detectados localmente: {len(components)}")
            
            # Detectar conexiones entre componentes
            connections = self.detect_connections(image, components)
            logger.info(f"Conexiones detectadas: {len(connections)}")
            
            # Analizar el circuito completo
            circuit_analysis = self.analyze_circuit(components, connections)
            
            # Generar resultado base con los métodos locales
            result = {
                "components": components,
                "component_count": len(components),
                "connections": connections,
                "connection_count": len(connections),
                "circuit_analysis": circuit_analysis,
                "image_dimensions": image.shape[:2],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_time": round(time.time() - start_time, 2)
            }
            
            # Integrar resultados de xAI si están disponibles
            if xai_results and xai_results.get("components"):
                # Marcar componentes de las diferentes fuentes
                for comp in result["components"]:
                    comp["detection_source"] = "local_cv"
                    comp["detected_by"] = "Visión por Computadora"
                
                # Preparar componentes de xAI con IDs continuos
                xai_components = xai_results.get("components", [])
                next_id = len(components) + 1
                
                for i, xai_comp in enumerate(xai_components):
                    # Asegurar que tiene un ID único (no duplicado con los locales)
                    xai_comp["id"] = next_id + i
                    xai_comp["detection_source"] = "xai"
                    xai_comp["detected_by"] = "xAI (Grok)"
                    
                    # Si no hay coordenadas de bounding box precisas, crear unas estimadas
                    if "bbox" not in xai_comp and "position" in xai_comp:
                        h, w = image.shape[:2]
                        # Posiciones aproximadas basadas en descripciones textuales
                        positions = {
                            "centro": [w//4, h//4, w//2, h//2],
                            "centro-izquierda": [0, h//4, w//3, h//2],
                            "centro-derecha": [2*w//3, h//4, w//3, h//2],
                            "arriba": [w//4, 0, w//2, h//3],
                            "abajo": [w//4, 2*h//3, w//2, h//3]
                        }
                        
                        for pos_key, bbox in positions.items():
                            if pos_key in xai_comp["position"].lower():
                                xai_comp["bbox"] = bbox
                                break
                        
                        # Si no hay coincidencia, usar posición genérica
                        if "bbox" not in xai_comp:
                            xai_comp["bbox"] = [w//4, h//4, w//2, h//2]
                
                # Integrar datos de circuito de xAI si están disponibles
                if "circuit_type" in xai_results and xai_results["circuit_type"]:
                    result["xai_circuit_analysis"] = {
                        "circuit_type": xai_results.get("circuit_type", ""),
                        "analysis_confidence": xai_results.get("analysis_confidence", 0.0)
                    }
                
                # Combinar componentes de ambas fuentes
                result["components"].extend(xai_components)
                result["component_count"] = len(result["components"])
                result["xai_enhanced"] = True
            
            # Guardar el resultado
            result_dir = 'results'
            os.makedirs(result_dir, exist_ok=True)
            
            result_file = os.path.join(result_dir, f"{os.path.basename(image_path)}_analysis.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Análisis completado y guardado en {result_file}")
            return result
            
        except Exception as e:
            logger.error(f"Error durante el análisis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}


if __name__ == "__main__":
    # Código de prueba para ejecutar el analizador con una imagen de ejemplo
    analyzer = AIAnalyzer()
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        result = analyzer.analyze_image(test_image)
        print(f"Componentes detectados: {result['component_count']}")
    else:
        print(f"No se encontró la imagen de prueba: {test_image}")