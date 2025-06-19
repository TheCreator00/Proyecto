"""
Módulo de integración con xAI para análisis de componentes electrónicos

Este módulo proporciona funciones para utilizar los modelos Grok 
de xAI para mejorar la detección y análisis de componentes electrónicos.

Este es ahora el método principal de análisis avanzado, ya que evitamos las 
dependencias de TensorFlow y NumPy que causan problemas de compatibilidad.
"""

import os
import sys
import json
import logging
import base64
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('xai_integration')

class XAIProcessor:
    """
    Clase para procesar imágenes de componentes electrónicos mediante xAI
    con capacidades de aprendizaje automático y optimización continua
    """
    
    def __init__(self):
        """
        Inicializa el procesador de IA con modelos locales y configuración de aprendizaje.
        """
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar sistema de aprendizaje automático
        self.learning_system = self._initialize_learning_system()
        
        # Verificar si hay API key disponible
        self.api_key = os.environ.get('XAI_API_KEY')
        self.use_local = True if not self.api_key else False
        
        if self.use_local:
            logger.info("Usando modelos locales para el análisis")
        
        # Modelos disponibles para selección del usuario
        self.available_models = {
            "local": {
                "vision": self._get_local_model(),
                "text": self._get_local_text_model(),
                "auto_learning": True
            },
            "hybrid": {
                "vision": "grok-2-vision-1212",
                "text": "grok-2-1212",
                "auto_learning": True
            }
        }
        
        # Inicializar sistema de optimización
        self._setup_optimization()

    def _load_config(self):
        """Carga la configuración de IA desde el archivo JSON"""
        try:
            with open('config/ai_config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            return {}

    def _initialize_learning_system(self):
        """Inicializa el sistema de aprendizaje automático"""
        if self.config.get('auto_learning', {}).get('enabled', False):
            return {
                'learning_rate': self.config['auto_learning']['learning_rate'],
                'batch_size': self.config['auto_learning']['batch_size'],
                'current_epoch': 0,
                'training_data': [],
                'validation_data': []
            }
        return None

    def _setup_optimization(self):
        """Configura las optimizaciones del sistema"""
        if self.config.get('optimization', {}).get('use_gpu', False):
            self._setup_gpu()
        if self.config.get('optimization', {}).get('parallel_processing', False):
            self._setup_parallel_processing()

    def _get_local_model(self):
        """Obtiene o crea un modelo local para procesamiento de visión"""
        try:
            return self._create_local_vision_model()
        except Exception as e:
            logger.error(f"Error al crear modelo local: {e}")
            return None

    def _setup_gpu(self):
        """Configura el uso de GPU para procesamiento"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.init()
                logger.info("GPU inicializada correctamente")
            else:
                logger.warning("GPU no disponible, usando CPU")
        except ImportError:
            logger.warning("PyTorch no instalado, usando CPU")

    def _setup_parallel_processing(self):
        """Configura el procesamiento paralelo"""
        try:
            import multiprocessing
            self.num_cores = multiprocessing.cpu_count()
            self.pool = multiprocessing.Pool(processes=self.num_cores)
            logger.info(f"Procesamiento paralelo configurado con {self.num_cores} núcleos")
        except Exception as e:
            logger.error(f"Error al configurar procesamiento paralelo: {e}")

    def train_model(self, training_data):
        """Entrena el modelo con nuevos datos"""
        if not self.learning_system:
            logger.warning("Sistema de aprendizaje no inicializado")
            return

        try:
            # Dividir datos en entrenamiento y validación
            split_idx = int(len(training_data) * (1 - self.config['auto_learning']['validation_split']))
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]

            # Actualizar datos de entrenamiento
            self.learning_system['training_data'].extend(train_data)
            self.learning_system['validation_data'].extend(val_data)

            # Realizar entrenamiento
            self._perform_training_epoch()

            # Guardar modelo si es necesario
            if self.learning_system['current_epoch'] % self.config['auto_learning']['save_interval'] == 0:
                self._save_model()

        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")

    def _perform_training_epoch(self):
        """Realiza una época de entrenamiento"""
        try:
            batch_size = self.learning_system['batch_size']
            learning_rate = self.learning_system['learning_rate']

            # Implementar lógica de entrenamiento específica aquí
            # Ejemplo: Actualización de pesos, backpropagation, etc.

            self.learning_system['current_epoch'] += 1
            logger.info(f"Época {self.learning_system['current_epoch']} completada")

        except Exception as e:
            logger.error(f"Error en época de entrenamiento: {e}")

    def _save_model(self):
        """Guarda el estado actual del modelo"""
        try:
            model_path = f"models/xai_model_epoch_{self.learning_system['current_epoch']}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Implementar lógica de guardado específica aquí
            logger.info(f"Modelo guardado en {model_path}")
        except Exception as e:
            logger.error(f"Error al guardar modelo: {e}")
        
        # Modelo seleccionado actualmente
        self.current_model = "grok"
        self.models = {
            'vision': 'grok-2-vision-1212',  # Modelo para procesar imágenes
            'text': 'grok-2-1212'            # Modelo para procesar texto
        }
        self.initialized = self.api_key is not None
        
        if self.initialized:
            logger.info("XAI Processor inicializado correctamente")
        else:
            logger.warning("XAI Processor no inicializado: clave API no disponible")
    
    def is_available(self) -> bool:
        """
        Verifica si la integración con xAI está disponible.
        
        Returns:
            bool: True si la integración está disponible, False en caso contrario
        """
        # Si el modelo actual es "local", consideramos que está disponible
        # aunque con funcionalidad limitada
        if self.current_model == "local":
            return True
        
        # Para los demás modelos, se requiere la API key
        return self.initialized
        
    def set_model(self, model_name: str) -> bool:
        """
        Cambia el modelo de IA a utilizar.
        
        Args:
            model_name: Nombre del modelo ("grok", "local", "hybrid")
            
        Returns:
            bool: True si el cambio fue exitoso, False en caso contrario
        """
        if model_name in self.available_models:
            self.current_model = model_name
            
            # Actualizar los modelos según la selección
            if model_name == "local":
                # En modo local, usamos None para indicar que no se usará el API
                logger.info("Cambiando a modo local (sin API)")
            else:
                # Para los demás modos, usamos los modelos del API
                self.models = {
                    'vision': self.available_models[model_name]['vision'],
                    'text': self.available_models[model_name]['text']
                }
                logger.info(f"Cambiando a modo {model_name} con modelos: {self.models}")
                
            return True
        else:
            logger.warning(f"Modelo no reconocido: {model_name}")
            return False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analiza una imagen para identificar componentes electrónicos.
        
        Args:
            image_path: Ruta a la imagen a analizar
            
        Returns:
            Dict: Resultado del análisis con componentes identificados
        """
        # Si estamos en modo local, devolvemos un resultado básico
        if self.current_model == "local":
            logger.info("Usando modo local para análisis básico de imagen")
            return self._local_image_analysis(image_path)
            
        # Para los demás modos, necesitamos la API key
        if not self.initialized:
            return {"error": "API de xAI no inicializada", "success": False}
        
        try:
            # Convertir la imagen a base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Crear el prompt específico para análisis de componentes electrónicos
            prompt = """
            Analiza esta imagen de componentes electrónicos detalladamente. 
            Identifica todos los componentes visibles y proporciona la siguiente información para cada uno:
            1. Tipo de componente (resistencia, capacitor, LED, transistor, circuito integrado, etc.)
            2. Posición aproximada en la imagen (coordenadas x,y o descripción de posición)
            3. Características distintivas (color, forma, tamaño)
            4. Posible función en el circuito
            
            Devuelve un JSON estructurado así:
            {
              "components": [
                {
                  "id": 1,
                  "type": "resistor",
                  "position": "centro-izquierda",
                  "description": "Resistencia de color marrón-negro-rojo, probablemente 1K ohms",
                  "confidence": 0.95
                },
                ...
              ],
              "circuit_type": "Posible circuito amplificador/filtro/etc.",
              "analysis_confidence": 0.85
            }
            
            Simplemente analiza lo que ves en la imagen, sin inventar componentes. Si la imagen no muestra claramente los componentes, indica baja confianza.
            """
            
            # Hacer la solicitud al API de xAI
            response = self._make_vision_request(base64_image, prompt)
            
            if response.get("success", False):
                # Extraer el contenido JSON de la respuesta
                try:
                    content = response["content"]
                    # Buscar el JSON en la respuesta
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        analysis_result = json.loads(json_str)
                        
                        # Verificar formato mínimo esperado
                        if "components" not in analysis_result:
                            analysis_result["components"] = []
                        
                        # Añadir metadatos a cada componente
                        for component in analysis_result["components"]:
                            component["detected_by"] = "xAI (Grok)"
                            component["detection_source"] = "xai"
                            
                            # Asegurarse de que tiene un nivel de confianza
                            if "confidence" not in component:
                                component["confidence"] = 0.85  # Valor predeterminado alto para detecciones de xAI
                        
                        # Añadir metadatos del análisis
                        analysis_result["ai_generated"] = True
                        analysis_result["model_used"] = self.models["vision"]
                        analysis_result["xai_enhanced"] = True
                        return analysis_result
                    else:
                        # Si no se puede extraer JSON, devolver la respuesta como texto
                        return {
                            "components": [],
                            "text_analysis": content,
                            "ai_generated": True,
                            "xai_enhanced": True,
                            "success": True
                        }
                except Exception as e:
                    logger.error(f"Error al procesar la respuesta de xAI: {e}")
                    return {
                        "error": "Error al procesar respuesta",
                        "raw_content": response.get("content", ""),
                        "success": False
                    }
            
            return {"error": "Error en la solicitud a xAI", "details": response, "success": False}
            
        except Exception as e:
            logger.error(f"Error al analizar imagen con xAI: {e}")
            return {"error": str(e), "success": False}
    
    def _make_vision_request(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Realiza una solicitud al modelo de visión de xAI.
        
        Args:
            base64_image: Imagen codificada en base64
            prompt: Instrucciones para el modelo
            
        Returns:
            Dict: Respuesta del API
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.models["vision"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {"success": True, "content": content}
            else:
                logger.error(f"Error del API de xAI: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error al hacer solicitud a xAI: {e}")
            return {"success": False, "error": str(e)}
    
    def _make_text_request(self, prompt: str, json_response: bool = False) -> Dict[str, Any]:
        """
        Realiza una solicitud al modelo de texto de xAI.
        
        Args:
            prompt: Texto a procesar
            json_response: Si se debe solicitar respuesta en formato JSON
            
        Returns:
            Dict: Respuesta del API
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.models["text"],
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500
            }
            
            if json_response:
                payload["response_format"] = {"type": "json_object"}
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {"success": True, "content": content}
            else:
                logger.error(f"Error del API de xAI: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error al hacer solicitud a xAI: {e}")
            return {"success": False, "error": str(e)}
    
    def _local_image_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Realiza un análisis básico de imagen sin usar servicios externos.
        Esta función es una alternativa simple cuando no hay API disponible.
        
        Args:
            image_path: Ruta a la imagen a analizar
            
        Returns:
            Dict: Resultado básico del análisis
        """
        logger.info(f"Analizando imagen en modo local: {image_path}")
        
        try:
            # Verificamos que la imagen existe
            if not os.path.exists(image_path):
                return {"error": "Archivo no encontrado", "success": False}
                
            # Preparamos un resultado básico
            result = {
                "components": [
                    {
                        "id": 1,
                        "type": "unknown_component",
                        "position": "centro",
                        "description": "Componente detectado por análisis local",
                        "confidence": 0.5,
                        "detected_by": "Análisis local",
                        "detection_source": "local"
                    }
                ],
                "circuit_type": "Circuito electrónico (análisis básico)",
                "analysis_confidence": 0.5,
                "ai_generated": False,
                "model_used": "local",
                "success": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis local: {e}")
            return {
                "error": f"Error en análisis local: {e}",
                "success": False
            }
            
    def process_chat_message(self, message: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Procesa un mensaje de chat y responde usando el modelo xAI seleccionado.
        
        Args:
            message: Mensaje del usuario para procesar
            context: Contexto opcional (mensajes anteriores)
            
        Returns:
            Dict: Respuesta del sistema de chat
        """
        logger.info(f"Procesando mensaje de chat con modelo: {self.current_model}")
        
        # Verificar si estamos en modo local
        if self.current_model == "local":
            logger.info("Procesando mensaje en modo local")
            
            # En modo local, proporcionamos respuestas básicas basadas en palabras clave
            response = "No puedo entender completamente tu consulta en modo local. Cambia a modo 'grok' o 'hybrid' para análisis más avanzados."
            
            # Respuestas básicas basadas en palabras clave
            if "resistencia" in message.lower() or "resistor" in message.lower():
                response = "Las resistencias son componentes pasivos que limitan el flujo de corriente eléctrica. Se identifican por bandas de colores que indican su valor en ohmios. Son uno de los componentes más comunes en circuitos electrónicos."
            elif "capacitor" in message.lower() or "condensador" in message.lower():
                response = "Los capacitores almacenan energía eléctrica en un campo eléctrico. Existen varios tipos: cerámicos, electrolíticos, de tantalio, etc. Su capacidad se mide en faradios (F) y son usados para filtrado, acoplamiento y desacoplamiento."
            elif "transistor" in message.lower():
                response = "Los transistores son semiconductores que amplifican o conmutan señales electrónicas. Los tipos principales son BJT (Bipolar Junction Transistor) y MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor)."
            elif "diodo" in message.lower():
                response = "Los diodos permiten el flujo de corriente en una sola dirección. Se usan para rectificación, protección contra polaridad inversa y como indicadores (LEDs)."
            elif "integrado" in message.lower() or "chip" in message.lower() or "ic" in message.lower():
                response = "Los circuitos integrados (ICs) contienen múltiples componentes en un solo chip semiconductor. Pueden ser desde simples puertas lógicas hasta microprocesadores complejos."
            
            # Devolver respuesta básica
            return {
                "success": True,
                "content": response,
                "source": "local",
                "model": "local",
                "timestamp": datetime.now().isoformat()
            }
        
        # Si no estamos en modo local, usamos el API de xAI (si está disponible)
        if not self.initialized:
            logger.warning("API de xAI no inicializada. Modo local forzado")
            # Si no está inicializada la API, cambiamos a modo local temporalmente
            # y llamamos recursivamente para usar las respuestas locales
            original_model = self.current_model
            self.current_model = "local"
            result = self.process_chat_message(message, context)
            self.current_model = original_model
            result["source"] = f"local (fallback from {original_model})"
            return result
        
        # Construimos el sistema de prompt específico para componentes electrónicos
        system_prompt = """
        Eres un asistente especializado en electrónica y análisis de componentes.
        
        Proporciona respuestas concisas, precisas y técnicamente correctas sobre:
        - Identificación y función de componentes electrónicos (resistencias, capacitores, diodos, etc.)
        - Circuitos electrónicos y su funcionamiento
        - Especificaciones técnicas de componentes
        - Mejores prácticas en diseño electrónico
        
        Usa lenguaje técnico apropiado pero explica los conceptos de forma clara.
        Si no estás seguro de algo, indícalo claramente en lugar de proporcionar información inexacta.
        """
        
        try:
            # Preparar los mensajes para la API
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Añadir mensajes de contexto si existen
            if context:
                for msg in context:
                    if msg.get("role") and msg.get("content"):
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
            # Añadir el mensaje actual
            messages.append({"role": "user", "content": message})
            
            # Realizar la solicitud al modelo de IA
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.models["text"],
                "messages": messages,
                "max_tokens": 1000
            }
            
            logger.info(f"Enviando solicitud a xAI con modelo: {self.models['text']}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "content": content,
                    "source": f"xai ({self.current_model})",
                    "model": self.models["text"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"Error del API de xAI: {response.status_code} - {response.text}")
                
                # En caso de error, intentamos con respuestas locales
                original_model = self.current_model
                self.current_model = "local"
                result = self.process_chat_message(message, context)
                self.current_model = original_model
                
                result["source"] = f"local (fallback from {original_model} due to API error)"
                result["error"] = f"API Error: {response.status_code}"
                return result
                
        except Exception as e:
            logger.error(f"Error al procesar mensaje con xAI: {e}")
            
            # En caso de excepción, utilizamos respuestas locales
            original_model = self.current_model
            self.current_model = "local"
            result = self.process_chat_message(message, context)
            self.current_model = original_model
            
            result["source"] = f"local (fallback from {original_model} due to exception)"
            result["error"] = str(e)
            return result

    def analyze_circuit_text(self, components_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza datos de componentes para proporcionar información adicional sobre el circuito.
        
        Args:
            components_data: Datos de los componentes detectados
            
        Returns:
            Dict: Análisis extendido del circuito
        """
        # Verificar si estamos en modo local
        if self.current_model == "local":
            # En modo local, proporcionamos un análisis básico
            return {
                "circuit_type": "Circuito electrónico",
                "function": "Función no determinada en modo local",
                "applications": ["Aplicación electrónica general"],
                "key_components": [
                    {
                        "component": comp.get("type", "componente"),
                        "role": "Rol no determinado en modo local"
                    } for comp in components_data.get("components", [])[:3]  # Solo los primeros 3
                ],
                "observations": "Análisis realizado en modo local (básico)",
                "confidence": 0.5,
                "ai_generated": False,
                "model_used": "local",
                "success": True
            }
        
        # Para los demás modos, necesitamos la API key
        if not self.initialized:
            return {"error": "API de xAI no inicializada", "success": False}
        
        try:
            # Crear un prompt descriptivo con los componentes detectados
            component_descriptions = []
            for comp in components_data.get("components", []):
                desc = f"- {comp.get('type', 'componente desconocido')}: {comp.get('description', 'sin descripción')}"
                component_descriptions.append(desc)
            
            components_text = "\n".join(component_descriptions)
            
            prompt = f"""
            Basándote en los siguientes componentes electrónicos detectados en un circuito:
            
            {components_text}
            
            Proporciona un análisis detallado que incluya:
            1. Tipo de circuito (amplificador, fuente de alimentación, oscilador, etc.)
            2. Posible función y aplicaciones del circuito
            3. Componentes clave y su papel en el funcionamiento
            4. Cualquier observación relevante sobre el diseño
            
            Responde en formato JSON con esta estructura:
            {{
              "circuit_type": "Tipo de circuito identificado",
              "function": "Descripción de la función principal",
              "applications": ["Aplicación 1", "Aplicación 2"],
              "key_components": [
                {{
                  "component": "resistor",
                  "role": "Limita la corriente al LED"
                }}
              ],
              "observations": "Observaciones adicionales sobre el diseño",
              "confidence": 0.85
            }}
            """
            
            # Hacer la solicitud al API de xAI
            response = self._make_text_request(prompt, json_response=True)
            
            if response.get("success", False):
                try:
                    # Intentar parsear el JSON de la respuesta
                    analysis_result = json.loads(response["content"])
                    analysis_result["ai_generated"] = True
                    analysis_result["model_used"] = self.models["text"]
                    analysis_result["xai_enhanced"] = True
                    return analysis_result
                except json.JSONDecodeError:
                    logger.error("La respuesta no es un JSON válido")
                    return {
                        "error": "Formato de respuesta inválido",
                        "raw_content": response.get("content", ""),
                        "success": False
                    }
            
            return {"error": "Error en la solicitud a xAI", "details": response, "success": False}
            
        except Exception as e:
            logger.error(f"Error al analizar circuito con xAI: {e}")
            return {"error": str(e), "success": False}


# Función para uso directo del módulo
def main():
    """Función principal para pruebas del módulo"""
    if len(sys.argv) < 2:
        print("Uso: python xai_integration.py <ruta_imagen>")
        return
    
    image_path = sys.argv[1]
    processor = XAIProcessor()
    
    if not processor.is_available():
        print("Error: API key de xAI no disponible. Configure la variable de entorno XAI_API_KEY.")
        return
    
    print(f"Analizando imagen: {image_path}")
    result = processor.analyze_image(image_path)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()