"""
Script para ejecutar la aplicación de análisis de componentes electrónicos de forma independiente
Para ser usado con PyInstaller para crear un ejecutable
"""

import os
import sys
import time
import logging
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analyzer_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("RunAppStandalone")

def create_required_directories():
    """Crea los directorios necesarios para la aplicación."""
    directories = [
        'templates', 
        'static', 
        'static/previews',
        'uploads',
        'outputs',
        'results',
        'data'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directorio creado: {directory}")

def get_application_path():
    """Obtiene la ruta de la aplicación, funciona tanto en desarrollo como en PyInstaller."""
    if getattr(sys, 'frozen', False):
        # Estamos ejecutando en un bundle (PyInstaller)
        application_path = os.path.dirname(sys.executable)
    else:
        # Estamos ejecutando en un entorno de desarrollo normal
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    return application_path

def copy_resources():
    """Copia recursos necesarios si estamos en modo ejecutable."""
    if getattr(sys, 'frozen', False):
        # Si estamos en modo ejecutable, configuramos las rutas
        import shutil
        base_path = get_application_path()
        
        # Aseguramos que existen los directorios en la carpeta del ejecutable
        create_required_directories()
        
        logger.info(f"Aplicación ejecutándose desde: {base_path}")

def run_app():
    """Ejecuta la aplicación."""
    logger.info("Iniciando Analizador de Componentes Electrónicos...")
    
    app_path = get_application_path()
    logger.info(f"Ruta de la aplicación: {app_path}")
    
    # Crear directorios necesarios
    create_required_directories()
    
    # Copiar recursos si es necesario
    copy_resources()
    
    try:
        # Importamos aquí para asegurarnos que los directorios ya fueron creados
        from web_app import app
        
        # Ejecutar la aplicación
        logger.info("Iniciando servidor web en http://localhost:5002")
        app.run(host='127.0.0.1', port=5002, debug=False)
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mantener ventana abierta en caso de error
        print("\nSe produjo un error al iniciar la aplicación.")
        print(f"Error: {str(e)}")
        print("Revisa el archivo de log para más detalles.")
        input("Presiona Enter para salir...")
        sys.exit(1)

if __name__ == "__main__":
    run_app()