"""
Script para iniciar la aplicación de análisis de electrónica

Nota: La aplicación se ejecuta en el puerto 5002
"""
import os
import sys
import time
import subprocess

def check_dependencies():
    """Verifica que están instaladas todas las dependencias necesarias."""
    dependencies = ["flask", "cv2", "numpy", "matplotlib", "fitz"]
    missing = []
    
    for dep in dependencies:
        try:
            if dep == "cv2":
                try:
                    import cv2
                except ImportError:
                    missing.append("opencv-python")
            elif dep == "fitz":
                try:
                    import fitz
                except ImportError:
                    missing.append("pymupdf")
            else:
                __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"Faltan las siguientes dependencias: {', '.join(missing)}")
        print("Puedes instalarlas con: pip install " + " ".join(missing))
        return False
        
    return True

def create_directories():
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
            print(f"Directorio creado: {directory}")

def run_app():
    """Ejecuta la aplicación."""
    print("\n==== Iniciando Analizador de Componentes Electrónicos ====")
    print("Verificando dependencias...")
    
    if not check_dependencies():
        print("Por favor, instala las dependencias necesarias e inténtalo de nuevo.")
        sys.exit(1)
    
    print("Creando directorios necesarios...")
    create_directories()
    
    print("Iniciando aplicación web...")
    try:
        # Eliminamos cualquier proceso que esté usando el puerto 5002
        try:
            subprocess.run("kill -9 $(lsof -t -i:5002)", shell=True)
            print("Puerto 5002 liberado.")
        except:
            pass
        
        print("La aplicación estará disponible en: http://localhost:5002")
        print("O añadiendo ':5002' a la URL de tu replit")
        print("\nPresiona Ctrl+C para salir\n")
        
        # Ejecutar la aplicación web
        subprocess.run(["python", "web_app.py"])
        
    except KeyboardInterrupt:
        print("\nDeteniendo aplicación...")
    except Exception as e:
        print(f"Error al ejecutar la aplicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()