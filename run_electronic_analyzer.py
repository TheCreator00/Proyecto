import os
import sys

# Asegurarse de que los módulos puedan importarse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importar los módulos necesarios
try:
    # Verificar las importaciones de Kivy primero
    import kivy
    print(f"Kivy versión: {kivy.__version__}")
    
    # Importar otros módulos necesarios
    import numpy as np
    print(f"NumPy versión: {np.__version__}")
    
    import cv2
    print(f"OpenCV versión: {cv2.__version__}")
    
    import matplotlib
    print(f"Matplotlib versión: {matplotlib.__version__}")
    
    import fitz
    print(f"PyMuPDF (fitz) versión: {fitz.__version__}")
    
    from PIL import Image
    print(f"Pillow versión: {Image.__version__}")
    
    print("Todas las dependencias están instaladas correctamente.")
    
    # Importar nuestros módulos
    import database_handler
    import file_processor
    import ai_analyzer
    import visualization
    
    print("Módulos personalizados importados correctamente.")
    
except ImportError as e:
    print(f"Error al importar dependencias: {e}")
    sys.exit(1)

# Ejecutar la aplicación
try:
    print("Iniciando la aplicación de análisis de electrónica...")
    
    # Crear directorios necesarios
    directories = ['data', 'uploads', 'uploads/previews', 'results', 'results/images', 'results/reports', 'assets', 'outputs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directorio creado: {directory}")
    
    # Crear imágenes placeholder
    placeholder_path = 'assets/placeholder.png'
    logo_path = 'assets/logo.png'
    
    if not os.path.exists(placeholder_path):
        # Crear una imagen simple como placeholder
        from PIL import ImageDraw
        img = Image.new('RGB', (400, 300), color=(240, 240, 240))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 399, 299], outline=(200, 200, 200), width=2)
        d.text((150, 150), "No Image", fill=(100, 100, 100))
        img.save(placeholder_path)
        print(f"Imagen placeholder creada: {placeholder_path}")
    
    if not os.path.exists(logo_path):
        # Crear un logo simple
        from PIL import ImageDraw
        img = Image.new('RGB', (400, 300), color=(200, 230, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([50, 50, 350, 250], fill=(100, 150, 250))
        d.text((120, 150), "Electronic Analyzer", fill=(255, 255, 255))
        img.save(logo_path)
        print(f"Imagen logo creada: {logo_path}")
    
    # Inicializar base de datos
    db = database_handler.DatabaseHandler()
    db.initialize_database()
    print("Base de datos inicializada.")
    
    # Ahora que todo está configurado, iniciar la aplicación
    from main import ElectronicAnalyzerApp
    print("Ejecutando aplicación principal...")
    ElectronicAnalyzerApp().run()
    
except Exception as e:
    print(f"Error al iniciar la aplicación: {e}")
    sys.exit(1)