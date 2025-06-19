# Analizador de Componentes Electrónicos

Esta aplicación permite analizar componentes electrónicos en archivos de imagen utilizando técnicas de procesamiento de imágenes y visión por computadora.

## Características

- **Análisis de componentes**: Detección y clasificación de componentes electrónicos como resistencias, capacitores, diodos, etc.
- **Detección de conexiones**: Identificación de conexiones entre componentes para analizar circuitos.
- **Generación de informes**: Creación de informes detallados con visualizaciones y estadísticas.
- **Captura de imágenes**: Capacidad para capturar imágenes directamente desde la cámara.
- **Exportación de resultados**: Exportación en diversos formatos (JSON, CSV, HTML, PNG).

## Cómo usar la aplicación

### Iniciar la aplicación

Hay dos formas de iniciar la aplicación:

1. Ejecutar el siguiente comando:
   ```
   python run_app.py
   ```

2. O ejecutar directamente:
   ```
   python web_app.py
   ```

La aplicación estará disponible en el puerto 5002:
- http://localhost:5002 (local)
- O tu-replit-url:5002 (en Replit)

### Interfaz de usuario

La aplicación tiene cuatro secciones principales:

1. **Cargar archivos**: Sube imágenes de componentes o esquemas electrónicos.
2. **Analizar**: Selecciona archivos para analizarlos.
3. **Ver resultados**: Visualiza los resultados de análisis.
4. **Exportar**: Exporta los resultados en diferentes formatos.

### Formatos de archivo admitidos

- Imágenes: JPG, JPEG, PNG, BMP, GIF
- Documentos: PDF

## Tecnologías utilizadas

- **Flask**: Framework web para la interfaz de usuario
- **OpenCV**: Biblioteca de visión por computadora para el análisis de imágenes
- **NumPy**: Procesamiento de matrices y operaciones matemáticas
- **Matplotlib**: Generación de gráficos y visualizaciones
- **PyMuPDF**: Procesamiento de documentos PDF

## Requisitos

- Python 3.6 o superior
- Dependencias específicas: flask, opencv-python, numpy, matplotlib, pymupdf

## Estructura de directorios

- `/templates`: Plantillas HTML para la interfaz web
- `/static`: Archivos estáticos (CSS, JavaScript, imágenes)
- `/uploads`: Archivos subidos por los usuarios
- `/outputs`: Resultados procesados y visualizaciones
- `/results`: Informes y exportaciones
- `/data`: Base de datos y archivos de configuración

## Notas

- La aplicación utiliza el puerto 5002 para evitar conflictos con otras aplicaciones
- Se implementa detección real de componentes, no simulaciones