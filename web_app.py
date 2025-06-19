"""
Aplicación web para el analizador de componentes electrónicos
"""
import os
import json
import time
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Importar módulos de la aplicación
from database_handler import DatabaseHandler
from file_processor import FileProcessor
from ai_analyzer import AIAnalyzer
from visualization import ResultVisualizer
import xai_integration

# Crear una instancia del procesador xAI
xai_processor = xai_integration.XAIProcessor()

# Configuración de la aplicación
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf', 'bmp', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB máximo
app.secret_key = os.urandom(24)

# Crear instancias de las clases principales
db_handler = DatabaseHandler()
file_processor = FileProcessor()
ai_analyzer = AIAnalyzer()
visualizer = ResultVisualizer()

# Asegurar que existen los directorios necesarios
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('results', exist_ok=True)

def allowed_file(filename):
    """Verifica si un archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Página principal"""
    # Obtener archivos cargados y analizados
    files = db_handler.get_all_files()
    analyzed_files = db_handler.get_analyzed_files()
    
    # Ordenar por fecha más reciente
    if files:
        files.sort(key=lambda x: x['upload_date'], reverse=True)
    
    if analyzed_files:
        analyzed_files.sort(key=lambda x: x['upload_date'], reverse=True)
    
    return render_template('index.html', files=files, analyzed_files=analyzed_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Subir un archivo"""
    # Verificar si hay un archivo en la solicitud
    if 'file' not in request.files:
        return render_template('index.html', status='No se seleccionó ningún archivo', error=True)
    
    file = request.files['file']
    
    # Verificar que el archivo sea válido
    if file.filename == '':
        return render_template('index.html', status='Nombre de archivo vacío', error=True)
    
    if not allowed_file(file.filename):
        return render_template('index.html', 
                              status=f'Formato no soportado. Formatos permitidos: {", ".join(ALLOWED_EXTENSIONS)}', 
                              error=True)
    
    try:
        # Generar un nombre seguro y único para el archivo
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        
        # Guardar el archivo
        file.save(file_path)
        
        # Procesar el archivo para obtener información y vista previa
        file_info = file_processor.process_file(file_path)
        
        # Guardar información en la base de datos
        db_handler.save_file_info(file_info)
        
        return redirect(url_for('index'))
    
    except Exception as e:
        app.logger.error(f"Error al subir archivo: {e}")
        return render_template('index.html', status=f'Error al procesar el archivo: {str(e)}', error=True)

@app.route('/preview/<file_id>')
def get_preview(file_id):
    """Obtener la vista previa de un archivo"""
    try:
        file_info = db_handler.get_file_by_id(file_id)
        
        if not file_info:
            return "Archivo no encontrado", 404
        
        # Si existe una vista previa, mostrarla
        if 'preview_path' in file_info and os.path.exists(file_info['preview_path']):
            directory = os.path.dirname(file_info['preview_path'])
            filename = os.path.basename(file_info['preview_path'])
            return send_from_directory(directory, filename)
        
        # Si no hay vista previa disponible
        return send_from_directory('static', 'no_preview.png')
    
    except Exception as e:
        app.logger.error(f"Error al obtener vista previa: {e}")
        return "Error al obtener vista previa", 500

@app.route('/analyze/<file_id>')
def analyze_file(file_id):
    """Analizar un archivo"""
    try:
        file_info = db_handler.get_file_by_id(file_id)
        
        if not file_info:
            return render_template('index.html', status='Archivo no encontrado', error=True)
        
        # Verificar que existe el archivo físicamente
        if not os.path.exists(file_info['path']):
            return render_template('index.html', status='El archivo físico no existe', error=True)
        
        # Analizar el archivo
        results = ai_analyzer.analyze_file(file_info['path'])
        
        # Verificar que no haya error en el análisis
        if 'error' in results:
            return render_template('index.html', 
                                  status=f'Error en el análisis: {results["error"]}', 
                                  error=True)
        
        # Guardar resultados en la base de datos
        db_handler.save_analysis_results(file_id, results)
        
        # Redirigir a la página de resultados
        return redirect(url_for('view_results', file_id=file_id))
    
    except Exception as e:
        app.logger.error(f"Error al analizar archivo: {e}")
        return render_template('index.html', status=f'Error en el análisis: {str(e)}', error=True)

@app.route('/results/<file_id>')
def view_results(file_id):
    """Ver resultados de análisis"""
    try:
        # Obtener información del archivo y resultados
        file_info = db_handler.get_file_by_id(file_id)
        analysis_results = db_handler.get_analysis_results(file_id)
        
        if not file_info or not analysis_results:
            return render_template('index.html', status='Resultados no encontrados', error=True)
        
        # Visualizar resultados en una imagen
        result_image = visualizer.visualize_results(analysis_results, file_info['path'])
        
        # Obtener todas las listas para el contexto de la plantilla
        files = db_handler.get_all_files()
        analyzed_files = db_handler.get_analyzed_files()
        
        return render_template('index.html', 
                              files=files, 
                              analyzed_files=analyzed_files,
                              result_file=file_info,
                              result_image=result_image)
    
    except Exception as e:
        app.logger.error(f"Error al mostrar resultados: {e}")
        return render_template('index.html', status=f'Error al mostrar resultados: {str(e)}', error=True)

@app.route('/report/<file_id>')
def view_report(file_id):
    """Ver informe detallado"""
    try:
        # Obtener información del archivo y resultados
        file_info = db_handler.get_file_by_id(file_id)
        analysis_results = db_handler.get_analysis_results(file_id)
        
        if not file_info or not analysis_results:
            return render_template('index.html', status='Resultados no encontrados', error=True)
        
        # Visualizar resultados en una imagen
        result_image = visualizer.visualize_results(analysis_results, file_info['path'])
        
        # Crear una función para obtener iconos de componentes
        def get_component_icon(component_type):
            icon_map = {
                'resistor': 'device-line',
                'capacitor': 'battery-2-charge-line',
                'inductor': 'spiral-line',
                'diode': 'share-forward-line',
                'transistor': 'chip-line',
                'led': 'lightbulb-flash-line',
                'ic': 'cpu-line',
                'connector': 'plug-line',
                'switch': 'toggle-line',
                'relay': 'connection-line',
                'transformer': 'terminal-box-line'
            }
            return icon_map.get(component_type, 'question-line')
        
        # Generar distribución de componentes y visualización del grafo
        component_plot = visualizer.generate_component_plot(analysis_results, file_info['name'])
        connection_graph = visualizer.generate_graph_visualization(analysis_results, file_info['name'])
        
        return render_template('report.html', 
                              file=file_info,
                              results=analysis_results,
                              result_image=result_image,
                              component_plot=component_plot,
                              connection_graph=connection_graph,
                              get_component_icon=get_component_icon)
    
    except Exception as e:
        app.logger.error(f"Error al mostrar informe: {e}")
        return render_template('index.html', status=f'Error al mostrar informe: {str(e)}', error=True)

@app.route('/export/<file_id>')
def export_results(file_id):
    """Exportar resultados"""
    try:
        # Obtener información del archivo y resultados
        file_info = db_handler.get_file_by_id(file_id)
        analysis_results = db_handler.get_analysis_results(file_id)
        
        if not file_info or not analysis_results:
            return render_template('index.html', status='Resultados no encontrados', error=True)
        
        # Verificar si se especificó un formato específico
        format_type = request.args.get('format', 'json')
        
        # Exportar resultados según el formato solicitado
        exports = visualizer.export_results(analysis_results, file_info['name'])
        
        if format_type == 'json' and 'json' in exports:
            return send_from_directory(os.path.dirname(exports['json']), os.path.basename(exports['json']), as_attachment=True)
        elif format_type == 'csv' and 'csv' in exports:
            return send_from_directory(os.path.dirname(exports['csv']), os.path.basename(exports['csv']), as_attachment=True)
        elif format_type == 'pdf':
            # Generar un informe HTML primero
            report_path = visualizer.generate_html_report(analysis_results, file_info['name'], file_info['path'])
            if report_path:
                return send_from_directory(os.path.dirname(report_path), os.path.basename(report_path), as_attachment=True)
        elif format_type == 'png' and 'component_plot' in exports:
            return send_from_directory(os.path.dirname(exports['component_plot']), os.path.basename(exports['component_plot']), as_attachment=True)
            
        # Redirigir a la página de resultados si no se pudo exportar
        return redirect(url_for('view_results', file_id=file_id))
    
    except Exception as e:
        app.logger.error(f"Error al exportar resultados: {e}")
        return render_template('index.html', status=f'Error al exportar resultados: {str(e)}', error=True)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Servir archivos estáticos"""
    return send_from_directory('static', filename)

# Ruta para API de chat
@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API para chat con IA sobre componentes electrónicos"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Datos de solicitud no válidos'}), 400
        
        message = data.get('message', '')
        model = data.get('model', 'grok')
        files = data.get('files', [])
        
        # Establecer el modelo en el procesador xAI
        xai_processor.set_model(model)
        
        # Procesar el mensaje con el modelo adecuado
        response = xai_processor.process_chat_message(message)
        
        if response.get('success', False):
            return jsonify({
                'id': str(uuid.uuid4()),
                'content': response.get('content', 'No se pudo procesar la consulta'),
                'timestamp': response.get('timestamp', datetime.now().isoformat()),
                'model': model
            })
        else:
            # Error al procesar el mensaje
            app.logger.error(f"Error en chat API: {response.get('error', 'Error desconocido')}")
            fallback_response = "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor intenta de nuevo."
            
            return jsonify({
                'id': str(uuid.uuid4()),
                'content': response.get('content', fallback_response),
                'timestamp': response.get('timestamp', datetime.now().isoformat()),
                'model': model,
                'error': response.get('error')
            })
    
    except Exception as e:
        app.logger.error(f"Error en chat API: {e}")
        return jsonify({
            'id': str(uuid.uuid4()),
            'content': "Ha ocurrido un error al procesar tu consulta. Por favor intenta de nuevo.",
            'timestamp': datetime.now().isoformat(),
            'model': 'error',
            'error': str(e)
        }), 500

# Ruta para cambiar el modelo de IA
@app.route('/api/model', methods=['POST'])
def set_model_api():
    """API para cambiar el modelo de IA"""
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({'success': False, 'error': 'Modelo no especificado'}), 400
        
        model = data['model']
        success = xai_processor.set_model(model)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Modelo establecido: {model}',
                'model': model
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Modelo no reconocido: {model}',
                'valid_models': list(xai_processor.available_models.keys())
            }), 400
            
    except Exception as e:
        app.logger.error(f"Error al establecer modelo: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Ruta para verificar el estado del servicio xAI
@app.route('/api/status', methods=['GET'])
def check_status_api():
    """API para verificar el estado del servicio xAI"""
    try:
        # Verificar si el servicio xAI está disponible
        is_available = xai_processor.is_available()
        
        return jsonify({
            'success': True,
            'available': is_available,
            'current_model': xai_processor.current_model,
            'api_key_configured': xai_processor.initialized
        })
            
    except Exception as e:
        app.logger.error(f"Error al verificar estado: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Ruta para guardar una imagen capturada con la cámara
@app.route('/save_capture', methods=['POST'])
def save_capture():
    """Guardar una captura de cámara"""
    try:
        # Verificar si hay datos de imagen
        if 'image_data' not in request.form:
            return jsonify({'success': False, 'error': 'No se recibieron datos de imagen'})
        
        image_data = request.form['image_data']
        
        # Eliminar el prefijo "data:image/jpeg;base64,"
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Generar un nombre único para la imagen
        unique_id = str(uuid.uuid4())
        filename = f"camera_capture_{unique_id}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Guardar la imagen
        import base64
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        # Procesar el archivo para obtener información y vista previa
        file_info = file_processor.process_file(file_path)
        
        # Guardar información en la base de datos
        db_handler.save_file_info(file_info)
        
        return jsonify({'success': True, 'file_id': file_info['id']})
    
    except Exception as e:
        app.logger.error(f"Error al guardar captura: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Si se ejecuta directamente, usar puerto 5002
    app.run(debug=True, host='0.0.0.0', port=5002)