"""
Módulo para gestionar la base de datos
"""
import os
import json
import sqlite3
import logging
from datetime import datetime

# Configurar el sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database_handler')

class DatabaseHandler:
    def __init__(self, db_path="data/electronic_analyzer.db"):
        """Inicializa el manejador de la base de datos."""
        # Asegurar que existe el directorio de datos
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def get_connection(self):
        """Obtiene una conexión a la base de datos."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Para acceder a las columnas por nombre
        return self.conn
    
    def initialize_database(self):
        """Inicializa la base de datos con las tablas necesarias si no existen."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Crear tabla de archivos
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                size INTEGER NOT NULL,
                type TEXT NOT NULL,
                extension TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                preview_path TEXT
            )
            ''')
            
            # Crear tabla para resultados de análisis
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                file_id TEXT PRIMARY KEY,
                results TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files (id)
            )
            ''')
            
            conn.commit()
            logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar la base de datos: {e}")
            if self.conn:
                self.conn.rollback()
    
    def save_file_info(self, file_info):
        """Guarda información de un archivo en la base de datos."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Insertar información del archivo
            cursor.execute('''
            INSERT OR REPLACE INTO files
            (id, name, path, size, type, extension, upload_date, preview_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_info['id'],
                file_info['name'],
                file_info['path'],
                file_info['size'],
                file_info['type'],
                file_info['extension'],
                file_info['upload_date'],
                file_info.get('preview_path', '')
            ))
            
            conn.commit()
            logger.info(f"Información del archivo guardada: {file_info['name']}")
            return file_info['id']
        except Exception as e:
            logger.error(f"Error al guardar información del archivo: {e}")
            if self.conn:
                self.conn.rollback()
            return None
    
    def get_file_by_id(self, file_id):
        """Obtiene información de un archivo por su ID."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM files WHERE id = ?', (file_id,))
            row = cursor.fetchone()
            
            if row:
                # Convertir a diccionario
                file_info = dict(row)
                return file_info
            
            return None
        except Exception as e:
            logger.error(f"Error al obtener archivo por ID: {e}")
            return None
    
    def get_all_files(self):
        """Obtiene todos los archivos en la base de datos."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM files ORDER BY upload_date DESC')
            rows = cursor.fetchall()
            
            # Convertir a lista de diccionarios
            files = [dict(row) for row in rows]
            return files
        except Exception as e:
            logger.error(f"Error al obtener todos los archivos: {e}")
            return []
    
    def save_analysis_results(self, file_id, results):
        """Guarda los resultados de un análisis en la base de datos."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Convertir resultados a JSON
            results_json = json.dumps(results)
            analysis_date = datetime.now().isoformat()
            
            # Insertar o actualizar resultados
            cursor.execute('''
            INSERT OR REPLACE INTO analysis_results
            (file_id, results, analysis_date)
            VALUES (?, ?, ?)
            ''', (file_id, results_json, analysis_date))
            
            conn.commit()
            logger.info(f"Resultados de análisis guardados para archivo: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar resultados de análisis: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def get_analysis_results(self, file_id):
        """Obtiene los resultados del análisis para un archivo específico."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM analysis_results WHERE file_id = ?', (file_id,))
            row = cursor.fetchone()
            
            if row:
                # Convertir a diccionario y deserializar JSON
                result_data = dict(row)
                result_data['results'] = json.loads(result_data['results'])
                return result_data['results']
            
            return None
        except Exception as e:
            logger.error(f"Error al obtener resultados de análisis: {e}")
            return None
    
    def get_analyzed_files(self):
        """Obtiene la lista de archivos que han sido analizados."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Obtener archivos con análisis
            cursor.execute('''
            SELECT f.* FROM files f
            INNER JOIN analysis_results ar ON f.id = ar.file_id
            ORDER BY ar.analysis_date DESC
            ''')
            rows = cursor.fetchall()
            
            # Convertir a lista de diccionarios
            files = [dict(row) for row in rows]
            return files
        except Exception as e:
            logger.error(f"Error al obtener archivos analizados: {e}")
            return []
    
    def __del__(self):
        """Cierra la conexión a la base de datos cuando el objeto es destruido."""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Código de prueba
    db = DatabaseHandler()
    
    # Probar la creación de un archivo
    test_file = {
        'id': 'test123',
        'name': 'test.jpg',
        'path': '/path/to/test.jpg',
        'size': 1024,
        'type': 'image',
        'extension': '.jpg',
        'upload_date': datetime.now().isoformat(),
        'preview_path': '/path/to/preview.jpg'
    }
    
    db.save_file_info(test_file)
    
    # Probar obtener archivo por ID
    retrieved_file = db.get_file_by_id('test123')
    if retrieved_file:
        print(f"Archivo recuperado: {retrieved_file['name']}")
    
    # Probar guardar y obtener resultados de análisis
    test_results = {
        'components': [
            {'id': 1, 'type': 'resistor', 'value': '10k'}
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    db.save_analysis_results('test123', test_results)
    
    # Probar obtener resultados
    retrieved_results = db.get_analysis_results('test123')
    if retrieved_results:
        print(f"Resultados recuperados: {len(retrieved_results['components'])} componentes")