# main.py - Punto de entrada principal de la aplicación

import os
import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.core.window import Window

# Importamos los módulos personalizados
from database_handler import DatabaseHandler
from file_processor import FileProcessor
from ai_analyzer import AIAnalyzer
from visualization import ResultVisualizer

# Configuración básica de Kivy
kivy.require('2.1.0')
Window.size = (1000, 800)

# Pantalla de inicio
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Título de la aplicación
        title_label = Label(
            text='Analizador de Electrónica con IA',
            font_size=32,
            size_hint=(1, 0.2)
        )
        
        # Botones de navegación
        btn_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.3))
        
        load_btn = Button(
            text='Cargar Archivo',
            on_press=self.go_to_load,
            background_color=(0.2, 0.6, 1, 1),
            font_size=20
        )
        
        analyze_btn = Button(
            text='Analizar Archivo',
            on_press=self.go_to_analyze,
            background_color=(0.2, 0.8, 0.2, 1),
            font_size=20
        )
        
        view_btn = Button(
            text='Ver Resultados',
            on_press=self.go_to_results,
            background_color=(1, 0.6, 0.2, 1),
            font_size=20
        )
        
        # Añadir elementos al layout
        btn_layout.add_widget(load_btn)
        btn_layout.add_widget(analyze_btn)
        btn_layout.add_widget(view_btn)
        
        # Descripción de la aplicación
        desc_label = Label(
            text='Esta aplicación te permite cargar archivos de electrónica,\n'
                 'analizarlos con Inteligencia Artificial y visualizar los resultados.',
            font_size=18,
            size_hint=(1, 0.3)
        )
        
        # Logo o imagen principal
        logo = Image(
            source='assets/logo.png',  # Asegúrate de tener esta imagen en una carpeta assets
            size_hint=(1, 0.4)
        )
        
        # Añadir todos los widgets al layout principal
        layout.add_widget(title_label)
        layout.add_widget(logo)
        layout.add_widget(desc_label)
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def go_to_load(self, instance):
        self.manager.current = 'load'
    
    def go_to_analyze(self, instance):
        self.manager.current = 'analyze'
    
    def go_to_results(self, instance):
        self.manager.current = 'results'

# Pantalla de carga de archivos
class LoadScreen(Screen):
    def __init__(self, **kwargs):
        super(LoadScreen, self).__init__(**kwargs)
        self.file_processor = FileProcessor()
        self.db_handler = DatabaseHandler()
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Título
        title_label = Label(
            text='Cargar Archivo de Electrónica',
            font_size=28,
            size_hint=(1, 0.1)
        )
        
        # Selector de archivos
        self.file_chooser = FileChooserListView(
            path=os.path.expanduser('~'),
            filters=['*.pdf', '*.jpg', '*.png', '*.jpeg', '*.bmp', '*.gif'],
            size_hint=(1, 0.7)
        )
        
        # Botones de acción
        btn_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.1))
        
        load_btn = Button(
            text='Cargar Archivo Seleccionado',
            on_press=self.load_file,
            background_color=(0.2, 0.6, 1, 1)
        )
        
        back_btn = Button(
            text='Volver al Inicio',
            on_press=self.go_back,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        
        btn_layout.add_widget(load_btn)
        btn_layout.add_widget(back_btn)
        
        # Etiqueta de estado
        self.status_label = Label(
            text='Selecciona un archivo para cargar',
            size_hint=(1, 0.1)
        )
        
        # Añadir widgets al layout principal
        layout.add_widget(title_label)
        layout.add_widget(self.file_chooser)
        layout.add_widget(btn_layout)
        layout.add_widget(self.status_label)
        
        self.add_widget(layout)
    
    def load_file(self, instance):
        if not self.file_chooser.selection:
            self.status_label.text = 'Error: No se ha seleccionado ningún archivo'
            return
        
        file_path = self.file_chooser.selection[0]
        
        try:
            # Procesar y guardar el archivo
            file_info = self.file_processor.process_file(file_path)
            self.db_handler.save_file_info(file_info)
            
            # Actualizar etiqueta de estado
            self.status_label.text = f'Archivo cargado exitosamente: {os.path.basename(file_path)}'
            
            # Mostrar popup de éxito
            popup = Popup(
                title='Archivo Cargado',
                content=Label(text=f'El archivo se ha cargado correctamente.\n¿Desea analizarlo ahora?'),
                size_hint=(0.6, 0.3)
            )
            
            # Layout para botones del popup
            popup_btn_layout = BoxLayout(orientation='horizontal', spacing=5)
            
            yes_btn = Button(text='Sí', on_press=lambda x: self.analyze_now(popup, file_info['id']))
            no_btn = Button(text='No', on_press=popup.dismiss)
            
            popup_btn_layout.add_widget(yes_btn)
            popup_btn_layout.add_widget(no_btn)
            
            popup.content = popup_btn_layout
            popup.open()
            
        except Exception as e:
            self.status_label.text = f'Error al cargar el archivo: {str(e)}'
    
    def analyze_now(self, popup, file_id):
        popup.dismiss()
        App.get_running_app().file_to_analyze = file_id
        self.manager.current = 'analyze'
    
    def go_back(self, instance):
        self.manager.current = 'home'

# Pantalla de análisis
class AnalyzeScreen(Screen):
    def __init__(self, **kwargs):
        super(AnalyzeScreen, self).__init__(**kwargs)
        self.ai_analyzer = AIAnalyzer()
        self.db_handler = DatabaseHandler()
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Título
        title_label = Label(
            text='Analizar Archivo con IA',
            font_size=28,
            size_hint=(1, 0.1)
        )
        
        # Área de visualización del archivo seleccionado
        self.image_view = Image(
            source='assets/placeholder.png',
            size_hint=(1, 0.5)
        )
        
        # Selector de archivos cargados
        file_selection_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        file_selection_label = Label(text='Seleccionar archivo cargado:')
        self.file_spinner_btn = Button(
            text='Seleccionar archivo...',
            on_press=self.show_file_selection
        )
        file_selection_layout.add_widget(file_selection_label)
        file_selection_layout.add_widget(self.file_spinner_btn)
        
        # Botones de acción
        btn_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.1))
        
        analyze_btn = Button(
            text='Iniciar Análisis',
            on_press=self.start_analysis,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        
        back_btn = Button(
            text='Volver al Inicio',
            on_press=self.go_back,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        
        btn_layout.add_widget(analyze_btn)
        btn_layout.add_widget(back_btn)
        
        # Etiqueta de estado
        self.status_label = Label(
            text='Selecciona un archivo para analizar',
            size_hint=(1, 0.1)
        )
        
        # Añadir widgets al layout principal
        layout.add_widget(title_label)
        layout.add_widget(self.image_view)
        layout.add_widget(file_selection_layout)
        layout.add_widget(btn_layout)
        layout.add_widget(self.status_label)
        
        self.add_widget(layout)
        
        # Variables para almacenar el archivo seleccionado
        self.selected_file_id = None
        
    def on_enter(self):
        # Si hay un archivo para analizar pasado desde otra pantalla
        app = App.get_running_app()
        if hasattr(app, 'file_to_analyze') and app.file_to_analyze:
            self.selected_file_id = app.file_to_analyze
            file_info = self.db_handler.get_file_by_id(self.selected_file_id)
            if file_info:
                self.file_spinner_btn.text = file_info['name']
                self.image_view.source = file_info['preview_path']
            app.file_to_analyze = None
    
    def show_file_selection(self, instance):
        # Obtener lista de archivos de la base de datos
        files = self.db_handler.get_all_files()
        
        # Crear layout para el popup
        content_layout = BoxLayout(orientation='vertical', spacing=5)
        
        # Crear botones para cada archivo
        for file in files:
            btn = Button(
                text=file['name'],
                on_press=lambda x, fid=file['id']: self.select_file(fid, x.text)
            )
            content_layout.add_widget(btn)
        
        # Crear y mostrar popup
        popup = Popup(
            title='Seleccionar Archivo',
            content=content_layout,
            size_hint=(0.7, 0.7)
        )
        popup.open()
    
    def select_file(self, file_id, file_name):
        self.selected_file_id = file_id
        self.file_spinner_btn.text = file_name
        
        # Obtener información del archivo y mostrar vista previa
        file_info = self.db_handler.get_file_by_id(file_id)
        if file_info and 'preview_path' in file_info:
            self.image_view.source = file_info['preview_path']
        
        # Cerrar cualquier popup abierto
        for widget in Window.children:
            if isinstance(widget, Popup):
                widget.dismiss()
    
    def start_analysis(self, instance):
        if not self.selected_file_id:
            self.status_label.text = 'Error: No se ha seleccionado ningún archivo'
            return
        
        try:
            # Obtener información del archivo
            file_info = self.db_handler.get_file_by_id(self.selected_file_id)
            
            # Actualizar etiqueta de estado
            self.status_label.text = f'Analizando archivo: {file_info["name"]}...'
            
            # Iniciar análisis
            analysis_results = self.ai_analyzer.analyze_file(file_info['path'])
            
            # Guardar resultados en la base de datos
            self.db_handler.save_analysis_results(self.selected_file_id, analysis_results)
            
            # Actualizar etiqueta de estado
            self.status_label.text = f'Análisis completado. Se encontraron {len(analysis_results["components"])} componentes.'
            
            # Mostrar popup de éxito
            popup = Popup(
                title='Análisis Completado',
                content=Label(text='El análisis se ha completado correctamente.\n¿Desea ver los resultados ahora?'),
                size_hint=(0.6, 0.3)
            )
            
            # Layout para botones del popup
            popup_btn_layout = BoxLayout(orientation='horizontal', spacing=5)
            
            yes_btn = Button(text='Sí', on_press=lambda x: self.view_results_now(popup, self.selected_file_id))
            no_btn = Button(text='No', on_press=popup.dismiss)
            
            popup_btn_layout.add_widget(yes_btn)
            popup_btn_layout.add_widget(no_btn)
            
            popup.content = popup_btn_layout
            popup.open()
            
        except Exception as e:
            self.status_label.text = f'Error durante el análisis: {str(e)}'
    
    def view_results_now(self, popup, file_id):
        popup.dismiss()
        App.get_running_app().file_to_view = file_id
        self.manager.current = 'results'
    
    def go_back(self, instance):
        self.manager.current = 'home'

# Pantalla de resultados
class ResultsScreen(Screen):
    def __init__(self, **kwargs):
        super(ResultsScreen, self).__init__(**kwargs)
        self.db_handler = DatabaseHandler()
        self.visualizer = ResultVisualizer()
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Título
        title_label = Label(
            text='Resultados del Análisis',
            font_size=28,
            size_hint=(1, 0.1)
        )
        
        # Área de visualización de resultados
        self.results_view = Image(
            source='assets/placeholder.png',
            size_hint=(1, 0.5)
        )
        
        # Selector de archivos analizados
        file_selection_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        file_selection_label = Label(text='Seleccionar archivo analizado:')
        self.file_spinner_btn = Button(
            text='Seleccionar archivo...',
            on_press=self.show_file_selection
        )
        file_selection_layout.add_widget(file_selection_label)
        file_selection_layout.add_widget(self.file_spinner_btn)
        
        # Botones de acción
        btn_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.1))
        
        view_details_btn = Button(
            text='Ver Detalles',
            on_press=self.show_details,
            background_color=(0.2, 0.6, 1, 1)
        )
        
        export_btn = Button(
            text='Exportar Resultados',
            on_press=self.export_results,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        
        back_btn = Button(
            text='Volver al Inicio',
            on_press=self.go_back,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        
        btn_layout.add_widget(view_details_btn)
        btn_layout.add_widget(export_btn)
        btn_layout.add_widget(back_btn)
        
        # Etiqueta de estado
        self.status_label = Label(
            text='Selecciona un archivo para ver resultados',
            size_hint=(1, 0.1)
        )
        
        # Añadir widgets al layout principal
        layout.add_widget(title_label)
        layout.add_widget(self.results_view)
        layout.add_widget(file_selection_layout)
        layout.add_widget(btn_layout)
        layout.add_widget(self.status_label)
        
        self.add_widget(layout)
        
        # Variables para almacenar el archivo seleccionado
        self.selected_file_id = None
        self.current_results = None
    
    def on_enter(self):
        # Si hay un archivo para ver pasado desde otra pantalla
        app = App.get_running_app()
        if hasattr(app, 'file_to_view') and app.file_to_view:
            self.selected_file_id = app.file_to_view
            file_info = self.db_handler.get_file_by_id(self.selected_file_id)
            if file_info:
                self.file_spinner_btn.text = file_info['name']
                self.load_results(self.selected_file_id)
            app.file_to_view = None
    
    def show_file_selection(self, instance):
        # Obtener lista de archivos analizados de la base de datos
        files = self.db_handler.get_analyzed_files()
        
        # Crear layout para el popup
        content_layout = BoxLayout(orientation='vertical', spacing=5)
        
        # Crear botones para cada archivo
        for file in files:
            btn = Button(
                text=file['name'],
                on_press=lambda x, fid=file['id']: self.select_file(fid, x.text)
            )
            content_layout.add_widget(btn)
        
        # Crear y mostrar popup
        popup = Popup(
            title='Seleccionar Archivo Analizado',
            content=content_layout,
            size_hint=(0.7, 0.7)
        )
        popup.open()
    
    def select_file(self, file_id, file_name):
        self.selected_file_id = file_id
        self.file_spinner_btn.text = file_name
        self.load_results(file_id)
        
        # Cerrar cualquier popup abierto
        for widget in Window.children:
            if isinstance(widget, Popup):
                widget.dismiss()
    
    def load_results(self, file_id):
        try:
            # Obtener información del archivo
            file_info = self.db_handler.get_file_by_id(file_id)
            
            # Obtener resultados del análisis
            analysis_results = self.db_handler.get_analysis_results(file_id)
            
            if not analysis_results:
                self.status_label.text = 'No se encontraron resultados para este archivo'
                return
            
            self.current_results = analysis_results['results']
            
            # Visualizar resultados
            visualized_path = self.visualizer.visualize_results(
                self.current_results, 
                file_info['path']
            )
            
            # Mostrar imagen con anotaciones
            if visualized_path and os.path.exists(visualized_path):
                self.results_view.source = visualized_path
                self.status_label.text = f'Mostrando resultados para: {file_info["name"]}'
            else:
                self.status_label.text = 'Error al visualizar resultados'
                
        except Exception as e:
            self.status_label.text = f'Error al cargar resultados: {str(e)}'
    
    def show_details(self, instance):
        if not self.selected_file_id or not self.current_results:
            self.status_label.text = 'No hay resultados para mostrar'
            return
        
        try:
            # Obtener información del archivo
            file_info = self.db_handler.get_file_by_id(self.selected_file_id)
            
            # Generar informe HTML
            html_path = self.visualizer.generate_html_report(
                self.current_results,
                file_info['name']
            )
            
            if html_path and os.path.exists(html_path):
                # Mostrar mensaje sobre el informe generado
                self.status_label.text = f'Informe HTML generado: {html_path}'
                
                # Intentar abrir el informe con el navegador predeterminado
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(html_path))
            else:
                self.status_label.text = 'Error al generar informe detallado'
                
        except Exception as e:
            self.status_label.text = f'Error al mostrar detalles: {str(e)}'
    
    def export_results(self, instance):
        if not self.selected_file_id or not self.current_results:
            self.status_label.text = 'No hay resultados para exportar'
            return
        
        try:
            # Obtener información del archivo
            file_info = self.db_handler.get_file_by_id(self.selected_file_id)
            
            # Exportar resultados
            export_paths = self.visualizer.export_results(
                self.current_results,
                file_info['name']
            )
            
            if export_paths:
                # Crear mensaje con las rutas de exportación
                export_msg = 'Resultados exportados:\n'
                for format_name, path in export_paths.items():
                    export_msg += f'- {format_name.upper()}: {path}\n'
                
                # Mostrar popup con información de exportación
                popup = Popup(
                    title='Exportación Completada',
                    content=Label(text=export_msg),
                    size_hint=(0.7, 0.7)
                )
                popup.open()
                
                self.status_label.text = 'Exportación completada'
            else:
                self.status_label.text = 'Error al exportar resultados'
                
        except Exception as e:
            self.status_label.text = f'Error durante la exportación: {str(e)}'
    
    def go_back(self, instance):
        self.manager.current = 'home'

# Aplicación principal
class ElectronicAnalyzerApp(App):
    def build(self):
        # Crear el gestor de pantallas
        sm = ScreenManager()
        
        # Añadir pantallas
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(LoadScreen(name='load'))
        sm.add_widget(AnalyzeScreen(name='analyze'))
        sm.add_widget(ResultsScreen(name='results'))
        
        # Asegurarse de que existan directorios necesarios
        self.create_required_directories()
        
        # Inicializar base de datos
        db = DatabaseHandler()
        db.initialize_database()
        
        return sm
    
    def create_required_directories(self):
        # Crear directorios necesarios para la aplicación
        directories = ['uploads', 'uploads/previews', 'results', 'results/images', 'results/reports', 'assets', 'outputs', 'data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Crear imágenes de placeholder si no existen
        placeholder_path = 'assets/placeholder.png'
        logo_path = 'assets/logo.png'
        
        if not os.path.exists(placeholder_path):
            # Crear una imagen simple como placeholder
            import numpy as np
            from PIL import Image as PILImage
            from PIL import ImageDraw, ImageFont
            
            img = PILImage.new('RGB', (400, 300), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 400, 300], outline=(200, 200, 200), width=2)
            d.text((150, 150), "No Image", fill=(100, 100, 100))
            img.save(placeholder_path)
        
        if not os.path.exists(logo_path):
            # Crear un logo simple
            img = PILImage.new('RGB', (400, 300), color=(200, 230, 255))
            d = ImageDraw.Draw(img)
            d.rectangle([50, 50, 350, 250], fill=(100, 150, 250))
            d.text((120, 150), "Electronic Analyzer", fill=(255, 255, 255))
            img.save(logo_path)

# Iniciar la aplicación si se ejecuta directamente
if __name__ == '__main__':
    ElectronicAnalyzerApp().run()