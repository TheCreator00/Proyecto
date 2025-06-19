"""
Aplicación Android del Analizador de Componentes Electrónicos
Versión Kivy para ser compilada con Buildozer
"""

import os
import sys
import time
import threading
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.properties import StringProperty, ObjectProperty, ListProperty
from kivy.utils import platform
from kivy.logger import Logger

# Importar módulos del analizador
try:
    from database_handler import DatabaseHandler
    from file_processor import FileProcessor
    from ai_analyzer import AIAnalyzer
    from visualization import ResultVisualizer
    import xai_integration
    
    # Indicador de que los módulos se cargaron correctamente
    MODULES_LOADED = True
except Exception as e:
    Logger.error(f"Error al importar módulos: {e}")
    MODULES_LOADED = False

# Configuración para permisos en Android
if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([
        Permission.INTERNET,
        Permission.CAMERA,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE
    ])

# Crear directorios necesarios
def create_app_directories():
    """Crea los directorios necesarios para la aplicación."""
    if platform == 'android':
        from android.storage import primary_external_storage_path
        base_path = os.path.join(primary_external_storage_path(), 'AnalizadorElectronico')
    else:
        base_path = '.'
        
    directories = [
        os.path.join(base_path, 'static'),
        os.path.join(base_path, 'static/previews'),
        os.path.join(base_path, 'uploads'),
        os.path.join(base_path, 'outputs'),
        os.path.join(base_path, 'results'),
        os.path.join(base_path, 'data')
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            Logger.info(f"Directory created: {directory}")

# Inicializar componentes de la aplicación
class AppComponents:
    """Clase para mantener los componentes principales de la aplicación."""
    def __init__(self):
        # Crear instancias solo si los módulos se cargaron correctamente
        if MODULES_LOADED:
            self.db_handler = DatabaseHandler()
            self.file_processor = FileProcessor()
            self.ai_analyzer = AIAnalyzer()
            self.visualizer = ResultVisualizer()
            self.xai_processor = xai_integration.XAIProcessor()
        else:
            self.db_handler = None
            self.file_processor = None
            self.ai_analyzer = None
            self.visualizer = None
            self.xai_processor = None

# Pantalla de Inicio
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Logo y título
        self.title_label = Label(
            text="Analizador de Componentes Electrónicos",
            font_size=24,
            size_hint_y=None,
            height=100
        )
        self.layout.add_widget(self.title_label)
        
        # Botones de navegación
        self.load_btn = Button(
            text="Cargar Archivo",
            size_hint_y=None,
            height=80,
            background_color=(0.2, 0.6, 0.8, 1.0)
        )
        self.load_btn.bind(on_press=self.go_to_load)
        self.layout.add_widget(self.load_btn)
        
        self.analyze_btn = Button(
            text="Analizar",
            size_hint_y=None,
            height=80,
            background_color=(0.2, 0.7, 0.2, 1.0)
        )
        self.analyze_btn.bind(on_press=self.go_to_analyze)
        self.layout.add_widget(self.analyze_btn)
        
        self.results_btn = Button(
            text="Ver Resultados",
            size_hint_y=None,
            height=80,
            background_color=(0.8, 0.3, 0.3, 1.0)
        )
        self.results_btn.bind(on_press=self.go_to_results)
        self.layout.add_widget(self.results_btn)
        
        # Versión
        self.version_label = Label(
            text="Versión 1.0.0",
            font_size=14,
            size_hint_y=None,
            height=40
        )
        self.layout.add_widget(self.version_label)
        
        # Agregar el layout a la pantalla
        self.add_widget(self.layout)
    
    def go_to_load(self, instance):
        self.manager.current = 'load'
    
    def go_to_analyze(self, instance):
        self.manager.current = 'analyze'
    
    def go_to_results(self, instance):
        self.manager.current = 'results'

# Pantalla de Carga
class LoadScreen(Screen):
    def __init__(self, app_components, **kwargs):
        super(LoadScreen, self).__init__(**kwargs)
        self.app_components = app_components
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Título
        self.title_label = Label(
            text="Cargar Archivo",
            font_size=22,
            size_hint_y=None,
            height=60
        )
        self.layout.add_widget(self.title_label)
        
        # Selector de archivo
        self.file_chooser = FileChooserListView(
            size_hint_y=0.7,
            filters=['*.jpg', '*.jpeg', '*.png', '*.pdf']
        )
        self.layout.add_widget(self.file_chooser)
        
        # Botón para cargar archivo
        self.load_btn = Button(
            text="Cargar Archivo Seleccionado",
            size_hint_y=None,
            height=60,
            background_color=(0.2, 0.6, 0.8, 1.0)
        )
        self.load_btn.bind(on_press=self.load_file)
        self.layout.add_widget(self.load_btn)
        
        # Botón para volver
        self.back_btn = Button(
            text="Volver",
            size_hint_y=None,
            height=50,
            background_color=(0.5, 0.5, 0.5, 1.0)
        )
        self.back_btn.bind(on_press=self.go_back)
        self.layout.add_widget(self.back_btn)
        
        # Agregar el layout a la pantalla
        self.add_widget(self.layout)
    
    def load_file(self, instance):
        if not self.file_chooser.selection:
            popup = Popup(
                title='Error',
                content=Label(text='Por favor, selecciona un archivo primero.'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
            return
        
        file_path = self.file_chooser.selection[0]
        
        try:
            # Procesar el archivo
            file_info = self.app_components.file_processor.process_file(file_path)
            
            # Guardar información en la base de datos
            self.app_components.db_handler.save_file_info(file_info)
            
            # Mostrar confirmación
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            content.add_widget(Label(text=f'Archivo cargado: {file_info["name"]}'))
            
            analyze_btn = Button(text='Analizar Ahora')
            analyze_btn.bind(on_press=lambda btn: self.analyze_now(popup, file_info['id']))
            content.add_widget(analyze_btn)
            
            later_btn = Button(text='Más Tarde')
            later_btn.bind(on_press=lambda btn: popup.dismiss())
            content.add_widget(later_btn)
            
            popup = Popup(
                title='Archivo Cargado',
                content=content,
                size_hint=(0.8, 0.4)
            )
            popup.open()
            
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(text=f'Error al procesar el archivo: {str(e)}'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
    
    def analyze_now(self, popup, file_id):
        popup.dismiss()
        
        # Pasar a la pantalla de análisis con el ID del archivo
        analyze_screen = self.manager.get_screen('analyze')
        analyze_screen.select_file(file_id, '')
        self.manager.current = 'analyze'
    
    def go_back(self, instance):
        self.manager.current = 'home'

# Pantalla de Análisis
class AnalyzeScreen(Screen):
    def __init__(self, app_components, **kwargs):
        super(AnalyzeScreen, self).__init__(**kwargs)
        self.app_components = app_components
        self.selected_file_id = None
        self.selected_file_name = ''
        
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Título
        self.title_label = Label(
            text="Analizar Archivo",
            font_size=22,
            size_hint_y=None,
            height=60
        )
        self.layout.add_widget(self.title_label)
        
        # Selección de archivo
        self.file_selection = BoxLayout(orientation='horizontal', size_hint_y=None, height=60)
        self.file_label = Label(text="Archivo: Ninguno seleccionado")
        self.file_selection.add_widget(self.file_label)
        
        self.select_btn = Button(
            text="Seleccionar Archivo",
            size_hint_x=0.4,
            background_color=(0.2, 0.6, 0.8, 1.0)
        )
        self.select_btn.bind(on_press=self.show_file_selection)
        self.file_selection.add_widget(self.select_btn)
        
        self.layout.add_widget(self.file_selection)
        
        # Opciones de análisis
        self.options_label = Label(
            text="Opciones de Análisis:",
            font_size=18,
            size_hint_y=None,
            height=40,
            halign='left'
        )
        self.options_label.bind(size=self.options_label.setter('text_size'))
        self.layout.add_widget(self.options_label)
        
        # Spinner para seleccionar el modelo
        self.model_selection = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        self.model_selection.add_widget(Label(text="Modelo de AI:"))
        
        self.model_spinner = Spinner(
            text='Automático',
            values=('Automático', 'Grok (xAI)', 'Local', 'Híbrido'),
            size_hint_x=0.7
        )
        self.model_selection.add_widget(self.model_spinner)
        
        self.layout.add_widget(self.model_selection)
        
        # Botón de análisis
        self.analyze_btn = Button(
            text="Iniciar Análisis",
            size_hint_y=None,
            height=70,
            background_color=(0.2, 0.7, 0.2, 1.0)
        )
        self.analyze_btn.bind(on_press=self.start_analysis)
        self.layout.add_widget(self.analyze_btn)
        
        # Estado del análisis
        self.status_label = Label(
            text="",
            size_hint_y=None,
            height=40
        )
        self.layout.add_widget(self.status_label)
        
        # Botón para volver
        self.back_btn = Button(
            text="Volver",
            size_hint_y=None,
            height=50,
            background_color=(0.5, 0.5, 0.5, 1.0)
        )
        self.back_btn.bind(on_press=self.go_back)
        self.layout.add_widget(self.back_btn)
        
        # Agregar el layout a la pantalla
        self.add_widget(self.layout)
    
    def on_enter(self):
        # Actualizar la lista de archivos al entrar a la pantalla
        if self.app_components.db_handler:
            self.files = self.app_components.db_handler.get_all_files()
    
    def show_file_selection(self, instance):
        # Mostrar popup con lista de archivos
        if not self.app_components.db_handler:
            popup = Popup(
                title='Error',
                content=Label(text='El sistema de base de datos no está disponible.'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
            return
            
        try:
            files = self.app_components.db_handler.get_all_files()
            
            if not files:
                popup = Popup(
                    title='Sin Archivos',
                    content=Label(text='No hay archivos disponibles. Por favor, carga un archivo primero.'),
                    size_hint=(0.7, 0.3)
                )
                popup.open()
                return
            
            # Crear layout para la lista de archivos
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            scroll_view = ScrollView(size_hint=(1, 0.8))
            files_layout = GridLayout(cols=1, spacing=5, size_hint_y=None)
            files_layout.bind(minimum_height=files_layout.setter('height'))
            
            for file in files:
                file_btn = Button(
                    text=file['name'],
                    size_hint_y=None,
                    height=50
                )
                file_btn.bind(on_press=lambda btn, fid=file['id'], fname=file['name']: 
                    self.select_file(fid, fname))
                files_layout.add_widget(file_btn)
            
            scroll_view.add_widget(files_layout)
            content.add_widget(scroll_view)
            
            # Botón para cerrar
            close_btn = Button(
                text='Cerrar',
                size_hint_y=None,
                height=50
            )
            close_btn.bind(on_press=lambda btn: popup.dismiss())
            content.add_widget(close_btn)
            
            popup = Popup(
                title='Seleccionar Archivo',
                content=content,
                size_hint=(0.9, 0.8)
            )
            popup.open()
            
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(text=f'Error al cargar la lista de archivos: {str(e)}'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
    
    def select_file(self, file_id, file_name):
        self.selected_file_id = file_id
        
        if not file_name:
            # Obtener el nombre del archivo de la base de datos
            try:
                file_info = self.app_components.db_handler.get_file_by_id(file_id)
                if file_info:
                    self.selected_file_name = file_info['name']
                else:
                    self.selected_file_name = f"Archivo ID: {file_id}"
            except:
                self.selected_file_name = f"Archivo ID: {file_id}"
        else:
            self.selected_file_name = file_name
        
        # Actualizar la etiqueta
        self.file_label.text = f"Archivo: {self.selected_file_name}"
    
    def start_analysis(self, instance):
        if not self.selected_file_id:
            popup = Popup(
                title='Error',
                content=Label(text='Por favor, selecciona un archivo primero.'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
            return
        
        # Actualizar estado
        self.status_label.text = "Analizando..."
        self.analyze_btn.disabled = True
        
        # Obtener el modelo seleccionado
        model_selected = self.model_spinner.text
        model_type = 'auto'
        if model_selected == 'Grok (xAI)':
            model_type = 'grok'
        elif model_selected == 'Local':
            model_type = 'local'
        elif model_selected == 'Híbrido':
            model_type = 'hybrid'
        
        # Iniciar análisis en un hilo separado
        threading.Thread(target=self._run_analysis, args=(self.selected_file_id, model_type)).start()
    
    def _run_analysis(self, file_id, model_type):
        try:
            # Obtener información del archivo
            file_info = self.app_components.db_handler.get_file_by_id(file_id)
            
            if not file_info:
                Clock.schedule_once(lambda dt: self._show_error("No se encontró el archivo seleccionado."))
                return
            
            # Analizar el archivo
            if file_info['type'] == 'image':
                results = self.app_components.ai_analyzer.analyze_image(file_info['path'])
            else:
                results = self.app_components.ai_analyzer.analyze_file(file_info['path'])
            
            # Guardar resultados en la base de datos
            self.app_components.db_handler.save_analysis_results(file_id, results)
            
            # Actualizar UI en el hilo principal
            Clock.schedule_once(lambda dt: self._analysis_complete(file_id))
            
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_error(f"Error durante el análisis: {str(e)}"))
    
    def _analysis_complete(self, file_id):
        self.status_label.text = "Análisis completado"
        self.analyze_btn.disabled = False
        
        # Mostrar popup con opciones
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text='El análisis se ha completado correctamente.'))
        
        view_btn = Button(text='Ver Resultados')
        view_btn.bind(on_press=lambda btn: self.view_results_now(popup, file_id))
        content.add_widget(view_btn)
        
        later_btn = Button(text='Más Tarde')
        later_btn.bind(on_press=lambda btn: popup.dismiss())
        content.add_widget(later_btn)
        
        popup = Popup(
            title='Análisis Completado',
            content=content,
            size_hint=(0.8, 0.4)
        )
        popup.open()
    
    def _show_error(self, message):
        self.status_label.text = "Error en el análisis"
        self.analyze_btn.disabled = False
        
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.7, 0.3)
        )
        popup.open()
    
    def view_results_now(self, popup, file_id):
        popup.dismiss()
        
        # Pasar a la pantalla de resultados
        results_screen = self.manager.get_screen('results')
        results_screen.load_results(file_id)
        self.manager.current = 'results'
    
    def go_back(self, instance):
        self.manager.current = 'home'

# Pantalla de Resultados
class ResultsScreen(Screen):
    def __init__(self, app_components, **kwargs):
        super(ResultsScreen, self).__init__(**kwargs)
        self.app_components = app_components
        self.current_file_id = None
        
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Título
        self.title_label = Label(
            text="Resultados de Análisis",
            font_size=22,
            size_hint_y=None,
            height=60
        )
        self.layout.add_widget(self.title_label)
        
        # Selección de archivo
        self.file_selection = BoxLayout(orientation='horizontal', size_hint_y=None, height=60)
        self.file_label = Label(text="Archivo: Ninguno seleccionado")
        self.file_selection.add_widget(self.file_label)
        
        self.select_btn = Button(
            text="Seleccionar Archivo",
            size_hint_x=0.4,
            background_color=(0.2, 0.6, 0.8, 1.0)
        )
        self.select_btn.bind(on_press=self.show_file_selection)
        self.file_selection.add_widget(self.select_btn)
        
        self.layout.add_widget(self.file_selection)
        
        # Área de resultados (ScrollView)
        self.scroll_view = ScrollView(size_hint=(1, 0.7))
        self.results_layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))
        
        # Inicialmente mostrar un mensaje
        self.results_layout.add_widget(Label(
            text="Selecciona un archivo para ver sus resultados.",
            size_hint_y=None,
            height=100
        ))
        
        self.scroll_view.add_widget(self.results_layout)
        self.layout.add_widget(self.scroll_view)
        
        # Botones de acción
        self.action_buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=60, spacing=10)
        
        self.details_btn = Button(
            text="Ver Detalles",
            background_color=(0.2, 0.7, 0.2, 1.0)
        )
        self.details_btn.bind(on_press=self.show_details)
        self.action_buttons.add_widget(self.details_btn)
        
        self.export_btn = Button(
            text="Exportar",
            background_color=(0.7, 0.5, 0.2, 1.0)
        )
        self.export_btn.bind(on_press=self.export_results)
        self.action_buttons.add_widget(self.export_btn)
        
        self.layout.add_widget(self.action_buttons)
        
        # Botón para volver
        self.back_btn = Button(
            text="Volver",
            size_hint_y=None,
            height=50,
            background_color=(0.5, 0.5, 0.5, 1.0)
        )
        self.back_btn.bind(on_press=self.go_back)
        self.layout.add_widget(self.back_btn)
        
        # Agregar el layout a la pantalla
        self.add_widget(self.layout)
    
    def on_enter(self):
        # Actualizar la lista de archivos al entrar a la pantalla
        pass
    
    def show_file_selection(self, instance):
        # Mostrar popup con lista de archivos analizados
        if not self.app_components.db_handler:
            popup = Popup(
                title='Error',
                content=Label(text='El sistema de base de datos no está disponible.'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
            return
            
        try:
            files = self.app_components.db_handler.get_analyzed_files()
            
            if not files:
                popup = Popup(
                    title='Sin Resultados',
                    content=Label(text='No hay archivos analizados disponibles. Por favor, analiza un archivo primero.'),
                    size_hint=(0.7, 0.3)
                )
                popup.open()
                return
            
            # Crear layout para la lista de archivos
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            scroll_view = ScrollView(size_hint=(1, 0.8))
            files_layout = GridLayout(cols=1, spacing=5, size_hint_y=None)
            files_layout.bind(minimum_height=files_layout.setter('height'))
            
            for file in files:
                file_btn = Button(
                    text=file['name'],
                    size_hint_y=None,
                    height=50
                )
                file_btn.bind(on_press=lambda btn, fid=file['id'], fname=file['name']: 
                    self.select_file(fid, fname))
                files_layout.add_widget(file_btn)
            
            scroll_view.add_widget(files_layout)
            content.add_widget(scroll_view)
            
            # Botón para cerrar
            close_btn = Button(
                text='Cerrar',
                size_hint_y=None,
                height=50
            )
            close_btn.bind(on_press=lambda btn: popup.dismiss())
            content.add_widget(close_btn)
            
            popup = Popup(
                title='Seleccionar Archivo Analizado',
                content=content,
                size_hint=(0.9, 0.8)
            )
            popup.open()
            
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(text=f'Error al cargar la lista de archivos: {str(e)}'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
    
    def select_file(self, file_id, file_name):
        # Actualizar selección y cargar resultados
        self.current_file_id = file_id
        self.file_label.text = f"Archivo: {file_name}"
        
        # Cargar y mostrar los resultados
        self.load_results(file_id)
    
    def load_results(self, file_id):
        # Limpiar resultados anteriores
        self.results_layout.clear_widgets()
        self.current_file_id = file_id
        
        try:
            # Obtener información del archivo
            file_info = self.app_components.db_handler.get_file_by_id(file_id)
            if not file_info:
                self.results_layout.add_widget(Label(
                    text="No se encontró información del archivo.",
                    size_hint_y=None,
                    height=100
                ))
                return
            
            # Actualizar etiqueta de archivo
            self.file_label.text = f"Archivo: {file_info['name']}"
            
            # Obtener resultados del análisis
            results = self.app_components.db_handler.get_analysis_results(file_id)
            if not results:
                self.results_layout.add_widget(Label(
                    text="No hay resultados disponibles para este archivo.",
                    size_hint_y=None,
                    height=100
                ))
                return
            
            # Mostrar vista previa si está disponible
            if 'preview_path' in file_info and file_info['preview_path']:
                preview = Image(
                    source=file_info['preview_path'],
                    size_hint=(1, None),
                    height=200
                )
                self.results_layout.add_widget(preview)
            
            # Mostrar resumen de resultados
            summary_label = Label(
                text="Resumen de Análisis:",
                font_size=18,
                size_hint_y=None,
                height=40,
                halign='left'
            )
            summary_label.bind(size=summary_label.setter('text_size'))
            self.results_layout.add_widget(summary_label)
            
            # Mostrar componentes detectados
            if 'components' in results and results['components']:
                components_count = len(results['components'])
                components_label = Label(
                    text=f"Componentes detectados: {components_count}",
                    size_hint_y=None,
                    height=30,
                    halign='left'
                )
                components_label.bind(size=components_label.setter('text_size'))
                self.results_layout.add_widget(components_label)
                
                # Listar componentes
                for i, component in enumerate(results['components']):
                    component_box = BoxLayout(
                        orientation='vertical',
                        size_hint_y=None,
                        height=80,
                        padding=5
                    )
                    
                    comp_name = component.get('type', 'Desconocido')
                    comp_conf = component.get('confidence', 0) * 100
                    
                    component_box.add_widget(Label(
                        text=f"{i+1}. {comp_name} (Confianza: {comp_conf:.1f}%)",
                        halign='left',
                        valign='top',
                        size_hint_y=None,
                        height=30
                    ))
                    
                    # Especificaciones
                    if 'specifications' in component:
                        specs = []
                        for key, value in component['specifications'].items():
                            specs.append(f"{key}: {value}")
                        
                        specs_text = ", ".join(specs)
                        specs_label = Label(
                            text=specs_text,
                            halign='left',
                            size_hint_y=None,
                            height=50,
                            font_size=14
                        )
                        specs_label.bind(size=specs_label.setter('text_size'))
                        component_box.add_widget(specs_label)
                    
                    self.results_layout.add_widget(component_box)
            
            # Mostrar análisis del circuito
            if 'circuit_analysis' in results and results['circuit_analysis']:
                circuit_label = Label(
                    text="Análisis del Circuito:",
                    font_size=18,
                    size_hint_y=None,
                    height=40,
                    halign='left'
                )
                circuit_label.bind(size=circuit_label.setter('text_size'))
                self.results_layout.add_widget(circuit_label)
                
                circuit_type = results['circuit_analysis'].get('type', 'Desconocido')
                circuit_desc = results['circuit_analysis'].get('description', '')
                
                circuit_info = Label(
                    text=f"Tipo: {circuit_type}\n{circuit_desc}",
                    halign='left',
                    size_hint_y=None,
                    height=100
                )
                circuit_info.bind(size=circuit_info.setter('text_size'))
                self.results_layout.add_widget(circuit_info)
        
        except Exception as e:
            error_label = Label(
                text=f"Error al cargar resultados: {str(e)}",
                size_hint_y=None,
                height=100
            )
            self.results_layout.add_widget(error_label)
    
    def show_details(self, instance):
        if not self.current_file_id:
            popup = Popup(
                title='Error',
                content=Label(text='Por favor, selecciona un archivo primero.'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
            return
        
        try:
            # Obtener resultados del análisis
            results = self.app_components.db_handler.get_analysis_results(self.current_file_id)
            file_info = self.app_components.db_handler.get_file_by_id(self.current_file_id)
            
            if not results or not file_info:
                popup = Popup(
                    title='Error',
                    content=Label(text='No se encontraron datos para este archivo.'),
                    size_hint=(0.7, 0.3)
                )
                popup.open()
                return
            
            # Generar reporte HTML
            report_path = self.app_components.visualizer.generate_html_report(
                results, file_info['name'], file_info.get('path', None))
            
            # Mostrar mensaje con la ubicación del reporte
            popup = Popup(
                title='Detalles Generados',
                content=Label(text=f'Se ha generado un reporte detallado en:\n{report_path}'),
                size_hint=(0.8, 0.4)
            )
            popup.open()
            
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(text=f'Error al generar detalles: {str(e)}'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
    
    def export_results(self, instance):
        if not self.current_file_id:
            popup = Popup(
                title='Error',
                content=Label(text='Por favor, selecciona un archivo primero.'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
            return
        
        try:
            # Obtener resultados del análisis
            results = self.app_components.db_handler.get_analysis_results(self.current_file_id)
            file_info = self.app_components.db_handler.get_file_by_id(self.current_file_id)
            
            if not results or not file_info:
                popup = Popup(
                    title='Error',
                    content=Label(text='No se encontraron datos para este archivo.'),
                    size_hint=(0.7, 0.3)
                )
                popup.open()
                return
            
            # Exportar resultados
            export_paths = self.app_components.visualizer.export_results(results, file_info['name'])
            
            # Mostrar mensaje con las ubicaciones de los archivos exportados
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            content.add_widget(Label(text='Resultados exportados a:'))
            
            for format_name, path in export_paths.items():
                content.add_widget(Label(
                    text=f"{format_name}: {path}",
                    font_size=14,
                    halign='left'
                ))
            
            close_btn = Button(
                text='Cerrar',
                size_hint_y=None,
                height=50
            )
            content.add_widget(close_btn)
            
            popup = Popup(
                title='Exportación Completada',
                content=content,
                size_hint=(0.8, 0.6)
            )
            
            close_btn.bind(on_press=lambda btn: popup.dismiss())
            popup.open()
            
        except Exception as e:
            popup = Popup(
                title='Error',
                content=Label(text=f'Error al exportar resultados: {str(e)}'),
                size_hint=(0.7, 0.3)
            )
            popup.open()
    
    def go_back(self, instance):
        self.manager.current = 'home'

# Aplicación principal
class ElectronicAnalyzerApp(App):
    def build(self):
        # Crear directorios necesarios
        create_app_directories()
        
        # Inicializar componentes
        self.app_components = AppComponents()
        
        # Crear administrador de pantallas
        sm = ScreenManager()
        
        # Añadir pantallas
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(LoadScreen(self.app_components, name='load'))
        sm.add_widget(AnalyzeScreen(self.app_components, name='analyze'))
        sm.add_widget(ResultsScreen(self.app_components, name='results'))
        
        return sm

if __name__ == '__main__':
    ElectronicAnalyzerApp().run()