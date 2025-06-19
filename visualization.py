"""
Módulo de visualización para resultados de análisis
"""
import os
import json
import time
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import io
import base64

# Configurar el sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization')

class ResultVisualizer:
    def __init__(self):
        """
        Inicializa el visualizador de resultados.
        """
        self.component_colors = {
            'resistor': (52, 152, 219),     # Azul
            'capacitor': (46, 204, 113),    # Verde
            'inductor': (155, 89, 182),     # Púrpura
            'diode': (231, 76, 60),         # Rojo
            'transistor': (243, 156, 18),   # Naranja
            'led': (241, 196, 15),          # Amarillo
            'ic': (26, 188, 156),           # Turquesa
            'connector': (127, 140, 141),   # Gris
            'switch': (230, 126, 34),       # Naranja oscuro
            'unknown': (189, 195, 199)      # Gris claro
        }
        
        # Configurar colores para matplotlib (RGB normalizado)
        self.mpl_colors = {
            'resistor': (0.204, 0.596, 0.859),
            'capacitor': (0.180, 0.800, 0.443),
            'inductor': (0.608, 0.349, 0.714),
            'diode': (0.906, 0.298, 0.235),
            'transistor': (0.953, 0.612, 0.071),
            'led': (0.945, 0.769, 0.059),
            'ic': (0.102, 0.737, 0.612),
            'connector': (0.498, 0.549, 0.553),
            'switch': (0.902, 0.494, 0.133),
            'unknown': (0.741, 0.765, 0.780)
        }
        
        # Crear directorios de salida si no existen
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('static', exist_ok=True)

    def visualize_results(self, results, original_image_path):
        """
        Visualiza los resultados del análisis en una imagen con anotaciones.
        
        Args:
            results (dict): Resultados del análisis
            original_image_path (str): Ruta a la imagen original
            
        Returns:
            str: Ruta a la imagen con anotaciones
        """
        try:
            # Cargar imagen original
            image = cv2.imread(original_image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen original: {original_image_path}")
                return None
                
            # Crear una copia para anotaciones
            annotated_image = image.copy()
            
            # Dibujar componentes detectados
            if 'components' in results:
                for component in results['components']:
                    # Obtener valores
                    comp_id = component['id']
                    comp_type = component['type']
                    confidence = component['confidence']
                    bbox = component['bbox']
                    
                    # Obtener color del componente
                    color = self.component_colors.get(comp_type, self.component_colors['unknown'])
                    # Convertir de RGB a BGR para OpenCV
                    color = (color[2], color[1], color[0])
                    
                    # Dibujar rectángulo
                    cv2.rectangle(annotated_image, 
                                 (bbox[0], bbox[1]), 
                                 (bbox[2], bbox[3]), 
                                 color, 2)
                    
                    # Añadir etiqueta
                    label = f"#{comp_id}: {comp_type} ({confidence:.2f})"
                    cv2.putText(annotated_image, label, 
                               (bbox[0], bbox[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                               
            # Dibujar conexiones
            if 'connections' in results and results['connections']:
                self._draw_connections(annotated_image, results['components'], results['connections'])
                
            # Guardar imagen anotada
            timestamp = int(time.time())
            output_path = f'static/annotated_{os.path.basename(original_image_path)}_{timestamp}.jpg'
            cv2.imwrite(output_path, annotated_image)
            
            # Crear imagen para la web (ruta relativa)
            web_path = output_path
            logger.info(f"Imagen con anotaciones guardada en {output_path}")
            
            return web_path
            
        except Exception as e:
            logger.error(f"Error al visualizar resultados: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def _draw_connections(self, image, components, connections):
        """
        Dibuja conexiones entre componentes.
        
        Args:
            image: Imagen donde dibujar
            components: Lista de componentes
            connections: Lista de conexiones
        """
        # Crear mapa de componentes por ID
        component_map = {comp['id']: comp for comp in components}
        
        # Dibujar líneas para cada conexión
        for conn in connections:
            from_id = conn['from_component']
            to_id = conn['to_component']
            conn_type = conn['type']
            
            # Verificar que ambos componentes existen
            if from_id not in component_map or to_id not in component_map:
                continue
                
            # Obtener centros de los componentes
            from_comp = component_map[from_id]
            to_comp = component_map[to_id]
            
            from_x = (from_comp['bbox'][0] + from_comp['bbox'][2]) // 2
            from_y = (from_comp['bbox'][1] + from_comp['bbox'][3]) // 2
            
            to_x = (to_comp['bbox'][0] + to_comp['bbox'][2]) // 2
            to_y = (to_comp['bbox'][1] + to_comp['bbox'][3]) // 2
            
            # Determinar color y grosor según tipo de conexión
            if conn_type == 'power':
                color = (0, 0, 255)  # Rojo
                thickness = 2
            elif conn_type == 'ground':
                color = (0, 0, 0)    # Negro
                thickness = 2
            else:
                color = (255, 0, 255)  # Magenta
                thickness = 1
                
            # Dibujar línea
            cv2.line(image, (from_x, from_y), (to_x, to_y), color, thickness)
            
            # Añadir etiqueta en el punto medio
            mid_x = (from_x + to_x) // 2
            mid_y = (from_y + to_y) // 2
            cv2.putText(image, conn_type, (mid_x, mid_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def generate_component_plot(self, results, original_file_name, dark_mode=False):
        """
        Genera un gráfico de distribución de componentes.
        
        Args:
            results (dict): Resultados del análisis
            original_file_name (str): Nombre del archivo original
            dark_mode (bool): Si se debe usar modo oscuro para el gráfico
            
        Returns:
            str: Ruta al gráfico generado
        """
        try:
            if 'components' not in results or not results['components']:
                logger.warning("No hay componentes para generar gráfico")
                return None
                
            # Contar componentes por tipo
            component_counts = {}
            for comp in results['components']:
                comp_type = comp['type']
                component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
                
            # Preparar datos para el gráfico
            labels = list(component_counts.keys())
            sizes = list(component_counts.values())
            colors = [self.mpl_colors.get(label, self.mpl_colors['unknown']) for label in labels]
            
            # Configurar estilo según modo
            if dark_mode:
                plt.style.use('dark_background')
                text_color = 'white'
            else:
                plt.style.use('default')
                text_color = 'black'
                
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Crear gráfico de pastel
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Personalizar apariencia
            for text in texts:
                text.set_color(text_color)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                
            ax.axis('equal')  # Equal aspect ratio
            plt.title('Distribución de Componentes', color=text_color, fontsize=14)
            
            # Guardar gráfico
            timestamp = int(time.time())
            output_path = f'static/component_dist_{timestamp}.png'
            plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error al generar gráfico de componentes: {e}")
            return None

    def generate_graph_visualization(self, results, original_file_name):
        """
        Genera una visualización del grafo de conexiones entre componentes.
        
        Args:
            results (dict): Resultados del análisis
            original_file_name (str): Nombre del archivo original
            
        Returns:
            str: Ruta a la visualización del grafo
        """
        try:
            if ('components' not in results or not results['components'] or
                'connections' not in results or not results['connections']):
                logger.warning("No hay suficientes datos para generar grafo")
                return None
                
            components = results['components']
            connections = results['connections']
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            
            # Preparar posiciones para los nodos (componentes)
            n_components = len(components)
            positions = {}
            
            # Posicionar en círculo
            radius = 5
            for i, comp in enumerate(components):
                angle = 2 * np.pi * i / n_components
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions[comp['id']] = (x, y)
                
            # Dibujar nodos
            for comp in components:
                comp_id = comp['id']
                comp_type = comp['type']
                color = self.mpl_colors.get(comp_type, self.mpl_colors['unknown'])
                
                # Dibujar círculo para el componente
                circle = plt.Circle(positions[comp_id], 0.5, color=color, alpha=0.7)
                ax.add_patch(circle)
                
                # Agregar etiqueta
                ax.text(positions[comp_id][0], positions[comp_id][1], f"#{comp_id}",
                       ha='center', va='center', color='white', fontweight='bold')
                
                # Añadir tipo debajo
                ax.text(positions[comp_id][0], positions[comp_id][1] - 0.8, comp_type,
                       ha='center', va='center', fontsize=8)
                
            # Dibujar conexiones
            for conn in connections:
                from_id = conn['from_component']
                to_id = conn['to_component']
                conn_type = conn['type']
                
                # Verificar que ambos componentes existen
                if from_id not in positions or to_id not in positions:
                    continue
                    
                # Determinar color según tipo
                if conn_type == 'power':
                    color = 'red'
                elif conn_type == 'ground':
                    color = 'black'
                else:
                    color = 'blue'
                    
                # Dibujar línea
                ax.plot([positions[from_id][0], positions[to_id][0]],
                        [positions[from_id][1], positions[to_id][1]],
                        color=color, linestyle='-', linewidth=1.5, alpha=0.6)
                
                # Calcular punto medio
                mid_x = (positions[from_id][0] + positions[to_id][0]) / 2
                mid_y = (positions[from_id][1] + positions[to_id][1]) / 2
                
                # Añadir etiqueta de conexión
                ax.text(mid_x, mid_y, conn_type, ha='center', va='center',
                       fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
                
            # Configurar apariencia
            ax.set_xlim(-radius-2, radius+2)
            ax.set_ylim(-radius-2, radius+2)
            ax.set_aspect('equal')
            ax.axis('off')
            
            plt.title('Grafo de Conexiones entre Componentes', fontsize=14)
            
            # Guardar visualización
            timestamp = int(time.time())
            output_path = f'static/connection_graph_{timestamp}.png'
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error al generar visualización de grafo: {e}")
            return None

    def export_results(self, results, original_file_name):
        """
        Exporta los resultados del análisis a varios formatos.
        
        Args:
            results (dict): Resultados del análisis
            original_file_name (str): Nombre del archivo original
            
        Returns:
            dict: Rutas a los archivos exportados
        """
        exports = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(original_file_name)[0]
        
        try:
            # Exportar a JSON
            json_path = f'outputs/{base_name}_analysis_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            exports['json'] = json_path
            
            # Exportar a CSV
            if 'components' in results and results['components']:
                csv_path = f'outputs/{base_name}_components_{timestamp}.csv'
                with open(csv_path, 'w', encoding='utf-8') as f:
                    # Encabezados
                    header_fields = ['id', 'type', 'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'area']
                    f.write(','.join(header_fields) + '\n')
                    
                    # Datos de cada componente
                    for comp in results['components']:
                        row = [
                            str(comp['id']),
                            comp['type'],
                            f"{comp['confidence']:.3f}",
                            str(comp['bbox'][0]),
                            str(comp['bbox'][1]),
                            str(comp['bbox'][2]),
                            str(comp['bbox'][3]),
                            str(comp['dimensions'][0]),
                            str(comp['dimensions'][1]),
                            f"{comp['area']:.1f}"
                        ]
                        f.write(','.join(row) + '\n')
                exports['csv'] = csv_path
            
            # Generar gráficos y visualizaciones
            component_plot = self.generate_component_plot(results, original_file_name)
            if component_plot:
                exports['component_plot'] = component_plot
                
            graph_viz = self.generate_graph_visualization(results, original_file_name)
            if graph_viz:
                exports['graph'] = graph_viz
                
            return exports
            
        except Exception as e:
            logger.error(f"Error al exportar resultados: {e}")
            return exports

    def generate_html_report(self, results, original_file_name, original_image_path=None):
        """
        Genera un informe HTML con los resultados del análisis.
        
        Args:
            results (dict): Resultados del análisis
            original_file_name (str): Nombre del archivo original
            original_image_path (str, optional): Ruta a la imagen original
            
        Returns:
            str: Ruta al archivo HTML generado
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(original_file_name)[0]
            report_path = f'outputs/{base_name}_report_{timestamp}.html'
            
            # Preparar visualizaciones
            visualizations = {}
            
            if original_image_path:
                visualizations['annotated'] = self.visualize_results(results, original_image_path)
                
            visualizations['component_plot'] = self.generate_component_plot(results, original_file_name)
            visualizations['connection_graph'] = self.generate_graph_visualization(results, original_file_name)
            
            # Crear HTML
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="es">\n')
                f.write('<head>\n')
                f.write('    <meta charset="UTF-8">\n')
                f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                f.write(f'    <title>Análisis de {original_file_name}</title>\n')
                f.write('    <style>\n')
                f.write('        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }\n')
                f.write('        .container { max-width: 1200px; margin: 0 auto; }\n')
                f.write('        h1, h2, h3 { color: #2c3e50; }\n')
                f.write('        .header { background-color: #3498db; color: white; padding: 20px; margin-bottom: 30px; border-radius: 5px; }\n')
                f.write('        .section { margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n')
                f.write('        table { width: 100%; border-collapse: collapse; margin: 20px 0; }\n')
                f.write('        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }\n')
                f.write('        th { background-color: #f2f2f2; }\n')
                f.write('        tr:hover { background-color: #f5f5f5; }\n')
                f.write('        .component-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }\n')
                f.write('        .component-card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }\n')
                f.write('        .component-card h3 { margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; }\n')
                f.write('        .viz-container { margin: 20px 0; text-align: center; }\n')
                f.write('        .viz-container img { max-width: 100%; border-radius: 5px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }\n')
                f.write('        .footer { margin-top: 50px; text-align: center; font-size: 0.9em; color: #7f8c8d; }\n')
                f.write('    </style>\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write('    <div class="container">\n')
                
                # Encabezado
                f.write('        <div class="header">\n')
                f.write(f'            <h1>Análisis de Componentes Electrónicos</h1>\n')
                f.write(f'            <p>Archivo: {original_file_name}</p>\n')
                f.write(f'            <p>Fecha de análisis: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>\n')
                f.write('        </div>\n')
                
                # Resumen
                f.write('        <div class="section">\n')
                f.write('            <h2>Resumen del Análisis</h2>\n')
                f.write('            <table>\n')
                f.write('                <tr><th>Componentes detectados</th><td>' + str(results.get('component_count', 0)) + '</td></tr>\n')
                f.write('                <tr><th>Conexiones detectadas</th><td>' + str(results.get('connection_count', 0)) + '</td></tr>\n')
                
                if 'circuit_analysis' in results and results['circuit_analysis']:
                    circuit = results['circuit_analysis']
                    f.write('                <tr><th>Tipo de circuito</th><td>' + circuit.get('type', 'Desconocido') + '</td></tr>\n')
                    f.write('                <tr><th>Complejidad</th><td>' + circuit.get('complexity', 'Simple') + '</td></tr>\n')
                    f.write('                <tr><th>Potencia estimada</th><td>' + circuit.get('estimated_power', 'Baja') + '</td></tr>\n')
                
                f.write('                <tr><th>Tiempo de análisis</th><td>' + str(results.get('analysis_time', 0)) + ' segundos</td></tr>\n')
                f.write('            </table>\n')
                f.write('        </div>\n')
                
                # Visualizaciones
                if visualizations:
                    f.write('        <div class="section">\n')
                    f.write('            <h2>Visualizaciones</h2>\n')
                    
                    if 'annotated' in visualizations and visualizations['annotated']:
                        f.write('            <div class="viz-container">\n')
                        f.write('                <h3>Componentes Detectados</h3>\n')
                        f.write(f'                <img src="/{visualizations["annotated"]}" alt="Componentes detectados">\n')
                        f.write('            </div>\n')
                        
                    if 'component_plot' in visualizations and visualizations['component_plot']:
                        f.write('            <div class="viz-container">\n')
                        f.write('                <h3>Distribución de Componentes</h3>\n')
                        f.write(f'                <img src="/{visualizations["component_plot"]}" alt="Distribución de componentes">\n')
                        f.write('            </div>\n')
                        
                    if 'connection_graph' in visualizations and visualizations['connection_graph']:
                        f.write('            <div class="viz-container">\n')
                        f.write('                <h3>Grafo de Conexiones</h3>\n')
                        f.write(f'                <img src="/{visualizations["connection_graph"]}" alt="Grafo de conexiones">\n')
                        f.write('            </div>\n')
                        
                    f.write('        </div>\n')
                
                # Componentes detectados
                if 'components' in results and results['components']:
                    f.write('        <div class="section">\n')
                    f.write('            <h2>Componentes Detectados</h2>\n')
                    f.write('            <div class="component-grid">\n')
                    
                    for comp in results['components']:
                        f.write('                <div class="component-card">\n')
                        f.write(f'                    <h3>#{comp["id"]}: {comp["type"].capitalize()}</h3>\n')
                        f.write(f'                    <p><strong>Confianza:</strong> {comp["confidence"]:.2f}</p>\n')
                        f.write(f'                    <p><strong>Posición:</strong> ({comp["bbox"][0]}, {comp["bbox"][1]}) - ({comp["bbox"][2]}, {comp["bbox"][3]})</p>\n')
                        f.write(f'                    <p><strong>Dimensiones:</strong> {comp["dimensions"][0]}x{comp["dimensions"][1]} px</p>\n')
                        
                        if 'specifications' in comp and comp['specifications']:
                            f.write('                    <h4>Especificaciones:</h4>\n')
                            f.write('                    <table>\n')
                            for key, value in comp['specifications'].items():
                                f.write(f'                        <tr><th>{key.replace("_", " ").capitalize()}</th><td>{value}</td></tr>\n')
                            f.write('                    </table>\n')
                            
                        f.write('                </div>\n')
                        
                    f.write('            </div>\n')
                    f.write('        </div>\n')
                
                # Conexiones
                if 'connections' in results and results['connections']:
                    f.write('        <div class="section">\n')
                    f.write('            <h2>Conexiones Detectadas</h2>\n')
                    f.write('            <table>\n')
                    f.write('                <tr><th>#</th><th>Desde</th><th>Hasta</th><th>Tipo</th></tr>\n')
                    
                    for i, conn in enumerate(results['connections'], 1):
                        f.write(f'                <tr><td>{i}</td><td>Componente #{conn["from_component"]}</td><td>Componente #{conn["to_component"]}</td><td>{conn["type"].capitalize()}</td></tr>\n')
                        
                    f.write('            </table>\n')
                    f.write('        </div>\n')
                
                # Pie de página
                f.write('        <div class="footer">\n')
                f.write('            <p>Generado por ElectroAnalyzer &copy; 2025</p>\n')
                f.write('        </div>\n')
                
                f.write('    </div>\n')
                f.write('</body>\n')
                f.write('</html>\n')
                
            logger.info(f"Informe HTML generado en {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error al generar informe HTML: {e}")
            return None
            
    def generate_matplotlib_figure(self, figure_type, results, output_format='png'):
        """
        Genera una figura de matplotlib según el tipo especificado.
        
        Args:
            figure_type (str): Tipo de figura ('component_dist', 'graph', etc.)
            results (dict): Resultados del análisis
            output_format (str): Formato de salida ('png', 'svg', etc.)
            
        Returns:
            bytes: Datos de la imagen
        """
        try:
            fig = None
            
            if figure_type == 'component_dist':
                if 'components' not in results or not results['components']:
                    return None
                    
                # Contar componentes por tipo
                component_counts = {}
                for comp in results['components']:
                    comp_type = comp['type']
                    component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
                    
                # Preparar datos para el gráfico
                labels = list(component_counts.keys())
                sizes = list(component_counts.values())
                colors = [self.mpl_colors.get(label, self.mpl_colors['unknown']) for label in labels]
                
                # Crear figura
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Crear gráfico de pastel
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels, 
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                
                # Personalizar apariencia
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    
                ax.axis('equal')
                plt.title('Distribución de Componentes')
                
            # TODO: Implementar otros tipos de figuras según necesidad
            
            if fig:
                # Guardar a un buffer en memoria
                buf = io.BytesIO()
                fig.savefig(buf, format=output_format, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                return buf.getvalue()
            return None
            
        except Exception as e:
            logger.error(f"Error al generar figura matplotlib: {e}")
            return None


if __name__ == "__main__":
    # Código de prueba
    visualizer = ResultVisualizer()
    
    # Datos de ejemplo para pruebas
    test_results = {
        "components": [
            {"id": 1, "type": "resistor", "confidence": 0.95, "bbox": [100, 100, 150, 120]},
            {"id": 2, "type": "capacitor", "confidence": 0.87, "bbox": [200, 150, 250, 200]},
            {"id": 3, "type": "transistor", "confidence": 0.76, "bbox": [300, 250, 350, 300]},
        ],
        "component_count": 3,
        "connections": [
            {"from_component": 1, "to_component": 2, "type": "signal"},
            {"from_component": 2, "to_component": 3, "type": "power"}
        ],
        "connection_count": 2
    }
    
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        result_image = visualizer.visualize_results(test_results, test_image)
        print(f"Imagen de resultados generada: {result_image}")
        
        report_path = visualizer.generate_html_report(test_results, "test_image.jpg", test_image)
        print(f"Informe HTML generado: {report_path}")
    else:
        print(f"No se encontró la imagen de prueba: {test_image}")