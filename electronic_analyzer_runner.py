import os
import sys
import subprocess
import time

def main():
    print("Iniciando Analizador de Electrónica...")
    
    # Crear directorios necesarios
    directories = ['templates', 'static', 'static/previews', 'uploads', 'outputs', 'results', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directorio creado: {directory}")
    
    # Variable global para el proceso Flask
    global flask_process
    flask_process = None
    
    # Ejecutar la aplicación Flask
    try:
        # Primero terminamos cualquier proceso que esté usando el puerto 5002
        try:
            subprocess.run("kill -9 $(lsof -t -i:5002)", shell=True)
            print("Puerto 5002 liberado.")
        except:
            pass
        
        # Ahora ejecutamos la aplicación Flask
        print("Ejecutando aplicación web...")
        flask_process = subprocess.Popen(["python", "web_app.py"], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE)
        
        # Esperar un momento para que la aplicación se inicie
        time.sleep(2)
        
        # Verificar si el proceso sigue en ejecución
        if flask_process.poll() is None:
            print("¡La aplicación se ha iniciado correctamente!")
            print("Accede a la aplicación en el navegador en: http://localhost:5002")
            print("O añadiendo ':5002' a la URL de Replit.")
            print("\nPresiona Ctrl+C para detener la aplicación.")
            
            # Mantener el script en ejecución
            while True:
                time.sleep(1)
        else:
            stdout, stderr = flask_process.communicate()
            print(f"Error al iniciar la aplicación: {stderr.decode()}")
    
    except KeyboardInterrupt:
        print("\nDeteniendo la aplicación...")
        if flask_process:
            flask_process.terminate()
        print("Aplicación detenida.")
    
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()