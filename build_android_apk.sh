#!/bin/bash

echo "=== Compilando APK del Analizador de Componentes Electrónicos ==="
echo "Asegúrate de ejecutar este script en un entorno Linux (preferiblemente Ubuntu)"
echo ""

# Verificar si Buildozer está instalado
if ! command -v buildozer &> /dev/null; then
    echo "Buildozer no está instalado. Instalando..."
    
    # Actualizar repositorios e instalar dependencias
    sudo apt update
    sudo apt install -y python3 python3-pip
    
    # Dependencias para Buildozer
    sudo apt install -y build-essential git python3 python3-dev ffmpeg libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev zlib1g-dev
    
    # Instalar Cython y Buildozer
    pip3 install --user Cython==0.29.33
    pip3 install --user buildozer
    
    echo "Buildozer instalado correctamente."
fi

# Verificar si existe el archivo icon.png
if [ ! -f "static/icon.png" ]; then
    echo "No se encontró el archivo icon.png. Creando directorio y archivo de icono..."
    mkdir -p static
    # Generar un icono básico
    echo "Asegúrate de proporcionar un archivo de ícono en static/icon.png (512x512 px)"
fi

# Verificar si existe el archivo presplash.png
if [ ! -f "static/presplash.png" ]; then
    echo "No se encontró el archivo presplash.png. Se usará un presplash por defecto."
    echo "Asegúrate de proporcionar un archivo de presplash en static/presplash.png (1024x1024 px)"
fi

# Crear enlace simbólico de main.py a main_kivy.py
if [ ! -f "main.py" ] || [ ! -L "main.py" ]; then
    echo "Creando enlace simbólico para main.py..."
    ln -sf main_kivy.py main.py
fi

# Ejecutar Buildozer
echo "Iniciando compilación con Buildozer..."
buildozer -v android debug

# Verificar si la compilación fue exitosa
if [ -f "bin/analizadorelectronico-*-debug.apk" ]; then
    echo ""
    echo "=== Compilación exitosa ==="
    echo "APK generado en: bin/analizadorelectronico-*-debug.apk"
    echo ""
    echo "Para instalar en un dispositivo Android:"
    echo "adb install -r bin/analizadorelectronico-*-debug.apk"
else
    echo ""
    echo "=== Error en la compilación ==="
    echo "Revisa los logs de buildozer para identificar el problema."
fi