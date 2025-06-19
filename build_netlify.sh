Y SI NO TENGO LA KEY?#!/bin/bash

# Configurar entorno Python
echo "Configurando entorno Python..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Instalar dependencias de Node.js
echo "Instalando dependencias de Node.js..."
npm install

# Construir la aplicación
echo "Construyendo la aplicación..."
npm run build

# Preparar directorio de funciones
echo "Preparando funciones serverless..."
mkdir -p dist/functions

# Copiar archivos de Python necesarios
echo "Copiando archivos de Python..."
cp -r ai_analyzer.py xai_integration.py config/* dist/functions/

# Instalar dependencias de Python en el directorio de funciones
cd dist/functions
pip install -r ../../requirements.txt -t .

echo "Construcción completada!"