@echo off
echo Compilando Analizador de Componentes Electronicos para Windows...

rem Instalar PyInstaller si no está instalado
pip install pyinstaller pillow

rem Crear el ejecutable usando el archivo spec
pyinstaller AnalizadorElectronico.spec

echo.
echo Compilacion completada. Verifica la carpeta "dist" para el ejecutable.
echo.
pause