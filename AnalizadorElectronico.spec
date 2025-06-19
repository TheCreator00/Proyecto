# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Lista de archivos y carpetas a incluir
added_files = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('data', 'data'),
    ('outputs', 'outputs'),
    ('results', 'results'),
    ('uploads', 'uploads'),
]

# Análisis para encontrar todos los módulos necesarios
a = Analysis(
    ['run_app_standalone.py'],
    pathex=['.'],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'flask',
        'werkzeug',
        'jinja2',
        'sqlite3',
        'numpy',
        'cv2',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'PIL',
        'tensorflow',
        'fitz',  # PyMuPDF
        'openai',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Configura el ejecutable
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AnalizadorElectronico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='static/favicon.ico' if os.path.exists('static/favicon.ico') else None,
)