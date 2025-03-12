# -*- mode: python ; coding: utf-8 -*-
"""
Template spec file for PyInstaller packaging of Multi-Max

This file is configured to work with the dependencies copied by copy_dependencies.py.
To use this file:

1. Run copy_dependencies.py first to copy all required dependencies
2. Rename this file to main.spec
3. Run pyinstaller with: pyinstaller --clean --noconfirm main.spec
"""

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('cv2_copy', 'cv2'),
        ('numpy_copy', 'numpy'),
        ('dotenv_copy', 'dotenv'),
        ('pygame_copy', 'pygame'),
        ('Quartz_copy', 'Quartz'),
        ('Cocoa_copy', 'Cocoa'),
        ('objc_copy', 'objc'),
        ('AppKit_copy', 'AppKit'),
        ('Foundation_copy', 'Foundation'),
        ('CoreFoundation_copy', 'CoreFoundation'),
        ('PyObjCTools_copy', 'PyObjCTools'),
        ('psutil_copy', 'psutil'),
        ('memory_profiler_copy', '.'),
    ],
    hiddenimports=[
        'numpy', 'numpy.core', 'numpy.core.multiarray', 'numpy.core.umath',
        'platform', 'secrets', 'pygame', 'pygame.locals', 'dotenv', 'dotenv.main',
        'Quartz', 'Cocoa', 'objc', 'AppKit', 'Foundation', 'CoreFoundation', 'PyObjCTools',
        'xml', 'xml.etree', 'xml.etree.ElementTree', 'psutil', 'memory_profiler',
        'asyncio', 'pdb', 'tracemalloc', 'queue',
        'argparse', 'logging', 'threading', 'signal', 'subprocess',
        'gc', 'time', 'os', 'traceback'
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='multi-max',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Uncomment the following to create a macOS app bundle
# app = BUNDLE(
#     exe,
#     name='Multi-Max.app',
#     icon=None,  # Add path to icon file here
#     bundle_identifier='com.yourdomain.multi-max',
#     info_plist={
#         'CFBundleName': 'Multi-Max',
#         'CFBundleDisplayName': 'Multi-Max',
#         'CFBundleVersion': '1.0',
#         'CFBundleShortVersionString': '1.0',
#         'NSHighResolutionCapable': 'True',
#     },
# ) 