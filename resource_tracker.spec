# -*- mode: python ; coding: utf-8 -*-
# Cross-platform build spec for resource_tracker

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

# collect psutil data files (needed for system resource detection)
psutil_datas = collect_data_files('psutil')

# minimal hidden imports - only what's actually used
hiddenimports = [
    'resource_tracker.base.resource_tracker_base',
    'resource_tracker.concretions.desktop_resource_tracker',
    'psutil',
    'json',
]

a = Analysis(
    ['resource_tracker_main.py'],
    pathex=[],
    binaries=[],
    datas=psutil_datas,  # only psutil data files needed
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # exclude large ML libraries - resource_tracker uses optional torch imports
        # and falls back to nvidia-smi, so we don't need to bundle these
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'keras',
        'sklearn',
        'scipy',
        'numpy',
        'pandas',
        'matplotlib',
        'PIL',
        'cv2',
        'whisperx',
        'faster_whisper',
        'transformers',
        'speechbrain',
        'pyannote',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='resource_tracker',
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
)

