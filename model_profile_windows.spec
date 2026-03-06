# -*- mode: python ; coding: utf-8 -*-
# Windows build spec for model_profile

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# collect resource_tracker data files and submodules (needed for model profiles config)
resource_tracker_datas = collect_data_files('resource_tracker')
resource_tracker_submodules = collect_submodules('resource_tracker')

# explicitly include model_profiles.json (collect_data_files might miss JSON files)
model_profiles_json = [
    ('resource_tracker/model_profiles/model_profiles.json', 'resource_tracker/model_profiles')
]

# collect psutil data files (needed for system resource detection)
psutil_datas = collect_data_files('psutil')

# minimal hidden imports - only what's actually used
hiddenimports = [
    'resource_tracker',
    'resource_tracker.base',
    'resource_tracker.base.resource_tracker_base',
    'resource_tracker.concretions',
    'resource_tracker.concretions.desktop_resource_tracker',
    'resource_tracker.model_profiles',
    'resource_tracker.model_profiles.model_profile_reader',
    'psutil',
    'json',
    'argparse',
] + resource_tracker_submodules

a = Analysis(
    ['model_profile_main.py'],
    pathex=[],
    binaries=[],
    datas=resource_tracker_datas + model_profiles_json + psutil_datas,  # resource_tracker for model_profiles.json, psutil for system detection
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # exclude large ML libraries - model_profile only reads profiles and checks compatibility
        # it doesn't download or use models, so no ML libraries needed
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
        'model_manager',  # not used by model_profile
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
    name='model_profile',
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
