# -*- mode: python ; coding: utf-8 -*-
# Windows build spec for model_manager

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# collect model_manager data files and submodules
model_manager_datas = collect_data_files('model_manager')
model_manager_submodules = collect_submodules('model_manager')

# collect resource_tracker data files (needed for model profiles)
resource_tracker_datas = collect_data_files('resource_tracker')
resource_tracker_submodules = collect_submodules('resource_tracker')

# collect whisperx and dependencies (needed for model downloads)
whisperx_datas = collect_data_files('whisperx')
whisperx_submodules = collect_submodules('whisperx')
faster_whisper_datas = collect_data_files('faster_whisper')

# collect psutil data files (needed for system resource detection in standalone mode)
psutil_datas = collect_data_files('psutil')

# hidden imports
hiddenimports = [
    'model_manager',
    'model_manager.base',
    'model_manager.base.model_manager_base',
    'model_manager.concretions',
    'model_manager.concretions.whisperx_model_manager',
    'resource_tracker',
    'resource_tracker.model_profiles',
    'resource_tracker.model_profiles.model_profile_reader',
    'whisperx',
    'whisperx.asr',
    'faster_whisper',
    'torch',
    'psutil',
    'json',
    'argparse',
    'gc',
] + model_manager_submodules + resource_tracker_submodules + whisperx_submodules

a = Analysis(
    ['model_manager_main.py'],
    pathex=[],
    binaries=[],
    datas=model_manager_datas + resource_tracker_datas + whisperx_datas + faster_whisper_datas + psutil_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='model_manager',
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

