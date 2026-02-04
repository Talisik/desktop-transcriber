# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# collect whisperx and dependencies
whisperx_datas = collect_data_files('whisperx')
whisperx_submodules = collect_submodules('whisperx')
faster_whisper_datas = collect_data_files('faster_whisper')
lightning_fabric_datas = collect_data_files('lightning_fabric')
speechbrain_datas = collect_data_files('speechbrain')
pyannote_datas = collect_data_files('pyannote')

# hidden imports for ML libs
hiddenimports = [
    'whisperx',
    'whisperx.asr',
    'whisperx.diarize',
    'whisperx.alignment',
    'faster_whisper',
    'torch',
    'torchaudio',
    'transformers',
    'pyannote.audio',
    'ctranslate2',
    'onnxruntime',
    'sklearn',
    'sklearn.utils._cython_blas',
    'scipy',
    'numpy',
    'lightning_fabric',
    'lightning_fabric.utilities.cloud_io',
    'speechbrain',
    'pyannote.audio',
    'pyannote.audio.pipelines',
    'pyannote.audio.pipelines.speaker_diarization',
    'pyannote.audio.pipelines.speaker_verification',
] + whisperx_submodules

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=whisperx_datas + faster_whisper_datas + lightning_fabric_datas + speechbrain_datas + pyannote_datas,
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
    name='main',
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