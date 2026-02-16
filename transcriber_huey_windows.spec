# -*- mode: python ; coding: utf-8 -*-
# Windows build spec for transcriber_huey

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

# collect whisperx and dependencies (same as main.spec)
whisperx_datas = collect_data_files('whisperx')
whisperx_submodules = collect_submodules('whisperx')
faster_whisper_datas = collect_data_files('faster_whisper')
lightning_fabric_datas = collect_data_files('lightning_fabric')

# Comprehensive speechbrain collection using collect_all
speechbrain_datas, speechbrain_binaries, speechbrain_hiddenimports = collect_all('speechbrain')
speechbrain_submodules = collect_submodules('speechbrain')

# Comprehensive pyannote collection using collect_all
pyannote_datas, pyannote_binaries, pyannote_hiddenimports = collect_all('pyannote')
pyannote_submodules = collect_submodules('pyannote')

# collect huey data files
huey_datas = collect_data_files('huey')
huey_submodules = collect_submodules('huey')

# Comprehensive transformers collection to fix torchcodec and metadata issues
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')

# hidden imports for ML libs + huey
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
    'speechbrain.dataio',
    'speechbrain.dataio.dataio',
    'speechbrain.dataio.encoder',
    'speechbrain.dataio.batch',
    'speechbrain.utils',
    'speechbrain.utils.data_utils',
    'speechbrain.core',
    'speechbrain.pretrained',
    'pyannote.audio',
    'pyannote.audio.models',
    'pyannote.audio.pipelines',
    'pyannote.audio.pipelines.speaker_diarization',
    'pyannote.audio.pipelines.speaker_verification',
    'pyannote.audio.pipelines.utils',
    'pytorch_lightning',
    'torchmetrics',
    'semver',
    'pandas',
    # huey imports
    'huey',
    'huey.storage',
    'huey.api',
    'huey.api.results',
    # our modules
    'transcriber_huey',  # our task module
    'transcriber',
    'transcriber.base',
    'transcriber.base.transcriber_base',
    'transcriber.concretions',
    'transcriber.concretions.whisperx_implementation',
    'transcriber.concretions.whisperx_implementation.whisperx_transcriber',
    'transcriber.concretions.whisperx_implementation.chemas',
    'transcriber.concretions.whisperx_implementation.chemas.payload',
    'transcriber.concretions.whisperx_implementation.chemas.payload.transcriber_argument_schema',
    'transcriber.concretions.whisperx_implementation.chemas.custom_types',
    'transcriber.concretions.whisperx_implementation.chemas.custom_types.parameter_types',
    'pydantic',
] + whisperx_submodules + huey_submodules + speechbrain_submodules + list(speechbrain_hiddenimports) + pyannote_submodules + list(pyannote_hiddenimports) + list(transformers_hiddenimports)

a = Analysis(
    ['transcriber_huey.py'],
    pathex=[],
    binaries=speechbrain_binaries + pyannote_binaries + transformers_binaries,
    datas=whisperx_datas + faster_whisper_datas + lightning_fabric_datas + speechbrain_datas + pyannote_datas + huey_datas + transformers_datas,
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
    name='transcriber_huey',
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










