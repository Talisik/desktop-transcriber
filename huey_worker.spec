# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

# collect whisperx and dependencies (same as main.spec)
whisperx_datas = collect_data_files('whisperx')
whisperx_submodules = collect_submodules('whisperx')
faster_whisper_datas = collect_data_files('faster_whisper')
lightning_fabric_datas = collect_data_files('lightning_fabric')

# Comprehensive speechbrain collection using collect_all
# This ensures all data files, binaries, and hidden imports are included
speechbrain_datas, speechbrain_binaries, speechbrain_hiddenimports = collect_all('speechbrain')
speechbrain_submodules = collect_submodules('speechbrain')

# Comprehensive pyannote collection using collect_all
# This ensures all data files, binaries, and hidden imports are included
pyannote_datas, pyannote_binaries, pyannote_hiddenimports = collect_all('pyannote')
pyannote_submodules = collect_submodules('pyannote')

# collect huey data files
huey_datas = collect_data_files('huey')
huey_submodules = collect_submodules('huey')

# collect munchkin_chunker data files and submodules
munchkin_chunker_datas = collect_data_files('munchkin_chunker')
munchkin_chunker_submodules = collect_submodules('munchkin_chunker')

# filter out nvidia CUDA binaries - these should be loaded from system, not bundled
def filter_cuda_binaries(binaries_list):
    """Filter out nvidia CUDA library binaries to avoid extraction errors."""
    filtered = []
    for binary in binaries_list:
        # binary is a tuple: (source_path, dest_path, ...)
        # exclude any binaries with nvidia/ in the path
        if len(binary) >= 2:
            source_path = binary[0]
            dest_path = binary[1]
            # skip nvidia CUDA libraries
            if 'nvidia/' not in str(source_path) and 'nvidia/' not in str(dest_path):
                filtered.append(binary)
        else:
            filtered.append(binary)
    return filtered

# filter CUDA binaries from collected binaries
all_binaries = speechbrain_binaries + pyannote_binaries
filtered_binaries = filter_cuda_binaries(all_binaries)

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
    # huey imports
    'huey',
    'huey.bin',
    'huey.bin.huey_consumer',
    'huey.consumer',
    'huey.storage',
    'huey.api',
    'huey.api.results',
    'huey.signals',
    # our modules
    'consumer',  # our consumer module
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
    # munchkin_chunker imports
    'munchkin_chunker',
] + whisperx_submodules + huey_submodules + speechbrain_submodules + list(speechbrain_hiddenimports) + pyannote_submodules + list(pyannote_hiddenimports) + munchkin_chunker_submodules

a = Analysis(
    ['huey_worker.py'],
    pathex=[],
    binaries=filtered_binaries,
    datas=whisperx_datas + faster_whisper_datas + lightning_fabric_datas + speechbrain_datas + pyannote_datas + huey_datas + munchkin_chunker_datas,
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
    name='huey_worker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['libcublas*.so*', 'libcudnn*.so*', 'libcufft*.so*', 'libcurand*.so*', 'libcusolver*.so*', 'libcusparse*.so*', '*nvidia*'],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

