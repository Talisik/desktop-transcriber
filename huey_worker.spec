# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

# collect whisperx and dependencies - use collect_all to include everything
# no filtering for whisperx dependencies - include all necessary data
whisperx_datas_raw, whisperx_binaries_raw, whisperx_hiddenimports_raw = collect_all('whisperx')
whisperx_submodules = collect_submodules('whisperx')

faster_whisper_datas_raw, faster_whisper_binaries_raw, faster_whisper_hiddenimports_raw = collect_all('faster_whisper')

lightning_fabric_datas_raw, lightning_fabric_binaries_raw, lightning_fabric_hiddenimports_raw = collect_all('lightning_fabric')

# Selective speechbrain collection - use collect_all but filter results
# collect_all preserves directory structure (needed for speechbrain.utils)
# but we filter out test/example data to reduce size
speechbrain_datas_raw, speechbrain_binaries_raw, speechbrain_hiddenimports_raw = collect_all('speechbrain')
# filter will be applied later after filter_data_files is defined
speechbrain_submodules = collect_submodules('speechbrain')

# Selective pyannote collection - use collect_all but filter results
pyannote_datas_raw, pyannote_binaries_raw, pyannote_hiddenimports_raw = collect_all('pyannote')
# filter will be applied later after filter_data_files is defined
pyannote_submodules = collect_submodules('pyannote')

# collect huey data files
huey_datas = collect_data_files('huey')
huey_submodules = collect_submodules('huey')

# collect munchkin_chunker data files and submodules - use collect_all to include onnx_model
# no filtering for munchkin_chunker - include all data including onnx models
munchkin_chunker_datas_raw, munchkin_chunker_binaries_raw, munchkin_chunker_hiddenimports_raw = collect_all('munchkin_chunker')
munchkin_chunker_submodules = collect_submodules('munchkin_chunker')
# filter will be applied later after filter_cuda_binaries is defined

# filter out test/example/doc data files to reduce bundle size
# but keep all whisperx and its dependencies data (no filtering)
def filter_data_files(datas_list, keep_all=False):
    """Filter out test, example, and documentation files."""
    # if keep_all is True, don't filter anything (for whisperx dependencies)
    if keep_all:
        return datas_list
    
    filtered = []
    exclude_patterns = [
        '/test/', '/tests/', '/example/', '/examples/', 
        '/doc/', '/docs/', '/__pycache__/', '/notebooks/',
        '.md', '.rst', '.ipynb',  # documentation files (but keep .txt files like version.txt)
        '/benchmark/', '/demo/', '/scripts/',  # development files
    ]
    # essential files that should never be filtered
    essential_files = ['version.txt', '__init__.py', 'requirements.txt']
    
    for data in datas_list:
        # data is a tuple: (source_path, dest_path)
        if len(data) >= 2:
            source_path = str(data[0])
            dest_path = str(data[1])
            # keep essential files (like version.txt)
            if any(essential in source_path or essential in dest_path for essential in essential_files):
                filtered.append(data)
            # skip if matches any exclude pattern
            elif not any(pattern in source_path or pattern in dest_path for pattern in exclude_patterns):
                filtered.append(data)
        else:
            filtered.append(data)
    return filtered

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

# filter speechbrain and pyannote data files to exclude test/example/doc files
speechbrain_datas = filter_data_files(speechbrain_datas_raw, keep_all=False)
pyannote_datas = filter_data_files(pyannote_datas_raw, keep_all=False)

# whisperx and its dependencies - no filtering, keep all data
whisperx_datas = whisperx_datas_raw  # no filtering - keep all whisperx data including tests
faster_whisper_datas = faster_whisper_datas_raw  # no filtering - keep all faster_whisper data
lightning_fabric_datas = lightning_fabric_datas_raw  # no filtering - keep all lightning_fabric data
whisperx_hiddenimports = list(whisperx_hiddenimports_raw)

# munchkin_chunker - no filtering, keep all data including onnx_model
munchkin_chunker_datas = munchkin_chunker_datas_raw  # no filtering - keep all munchkin_chunker data including onnx_model
munchkin_chunker_hiddenimports = list(munchkin_chunker_hiddenimports_raw) if munchkin_chunker_hiddenimports_raw else []

# filter CUDA binaries from collected binaries
speechbrain_binaries = filter_cuda_binaries(speechbrain_binaries_raw)
pyannote_binaries = filter_cuda_binaries(pyannote_binaries_raw)
# whisperx dependencies binaries - filter CUDA but keep everything else
whisperx_binaries = filter_cuda_binaries(whisperx_binaries_raw)
faster_whisper_binaries = filter_cuda_binaries(faster_whisper_binaries_raw)
lightning_fabric_binaries = filter_cuda_binaries(lightning_fabric_binaries_raw)
# munchkin_chunker binaries (if any) - filter CUDA but keep everything else
munchkin_chunker_binaries = filter_cuda_binaries(munchkin_chunker_binaries_raw) if munchkin_chunker_binaries_raw else []
all_binaries = speechbrain_binaries + pyannote_binaries + whisperx_binaries + faster_whisper_binaries + lightning_fabric_binaries + munchkin_chunker_binaries
filtered_binaries = all_binaries

# get hidden imports
speechbrain_hiddenimports = list(speechbrain_hiddenimports_raw)
pyannote_hiddenimports = list(pyannote_hiddenimports_raw)

# filter data files to exclude test/example/doc files
all_datas = whisperx_datas + faster_whisper_datas + lightning_fabric_datas + speechbrain_datas + pyannote_datas + huey_datas + munchkin_chunker_datas

# munchkin_chunker expects onnx_model at root of _MEIPASS, not under munchkin_chunker/
# add explicit datas entry to copy onnx_model directory to root level
# find munchkin_chunker package path in venv
venv_path = os.path.dirname(os.path.dirname(sys.executable)) if hasattr(sys, 'executable') else None
munchkin_onnx_src = None
if venv_path:
    test_path = os.path.join(venv_path, 'lib', 'python3.11', 'site-packages', 'munchkin_chunker', 'onnx_model')
    if os.path.exists(test_path):
        munchkin_onnx_src = test_path
else:
    # fallback: try site-packages
    import site
    for sp in site.getsitepackages():
        test_path = os.path.join(sp, 'munchkin_chunker', 'onnx_model')
        if os.path.exists(test_path):
            munchkin_onnx_src = test_path
            break

if munchkin_onnx_src:
    # datas format: (source_path, destination_path_in_bundle)
    # destination is 'onnx_model' at root (not 'munchkin_chunker/onnx_model')
    onnx_model_datas = [(munchkin_onnx_src, 'onnx_model')]
    all_datas = all_datas + onnx_model_datas

filtered_datas = all_datas

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
] + whisperx_submodules + huey_submodules + speechbrain_submodules + list(speechbrain_hiddenimports) + pyannote_submodules + list(pyannote_hiddenimports) + munchkin_chunker_submodules + list(munchkin_chunker_hiddenimports)

a = Analysis(
    ['huey_worker.py'],
    pathex=[],
    binaries=filtered_binaries,
    datas=filtered_datas,
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
    [],
    exclude_binaries=True,  # onedir mode - avoids 4GB archive limit
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['libcublas*.so*', 'libcudnn*.so*', 'libcufft*.so*', 'libcurand*.so*', 'libcusolver*.so*', 'libcusparse*.so*', '*nvidia*'],
    name='huey_worker',
)

