# -*- mode: python ; coding: utf-8 -*-
# Windows build spec for huey_worker

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

# collect munchkin_chunker data files and submodules - use collect_all to include onnx_model
# no filtering for munchkin_chunker - include all data including onnx models
munchkin_chunker_datas_raw, munchkin_chunker_binaries_raw, munchkin_chunker_hiddenimports_raw = collect_all('munchkin_chunker')
munchkin_chunker_submodules = collect_submodules('munchkin_chunker')

# Comprehensive transformers collection to fix torchcodec and metadata issues
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')

# Comprehensive torchcodec collection - needed for whisperx audio processing
# torchcodec requires FFmpeg DLLs but we'll handle that separately
torchcodec_datas, torchcodec_binaries, torchcodec_hiddenimports = collect_all('torchcodec')
torchcodec_submodules = collect_submodules('torchcodec')

# munchkin_chunker - no filtering, keep all data including onnx_model
munchkin_chunker_datas = munchkin_chunker_datas_raw  # no filtering - keep all munchkin_chunker data including onnx_model
munchkin_chunker_hiddenimports = list(munchkin_chunker_hiddenimports_raw) if munchkin_chunker_hiddenimports_raw else []
munchkin_chunker_binaries = munchkin_chunker_binaries_raw if munchkin_chunker_binaries_raw else []

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
    # torchcodec imports
    'torchcodec',
    'torchcodec._core',
    'torchcodec._core.ops',
] + whisperx_submodules + huey_submodules + speechbrain_submodules + list(speechbrain_hiddenimports) + pyannote_submodules + list(pyannote_hiddenimports) + list(transformers_hiddenimports) + munchkin_chunker_submodules + torchcodec_submodules + list(torchcodec_hiddenimports)

# filter data files to exclude test/example/doc files
all_datas = whisperx_datas + faster_whisper_datas + lightning_fabric_datas + speechbrain_datas + pyannote_datas + huey_datas + transformers_datas + munchkin_chunker_datas + torchcodec_datas

# munchkin_chunker expects onnx_model at root of _MEIPASS, not under munchkin_chunker/
# add explicit datas entry to copy onnx_model directory to root level
# improved path detection with better error handling and multiple fallback strategies
munchkin_onnx_src = None
onnx_model_file = None

# strategy 1: try venv path (most common in CI/build environments)
venv_path = os.path.dirname(os.path.dirname(sys.executable)) if hasattr(sys, 'executable') else None
if venv_path:
    # Windows: try python3.11, python3.12, python3.13 depending on version
    for py_version in ['python3.11', 'python3.12', 'python3.13']:
        test_path = os.path.join(venv_path, 'lib', py_version, 'site-packages', 'munchkin_chunker', 'onnx_model')
        model_file_path = os.path.join(test_path, 'model.onnx')
        if os.path.exists(test_path) and os.path.isdir(test_path):
            # verify model.onnx exists
            if os.path.exists(model_file_path):
                munchkin_onnx_src = test_path
                onnx_model_file = model_file_path
                break

# strategy 2: try site-packages (fallback for system installs)
if not munchkin_onnx_src:
    try:
        import site
        for sp in site.getsitepackages():
            test_path = os.path.join(sp, 'munchkin_chunker', 'onnx_model')
            model_file_path = os.path.join(test_path, 'model.onnx')
            if os.path.exists(test_path) and os.path.isdir(test_path):
                if os.path.exists(model_file_path):
                    munchkin_onnx_src = test_path
                    onnx_model_file = model_file_path
                    break
    except Exception:
        pass

# strategy 3: try relative to current working directory (for development)
if not munchkin_onnx_src:
    cwd = os.getcwd()
    test_paths = [
        os.path.join(cwd, 'munchkin_chunker', 'onnx_model'),
        os.path.join(cwd, '..', 'munchkin_chunker', 'onnx_model'),
    ]
    for test_path in test_paths:
        model_file_path = os.path.join(test_path, 'model.onnx')
        if os.path.exists(test_path) and os.path.isdir(test_path):
            if os.path.exists(model_file_path):
                munchkin_onnx_src = os.path.abspath(test_path)
                onnx_model_file = model_file_path
                break

if munchkin_onnx_src and onnx_model_file:
    # datas format: (source_path, destination_path_in_bundle)
    # destination is 'onnx_model' at root (not 'munchkin_chunker/onnx_model')
    # this ensures munchkin_chunker can find it at runtime in _MEIPASS/onnx_model/
    onnx_model_datas = [(munchkin_onnx_src, 'onnx_model')]
    all_datas = all_datas + onnx_model_datas
    print(f"[PyInstaller] Found ONNX model at: {munchkin_onnx_src}")
    print(f"[PyInstaller] Model file exists: {os.path.exists(onnx_model_file)}")
else:
    print(f"[PyInstaller] WARNING: ONNX model not found. munchkin_chunker may fail at runtime.")
    print(f"[PyInstaller] Searched in venv: {venv_path}")
    try:
        import site
        site_packages = site.getsitepackages()
        print(f"[PyInstaller] Searched in site-packages: {site_packages}")
    except Exception:
        print(f"[PyInstaller] Could not query site-packages")

filtered_datas = all_datas

all_binaries = speechbrain_binaries + pyannote_binaries + transformers_binaries + munchkin_chunker_binaries + torchcodec_binaries

# collect FFmpeg DLLs if available (torchcodec needs them on Windows)
# FFmpeg DLLs should be in the same directory as ffmpeg.exe
# we'll try to find them from common locations or environment
ffmpeg_dlls = []
ffmpeg_exe = None

# try FFMPEG_PATH first (most reliable)
ffmpeg_path_env = os.environ.get('FFMPEG_PATH')
if ffmpeg_path_env and os.path.exists(ffmpeg_path_env):
    if os.path.isfile(ffmpeg_path_env):
        ffmpeg_exe = ffmpeg_path_env
    elif os.path.isdir(ffmpeg_path_env):
        # if it's a directory, look for ffmpeg.exe inside
        potential_exe = os.path.join(ffmpeg_path_env, 'ffmpeg.exe')
        if os.path.exists(potential_exe):
            ffmpeg_exe = potential_exe

# fallback to PATH search
if not ffmpeg_exe:
    import shutil
    ffmpeg_exe = shutil.which('ffmpeg.exe') or shutil.which('ffmpeg')

if ffmpeg_exe:
    ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_exe))
    # look for common FFmpeg DLLs that torchcodec needs
    # these are the DLLs required by torchcodec on Windows
    dll_names = ['avcodec-*.dll', 'avformat-*.dll', 'avutil-*.dll', 'swresample-*.dll', 'swscale-*.dll']
    import glob
    for dll_pattern in dll_names:
        dll_files = glob.glob(os.path.join(ffmpeg_dir, dll_pattern))
        for dll_file in dll_files:
            # bundle DLLs in root of _MEIPASS so torchcodec can find them
            # torchcodec looks for DLLs in the same directory as the executable or in PATH
            ffmpeg_dlls.append((dll_file, os.path.basename(dll_file)))
    if ffmpeg_dlls:
        print(f"[PyInstaller] Found {len(ffmpeg_dlls)} FFmpeg DLLs in: {ffmpeg_dir}")
        all_binaries = all_binaries + ffmpeg_dlls
    else:
        print(f"[PyInstaller] WARNING: FFmpeg DLLs not found in {ffmpeg_dir}. torchcodec may fail to load.")
        print(f"[PyInstaller] Ensure FFmpeg 'full-shared' version is installed with DLLs.")
        print(f"[PyInstaller] Required DLLs: avcodec-*.dll, avformat-*.dll, avutil-*.dll, swresample-*.dll, swscale-*.dll")
else:
    print(f"[PyInstaller] WARNING: ffmpeg.exe not found. torchcodec may fail to load.")
    print(f"[PyInstaller] Set FFMPEG_PATH or ensure ffmpeg is in PATH during build.")

a = Analysis(
    ['huey_worker.py'],
    pathex=[],
    binaries=all_binaries,
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
    exclude_binaries=True,  # onedir mode - avoids 4GB archive limit and improves startup time
    name='huey_worker',
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='huey_worker',
)

