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

# collect munchkin_chunker data files and submodules
munchkin_chunker_datas = collect_data_files('munchkin_chunker')
munchkin_chunker_submodules = collect_submodules('munchkin_chunker')

# Comprehensive transformers collection to fix torchcodec and metadata issues
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')

# Comprehensive torchcodec collection - needed for pyannote/whisperx audio decoding in frozen apps
# Note: torchcodec may require FFmpeg DLLs at runtime on Windows
torchcodec_datas, torchcodec_binaries, torchcodec_hiddenimports = collect_all('torchcodec')
torchcodec_submodules = collect_submodules('torchcodec')

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
    # munchkin_chunker imports
    'munchkin_chunker',
] + whisperx_submodules + huey_submodules + speechbrain_submodules + list(speechbrain_hiddenimports) + pyannote_submodules + list(pyannote_hiddenimports) + list(transformers_hiddenimports) + munchkin_chunker_submodules + torchcodec_submodules + list(torchcodec_hiddenimports) + [
    # torchcodec imports
    'torchcodec',
    'torchcodec._core',
    'torchcodec._core.ops',
]

all_binaries = speechbrain_binaries + pyannote_binaries + transformers_binaries + torchcodec_binaries

# collect FFmpeg DLLs if available (torchcodec may need them on Windows)
# FFmpeg DLLs should be in the same directory as ffmpeg.exe
ffmpeg_dlls = []
ffmpeg_exe = None

# try FFMPEG_PATH first (most reliable)
ffmpeg_path_env = os.environ.get('FFMPEG_PATH')
if ffmpeg_path_env and os.path.exists(ffmpeg_path_env):
    if os.path.isfile(ffmpeg_path_env):
        ffmpeg_exe = ffmpeg_path_env
    elif os.path.isdir(ffmpeg_path_env):
        potential_exe = os.path.join(ffmpeg_path_env, 'ffmpeg.exe')
        if os.path.exists(potential_exe):
            ffmpeg_exe = potential_exe

# fallback to PATH search
if not ffmpeg_exe:
    import shutil
    ffmpeg_exe = shutil.which('ffmpeg.exe') or shutil.which('ffmpeg')

if ffmpeg_exe:
    ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_exe))
    dll_names = ['avcodec-*.dll', 'avformat-*.dll', 'avutil-*.dll', 'swresample-*.dll', 'swscale-*.dll']
    import glob
    for dll_pattern in dll_names:
        for dll_file in glob.glob(os.path.join(ffmpeg_dir, dll_pattern)):
            ffmpeg_dlls.append((dll_file, os.path.basename(dll_file)))
    if ffmpeg_dlls:
        all_binaries = all_binaries + ffmpeg_dlls
        print(f"[PyInstaller] Found {len(ffmpeg_dlls)} FFmpeg DLLs in: {ffmpeg_dir}")
    else:
        print(f"[PyInstaller] WARNING: FFmpeg DLLs not found in {ffmpeg_dir}. torchcodec may fail to load.")
        print(f"[PyInstaller] Ensure FFmpeg 'full-shared' version is installed with DLLs.")
else:
    print(f"[PyInstaller] WARNING: ffmpeg.exe not found. torchcodec may fail to load.")
    print(f"[PyInstaller] Set FFMPEG_PATH or ensure ffmpeg is in PATH during build.")

a = Analysis(
    ['transcriber_huey.py'],
    pathex=[],
    binaries=all_binaries,
    datas=whisperx_datas + faster_whisper_datas + lightning_fabric_datas + speechbrain_datas + pyannote_datas + huey_datas + transformers_datas + munchkin_chunker_datas + torchcodec_datas,
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





















