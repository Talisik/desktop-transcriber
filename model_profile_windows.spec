# -*- mode: python ; coding: utf-8 -*-
# Windows build spec for model_profile

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# get absolute path to spec file directory (project root)
# __file__ is not available in spec context, use os.getcwd() since PyInstaller runs from project root
spec_root = Path(os.getcwd())

# collect resource_tracker data files and submodules
resource_tracker_datas = collect_data_files('resource_tracker')
resource_tracker_submodules = collect_submodules('resource_tracker')

# collect psutil data files (needed for system resource detection)
psutil_datas = collect_data_files('psutil')

# hidden imports
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
    [str(spec_root / 'model_profile_main.py')],
    pathex=[str(spec_root)],
    binaries=[],
    datas=resource_tracker_datas + psutil_datas,
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

