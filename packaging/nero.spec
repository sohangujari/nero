# PyInstaller spec — builds a single-file `nero` binary bundling Python 3.12 and
# all dependencies. Model weights (kokoro-onnx, whisper) are NOT bundled; they're
# fetched to a per-user cache on first launch to keep the binary small.
#
# Build:  uv run pyinstaller packaging/nero.spec
# Output: dist/nero  (rename per-OS in the release workflow)

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

hiddenimports = []
datas = []
binaries = []

# Native/ML packages that PyInstaller can't fully trace by static analysis.
# (sounddevice/soundfile are single modules with dedicated contrib hooks that
# already bundle their PortAudio/libsndfile binaries — no manual collection.)
for pkg in ("onnxruntime", "faster_whisper"):
    hiddenimports += collect_submodules(pkg)
    binaries += collect_dynamic_libs(pkg)
    datas += collect_data_files(pkg)

# kokoro-onnx ships tokenizer/phonemizer data and config.
datas += collect_data_files("kokoro_onnx")
hiddenimports += collect_submodules("kokoro_onnx")

# litellm bundles token cost/model metadata JSON it loads at runtime.
datas += collect_data_files("litellm")

# keyring resolves OS backends via entry points at runtime.
hiddenimports += collect_submodules("keyring")

a = Analysis(
    ["../nero/__main__.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "tensorflow"],  # not used by kokoro-onnx path; keep size down
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="nero",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
