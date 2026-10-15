@echo off
REM ============================================================
REM  Double-click this file to build FaceSearchGUI.exe
REM
REM  It does everything by itself: sets up Python, installs the
REM  needed libraries, and builds the program. The first (and only)
REM  time takes a while - 10 to 30 minutes - just let it run.
REM ============================================================
cd /d "%~dp0"

echo [1 of 3] Setting up...
py -3 -m venv .venv
if errorlevel 1 (
    echo.
    echo Could not set up Python. Make sure Python 3.10 or newer is
    echo installed from python.org and that "Add python.exe to PATH"
    echo was ticked during setup.
    pause
    exit /b 1
)

echo [2 of 3] Installing libraries - this is the slow part, please wait...
.venv\Scripts\python -m pip install --upgrade pip
REM  Remove stale packages from older builds. InsightFace is no longer used;
REM  pillow-heif was replaced by pi-heif (its wheels bundle a GPL encoder);
REM  the others' native DLLs crash PyInstaller.
.venv\Scripts\python -m pip uninstall -y insightface pillow-heif windows-toasts winrt-runtime winrt-Windows.Data.Xml.Dom winrt-Windows.Foundation winrt-Windows.Foundation.Collections winrt-Windows.UI.Notifications 2>nul
.venv\Scripts\python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo.
    echo Installing the libraries failed. Check your internet
    echo connection and double-click this file to try again.
    pause
    exit /b 1
)

REM  Graphics-card support: swap the processor-only onnxruntime for the
REM  DirectML build (works with most Windows graphics cards). Both install
REM  the same "onnxruntime" module, so remove both first. If anything goes
REM  wrong, go back to the processor-only build - the app still works.
echo Adding graphics-card support...
.venv\Scripts\python -m pip uninstall -y onnxruntime onnxruntime-directml >nul 2>nul
.venv\Scripts\python -m pip install onnxruntime-directml==1.24.4
if errorlevel 1 goto cpu_only
.venv\Scripts\python -c "import onnxruntime; assert 'DmlExecutionProvider' in onnxruntime.get_available_providers()"
if errorlevel 1 goto cpu_only
echo Graphics-card support added.
goto build
:cpu_only
echo Graphics-card support isn't available here; the program will use the processor.
.venv\Scripts\python -m pip uninstall -y onnxruntime onnxruntime-directml >nul 2>nul
.venv\Scripts\python -m pip install onnxruntime==1.30.0
:build

echo [3 of 3] Building FaceSearchGUI.exe - almost there...
REM  Clean any previous build so a stale exe can never be reused.
REM  A running (or virus-scanned) exe can't be deleted, but it can be
REM  renamed: move it aside so the new one can be written, and clear out
REM  copies moved aside by earlier builds.
if exist build rmdir /s /q build
if exist dist\FaceSearchGUI_old_*.exe del /q dist\FaceSearchGUI_old_*.exe >nul 2>nul
if exist dist\FaceSearchGUI.exe ren dist\FaceSearchGUI.exe FaceSearchGUI_old_%RANDOM%.exe >nul 2>nul
if exist dist\FaceSearchGUI.exe (
    echo.
    echo The old dist\FaceSearchGUI.exe is locked and can't be replaced.
    echo Close Face Search ^(and end FaceSearchGUI in Task Manager if needed^),
    echo then double-click this file again. Restarting the PC also releases it.
    pause
    exit /b 1
)
.venv\Scripts\flet pack face_search_gui.py --name "FaceSearchGUI" --product-name "Face Search" --product-version "2.0" --icon app_icon.ico --yes
if errorlevel 1 (
    echo.
    echo The build failed. Scroll up to see the error message.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Your program is here:
echo    dist\FaceSearchGUI.exe
echo.
echo  Double-click it to run. You can copy it anywhere you like,
echo  even onto another computer - it needs nothing else installed.
echo.
echo  Note: the very first launch downloads the face model once
echo  (about 262 MB). After that it works fully offline.
echo ============================================================
pause
