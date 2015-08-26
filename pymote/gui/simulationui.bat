@echo off
if /I [%1]==[] goto all
if /I [%1]==[help] goto help


goto end

:help
Echo Compilation of simulationui.py and icons_rc.py
goto end

:all
echo Compilation of simulationui.py 
call pyside-uic simulation.ui -o simulationui.py
REM pyside-rcc.exe is in site-packages/PySide dir so either put it in a PATH or copy exe to Scripts
echo Compilation of icons_rc.py 
call pyside-rcc icons.qrc -o icons_rc.py
goto end

:end
