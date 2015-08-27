@echo off
if NOT "%PYMOTE_ENV%" == "" ( call %PYMOTE_ENV%\Scripts\activate.bat )
ipython --pylab=qt --profile=pymote