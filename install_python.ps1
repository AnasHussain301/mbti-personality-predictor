$ErrorActionPreference = 'Stop'

$pythonUrl = 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe'
$dest = Join-Path $env:TEMP 'python311.exe'
$installDir = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311'

if (-Not (Test-Path $dest)) {
    Write-Host "Downloading Python installer to $dest..."
    Invoke-WebRequest -Uri $pythonUrl -OutFile $dest -UseBasicParsing
} else {
    Write-Host "Installer already downloaded at $dest"
}

Write-Host "Installing Python to $installDir..."
Start-Process -FilePath $dest -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_pip=1","TargetDir=$installDir" -Wait -NoNewWindow

$pythonExe = Join-Path $installDir 'python.exe'
if (-Not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found after install: $pythonExe"
    exit 1
}

Write-Host "Python installed at $pythonExe"
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt
& $pythonExe --version
