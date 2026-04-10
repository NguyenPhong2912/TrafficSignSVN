$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (Test-Path ".\.venv\Scripts\python.exe") {
    & ".\.venv\Scripts\python.exe" ".\desktop_app.py"
} elseif (Test-Path "..\traffic\Scripts\python.exe") {
    & "..\traffic\Scripts\python.exe" ".\desktop_app.py"
} else {
    & "python" ".\desktop_app.py"
}
