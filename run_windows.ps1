$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$engineDir = Join-Path $scriptDir "legacy_engine/bin/windows"
$guiDir = Join-Path $scriptDir "legacy_gui/bin/windows"
$engineBin = Join-Path $engineDir "JunQiEngine.exe"
$guiBin = Join-Path $guiDir "JunQiGUI.exe"

if (-not (Test-Path $engineBin) -or -not (Test-Path $guiBin)) {
    $makeCommand = Get-Command mingw32-make -ErrorAction SilentlyContinue
    if ($null -eq $makeCommand) {
        $makeCommand = Get-Command make -ErrorAction SilentlyContinue
    }
    if ($null -eq $makeCommand) {
        throw "Windows binaries are missing. Install MSYS2 UCRT64 and run 'mingw32-make windows' from its terminal first."
    }
    Push-Location $scriptDir
    try {
        & $makeCommand.Source windows
        if ($LASTEXITCODE -ne 0) { throw "Windows build failed with exit code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }
}

$engineProcesses = @()
$guiProcess = $null
try {
    # The GUI talks to two local engine seats: 6678 and 5678, and listens on
    # UDP 1234. Keep the process handles local so cleanup cannot affect an
    # unrelated JunQi process.
    $engineProcesses += Start-Process -FilePath $engineBin -WorkingDirectory $engineDir `
        -ArgumentList @("--seat", "0", "--local-port", "6678", "--remote-port", "1234") `
        -PassThru -WindowStyle Hidden
    $engineProcesses += Start-Process -FilePath $engineBin -WorkingDirectory $engineDir `
        -ArgumentList @("--seat", "1", "--local-port", "5678", "--remote-port", "1234") `
        -PassThru -WindowStyle Hidden

    Start-Sleep -Milliseconds 500
    foreach ($process in $engineProcesses) {
        if ($process.HasExited) {
            throw "A JunQi engine exited during startup. Check whether UDP ports 5678/6678 are already in use."
        }
    }

    Write-Host "Starting the JunQi Windows client. Close the window to stop both engines."
    $guiProcess = Start-Process -FilePath $guiBin -WorkingDirectory $guiDir -PassThru
    Wait-Process -Id $guiProcess.Id
} finally {
    if ($guiProcess -ne $null -and -not $guiProcess.HasExited) {
        Stop-Process -Id $guiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    foreach ($process in $engineProcesses) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
