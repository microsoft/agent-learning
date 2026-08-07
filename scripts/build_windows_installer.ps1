[CmdletBinding()]
param(
    [string]$AppVersion = "dev",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$SdkRoot = Resolve-Path (Join-Path $ScriptDir "..")

Write-Host "=== agent-learning Windows installer build ===" -ForegroundColor Cyan
Write-Host "SDK root:    $SdkRoot"
Write-Host "App version: $AppVersion"
Write-Host "Skip tests:  $SkipTests"

Push-Location $SdkRoot
try {
    python -m pip install --upgrade pip

    if (-not $SkipTests) {
        python -m pip install -e ".[dev]"
        python -m pytest -q
    }

    python -m pip install pyinstaller

    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "dist-installer") { Remove-Item -Recurse -Force "dist-installer" }

    python -m PyInstaller `
        --clean `
        --noconfirm `
        --onefile `
        --name "agent-learn" `
        "packaging/windows/agent_learning_entry.py"

    $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) {
        throw "Inno Setup not found at '$iscc'. Install Inno Setup 6 and retry."
    }

    & $iscc "/DAppVersion=$AppVersion" "packaging/windows/agent-learning-installer.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compiler failed with exit code $LASTEXITCODE."
    }

    Write-Host "Built installer artifacts in dist-installer/" -ForegroundColor Green
}
finally {
    Pop-Location
}
