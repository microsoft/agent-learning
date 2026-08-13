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
        --onedir `
        --specpath "build" `
        --name "agent-learn" `
        "packaging/windows/agent_learning_entry.py"

    $isccCandidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    $iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) {
        $iscc = (Get-Command "ISCC.exe" -ErrorAction SilentlyContinue).Source
    }
    if (-not $iscc) {
        throw "Inno Setup 6 not found. Install it and retry."
    }

    & $iscc "/DAppVersion=$AppVersion" "packaging/windows/agent-learning-installer.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compiler failed with exit code $LASTEXITCODE."
    }

    $zipPath = "dist-installer/agent-learning-cli-$AppVersion-windows-x64.zip"
    Compress-Archive -Path "dist/agent-learn/*" -DestinationPath $zipPath -Force

    Write-Host "Built installer artifacts in dist-installer/" -ForegroundColor Green
}
finally {
    Pop-Location
}
