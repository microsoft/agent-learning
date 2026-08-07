# scripts/publish.ps1
#
# Build and publish agent-learning to PyPI / TestPyPI.
#
# Usage:
#   .\scripts\publish.ps1                       # build only (no upload)
#   .\scripts\publish.ps1 -Target testpypi      # upload to TestPyPI
#   .\scripts\publish.ps1 -Target pypi          # upload to real PyPI
#   .\scripts\publish.ps1 -Target testpypi -SkipTests
#
# Credentials (use ONE of):
#   - Env var TWINE_PASSWORD set to your API token (TWINE_USERNAME defaults to __token__).
#   - ~/.pypirc with [pypi] / [testpypi] sections.
#   - For CI, prefer PyPI Trusted Publishers (see .github/workflows/publish.yaml).
#
# IMPORTANT: PyPI may reserve the "azure-*" namespace for Microsoft. Verify
# you can register the project name (try TestPyPI first) before pushing to
# production PyPI.

[CmdletBinding()]
param(
    [ValidateSet('none', 'testpypi', 'pypi')]
    [string]$Target = 'none',

    [switch]$SkipTests,

    [switch]$NoClean
)

$ErrorActionPreference = 'Stop'

# Locate the SDK root (parent of this script's directory)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$SdkRoot   = Resolve-Path (Join-Path $ScriptDir '..')

Write-Host "=== agent-learning publish ===" -ForegroundColor Cyan
Write-Host "SDK root:   $SdkRoot"
Write-Host "Target:     $Target"
Write-Host "Skip tests: $SkipTests"
Write-Host "No clean:   $NoClean"
Write-Host ""

Push-Location $SdkRoot
try {
    # Pick a python interpreter
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) { $python = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
    if (-not $python) { throw "No python interpreter found on PATH." }
    Write-Host "Using python: $python" -ForegroundColor DarkGray

    # 1. Clean previous artifacts
    if (-not $NoClean) {
        Write-Host "`n[1/5] Cleaning dist/, build/, *.egg-info ..." -ForegroundColor Yellow
        foreach ($p in @('dist', 'build')) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Path 'src' -Recurse -Directory -Filter '*.egg-info' -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-Item -Recurse -Force $_.FullName
        }
    }

    # 2. Install/upgrade build tooling
    Write-Host "`n[2/5] Ensuring build + twine are installed ..." -ForegroundColor Yellow
    & $python -m pip install --upgrade pip build twine
    if ($LASTEXITCODE -ne 0) { throw "pip install build+twine failed." }

    # 3. Run tests (optional)
    if (-not $SkipTests) {
        Write-Host "`n[3/5] Running pytest ..." -ForegroundColor Yellow
        & $python -m pip install --quiet -e ".[dev]"
        if ($LASTEXITCODE -ne 0) { throw "pip install dev extras failed." }
        & $python -m pytest tests
        if ($LASTEXITCODE -ne 0) { throw "Tests failed. Re-run with -SkipTests to bypass." }
    } else {
        Write-Host "`n[3/5] Skipping tests (-SkipTests)." -ForegroundColor DarkYellow
    }

    # 4. Build sdist + wheel
    Write-Host "`n[4/5] Building distributions ..." -ForegroundColor Yellow
    & $python -m build
    if ($LASTEXITCODE -ne 0) { throw "build failed." }

    Write-Host "`nArtifacts:" -ForegroundColor DarkGray
    Get-ChildItem -Path 'dist' | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length / 1KB)) KB)" }

    Write-Host "`nValidating metadata with twine check ..." -ForegroundColor DarkGray
    & $python -m twine check dist/*
    if ($LASTEXITCODE -ne 0) { throw "twine check failed." }

    # 5. Upload
    switch ($Target) {
        'none' {
            Write-Host "`n[5/5] Build complete. No upload requested." -ForegroundColor Green
            Write-Host "    Re-run with -Target testpypi or -Target pypi to publish."
        }
        'testpypi' {
            Write-Host "`n[5/5] Uploading to TestPyPI ..." -ForegroundColor Yellow
            if (-not $env:TWINE_USERNAME) { $env:TWINE_USERNAME = '__token__' }
            if (-not $env:TWINE_PASSWORD) {
                Write-Host "TWINE_PASSWORD not set; twine will fall back to ~/.pypirc." -ForegroundColor DarkGray
            }
            & $python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
            if ($LASTEXITCODE -ne 0) { throw "TestPyPI upload failed." }
            Write-Host "`nPublished. View at:" -ForegroundColor Green
            Write-Host "  https://test.pypi.org/project/agent-learning/"
        }
        'pypi' {
            Write-Host "`n[5/5] Uploading to PyPI (production) ..." -ForegroundColor Yellow
            if (-not $env:TWINE_USERNAME) { $env:TWINE_USERNAME = '__token__' }
            if (-not $env:TWINE_PASSWORD) {
                Write-Host "TWINE_PASSWORD not set; twine will fall back to ~/.pypirc." -ForegroundColor DarkGray
            }
            & $python -m twine upload dist/*
            if ($LASTEXITCODE -ne 0) { throw "PyPI upload failed." }
            Write-Host "`nPublished. View at:" -ForegroundColor Green
            Write-Host "  https://pypi.org/project/agent-learning/"
        }
    }
}
finally {
    Pop-Location
}
