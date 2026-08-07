<#
.SYNOPSIS
    run_animation.ps1 — Set up a Python venv, install dependencies, and
    render the Agent Learning animation with Manim.

.DESCRIPTION
    PowerShell port of run_animation.sh. Creates a local virtual
    environment, installs the pinned requirements, and renders the
    three-act "AgentLearning" scene from animation.py that visualises
    the SDK's policy → judges → learner loop.

.PARAMETER Quality
    Manim render quality: l (480p15), m (720p30), h (1080p60),
    p (1440p60) or k (2160p60). Defaults to 'l'.

.PARAMETER Scene
    The Manim Scene class to render. Defaults to 'AgentLearning'.

.PARAMETER Script
    The Manim script file. Defaults to 'animation.py'.

.PARAMETER Preview
    Open the rendered video when the render finishes (manim -p).

.EXAMPLE
    .\run_animation.ps1
    .\run_animation.ps1 -Quality h -Preview
#>

[CmdletBinding()]
param(
    [ValidateSet('l', 'm', 'h', 'p', 'k')]
    [string]$Quality = 'l',
    [string]$Scene = 'AgentLearning',
    [string]$Script = 'animation.py',
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'

# Relax the execution policy for this process so an activated venv (and
# this script) are allowed to run without changing machine-wide policy.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $ScriptDir '.venv'
$ReqFile = Join-Path $ScriptDir 'requirements.txt'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

function Write-Step { param([string]$Message) Write-Host $Message -ForegroundColor Cyan }
function Write-Ok { param([string]$Message) Write-Host $Message -ForegroundColor Green }

# ── 1. Create the venv if it doesn't exist ─────────────────────────
if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating Python virtual environment in $VenvDir ..."
    python -m venv $VenvDir
}

# ── 2. Install / upgrade requirements ──────────────────────────────
# Call the venv interpreter directly rather than activating, which is
# the most robust path on locked-down Windows shells.
Write-Step "Installing requirements from $ReqFile ..."
& $VenvPython -m pip install --quiet --upgrade pip
& $VenvPython -m pip install --quiet -r $ReqFile

# ── 3. Render the animation ────────────────────────────────────────
$ScriptPath = Join-Path $ScriptDir $Script
Write-Host ''
Write-Ok "Rendering: $Script → scene $Scene (quality $Quality)"
Write-Ok "Output will be in $ScriptDir\media\"
Write-Host ''

$manimArgs = @('-m', 'manim', '--quality', $Quality)
if ($Preview) { $manimArgs += '-p' }
$manimArgs += @($ScriptPath, $Scene)

Push-Location $ScriptDir
try {
    & $VenvPython @manimArgs
}
finally {
    Pop-Location
}
