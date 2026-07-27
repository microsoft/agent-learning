#!/usr/bin/env pwsh
# =====================================================================
# Pre-provision hook (Windows / pwsh)
# 1. Ensures AZURE_LOCATION has a sensible default.
# 2. Ensures an SSH public key exists for the VM administrator, generating
#    one under ~/.ssh if the azd env doesn't already carry AZURE_SSH_PUBLIC_KEY.
# =====================================================================
param()
$ErrorActionPreference = 'Stop'

# 1) Default region.
$location = if ($env:AZURE_LOCATION) { $env:AZURE_LOCATION } else { 'eastus2' }
azd env set AZURE_LOCATION $location | Out-Null
Write-Host "AZURE_LOCATION=$location"

# 2) Ensure an SSH public key for the VM administrator.
$envValues = @{}
foreach ($line in (azd env get-values)) {
  if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
    $envValues[$matches[1]] = $matches[2].Trim().Trim('"')
  }
}

if (-not [string]::IsNullOrWhiteSpace($envValues['AZURE_SSH_PUBLIC_KEY'])) {
  Write-Host 'AZURE_SSH_PUBLIC_KEY already set; skipping key generation.'
  return
}

$envName = if ($env:AZURE_ENV_NAME) { $env:AZURE_ENV_NAME } else { 'agent-learning' }
$sshDir = Join-Path $HOME '.ssh'
$keyPath = Join-Path $sshDir "agent-learning-$envName"

try {
  if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Path $sshDir -Force | Out-Null }
  if (-not (Test-Path "$keyPath.pub")) {
    Write-Host "Generating SSH key pair at $keyPath ..."
    ssh-keygen -t rsa -b 4096 -f $keyPath -N '""' -C 'agent-learning-vm' | Out-Null
  }
  $pub = (Get-Content "$keyPath.pub" -Raw).Trim()
  azd env set AZURE_SSH_PUBLIC_KEY "$pub" | Out-Null
  Write-Host "SSH public key stored in azd env. Private key kept at: $keyPath"
}
catch {
  Write-Warning "Could not auto-generate an SSH key: $($_.Exception.Message)"
  Write-Warning 'Provide one manually and re-run:  azd env set AZURE_SSH_PUBLIC_KEY "<your-openssh-public-key>"'
  throw
}
