param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$envFile = Join-Path $ProjectRoot ".env"

function Get-TokenFromEnvFile([string]$Path) {
    if (-not (Test-Path $Path)) {
        return ""
    }

    foreach ($line in Get-Content -Path $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -match '^(HF_TOKEN|HUGGINGFACE_HUB_TOKEN)\s*=\s*(.+)$') {
            $value = $Matches[2].Trim().Trim('"').Trim("'")
            if (-not [string]::IsNullOrWhiteSpace($value)) {
                return $value
            }
        }
    }
    return ""
}

function Upsert-EnvLine([string]$Path, [string]$Key, [string]$Value) {
    $lines = @()
    if (Test-Path $Path) {
        $lines = Get-Content -Path $Path -Encoding UTF8
    }

    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$Key\s*=") {
            $lines[$i] = "$Key=$Value"
            $updated = $true
        }
    }

    if (-not $updated) {
        if ($lines.Count -gt 0 -and $lines[-1].Trim() -ne "") {
            $lines += ""
        }
        $lines += "$Key=$Value"
    }

    Set-Content -Path $Path -Value $lines -Encoding UTF8
}

$token = Get-TokenFromEnvFile -Path $envFile
if ([string]::IsNullOrWhiteSpace($token)) {
    $token = $env:HF_TOKEN
}
if ([string]::IsNullOrWhiteSpace($token)) {
    $token = $env:HUGGINGFACE_HUB_TOKEN
}

if ([string]::IsNullOrWhiteSpace($token)) {
    $token = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
}
if ([string]::IsNullOrWhiteSpace($token)) {
    $token = [Environment]::GetEnvironmentVariable("HUGGINGFACE_HUB_TOKEN", "User")
}

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host ""
    Write-Host "HuggingFace LLM 토큰이 필요합니다." -ForegroundColor Yellow
    Write-Host "https://huggingface.co/settings/tokens 에서 토큰 생성 후 입력하세요." -ForegroundColor Yellow
    $token = Read-Host "HF_TOKEN 입력 (건너뛰려면 Enter)"
}

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "HF 토큰 입력이 건너뛰어졌습니다. 챗봇 LLM 기능은 비활성 상태입니다." -ForegroundColor Yellow
    exit 0
}

$token = $token.Trim()
$env:HF_TOKEN = $token
$env:HUGGINGFACE_HUB_TOKEN = $token

[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")
[Environment]::SetEnvironmentVariable("HUGGINGFACE_HUB_TOKEN", $token, "User")

if (-not (Test-Path $envFile)) {
    New-Item -Path $envFile -ItemType File -Force | Out-Null
}

Upsert-EnvLine -Path $envFile -Key "HF_TOKEN" -Value $token
Upsert-EnvLine -Path $envFile -Key "HUGGINGFACE_HUB_TOKEN" -Value $token

Write-Host "HF 토큰이 저장되었습니다. (User env + $envFile)" -ForegroundColor Green
