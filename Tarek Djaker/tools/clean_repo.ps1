param(
  [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Write-Host "Cleaning notebooks and archiving outputs under: $Root" -ForegroundColor Cyan

# Strip outputs and execution counts from all notebooks
$ipynbs = Get-ChildItem -Path $Root -Recurse -Filter *.ipynb -File
foreach ($nb in $ipynbs) {
  try {
    $json = Get-Content -Raw -LiteralPath $nb.FullName | ConvertFrom-Json
  } catch {
    Write-Warning "Skipping invalid JSON: $($nb.FullName)"; continue
  }
  if ($null -ne $json.cells) {
    foreach ($cell in $json.cells) {
      if ($null -ne $cell.outputs) { $cell.outputs = @() }
      if ($cell.PSObject.Properties.Name -contains 'execution_count') { $cell.execution_count = $null }
    }
  }
  $json | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $nb.FullName -Encoding UTF8
  Write-Host "Cleaned: $($nb.FullName)" -ForegroundColor DarkGray
}

# Archive generated CSVs
$archive = Join-Path $Root 'archive_outputs'
if (-not (Test-Path $archive)) { New-Item -ItemType Directory -Path $archive | Out-Null }

$targets = Get-ChildItem -Path $Root -Recurse -Include submission.csv, predictions.csv -File
foreach ($f in $targets) {
  $rel = $f.FullName.Substring($Root.Length).TrimStart('\\')
  $safe = ($rel -replace '\\','_')
  $dest = Join-Path $archive $safe
  Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
  Remove-Item -LiteralPath $f.FullName -Force
  Write-Host "Archived: $rel -> archive_outputs/$safe" -ForegroundColor DarkGray
}

Write-Host "Done." -ForegroundColor Green

