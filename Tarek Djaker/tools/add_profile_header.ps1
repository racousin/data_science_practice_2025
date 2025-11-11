param(
  [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")),
  [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Write-Host "Inserting profile header cell into notebooks under: $Root" -ForegroundColor Cyan

function Add-HeaderCell([string]$path) {
  try { $nb = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json } catch { return $false }
  if (-not $nb.cells) { return $false }
  # Build the header cell content
  $libPath = (Resolve-Path (Join-Path $PSScriptRoot "..\lib")).Path
  $src = @(
    "# --- Tarek Djaker notebook profile ---",
    "import sys, os",
    "sys.path.append(r'${libPath}')",
    "from tarek_profile import nb_init, profile_banner",
    "nb_init()",
    "profile_banner(title=None)",
    "# -------------------------------------"
  ) -join "`n"

  # Detect if header already present
  foreach ($cell in $nb.cells) {
    if ($cell.cell_type -eq 'code' -and $cell.source -join '' -like '*tarek_profile*') { return $false }
  }

  # Prepend the new cell
  $header = [pscustomobject]@{cell_type='code'; execution_count=$null; metadata=@{}; outputs=@(); source=@($src)}
  $nb.cells = @($header) + @($nb.cells)
  if (-not $WhatIf) { $nb | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $path -Encoding UTF8 }
  return $true
}

$count = 0
Get-ChildItem -Path $Root -Recurse -Filter *.ipynb -File | ForEach-Object {
  if (Add-HeaderCell -path $_.FullName) { $count++ ; Write-Host "Updated: $($_.FullName)" -ForegroundColor DarkGray }
}

Write-Host "Done. Notebooks updated: $count" -ForegroundColor Green

