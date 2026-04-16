<#
.SYNOPSIS
  Объединяет содержимое файлов в один текстовый файл с разделителями и полными путями.

  По умолчанию: только bt.py, config.py, etl.py, train.py (рядом с этим скриптом).

.EXAMPLE
  .\Export-FilesToTxt.ps1 -OutFile .\bundle.txt

.EXAMPLE
  .\Export-FilesToTxt.ps1 -Path .\scan.py -OutFile .\extra.txt
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, HelpMessage = "Файлы или папки; по умолчанию bt.py, config.py, etl.py, train.py")]
    [string[]]$Path = @("bt.py", "config.py", "etl.py", "train.py"),

    [Parameter(Mandatory = $true, Position = 1, HelpMessage = "Итоговый .txt")]
    [string]$OutFile,

    [string]$Filter = "*",
    [switch]$Recurse,
    [string[]]$ExcludeDirectory = @(".git", ".venv", "node_modules", "__pycache__", ".cursor"),

    [ValidateSet("UTF8", "UTF8BOM")]
    [string]$Encoding = "UTF8"
)

$ErrorActionPreference = "Stop"
$sep = "=" * 78

# Относительные пути — от каталога скрипта (чтобы работало из любой cwd)
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Path = foreach ($p in $Path) {
    if ([IO.Path]::IsPathRooted($p)) {
        [IO.Path]::GetFullPath($p)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $scriptRoot $p))
    }
}

function Get-FilesFromInput {
    param([string[]]$Inputs, [string]$Filter, [bool]$Recurse, [string[]]$ExcludeDirectory)

    $list = [System.Collections.Generic.List[string]]::new()
    foreach ($p in $Inputs) {
        $item = Get-Item -LiteralPath $p -ErrorAction Stop
        if ($item.PSIsContainer) {
            $gciParams = @{
                Path          = $item.FullName
                File          = $true
                Filter        = $Filter
                ErrorAction   = "Stop"
            }
            if ($Recurse) {
                $gciParams["Recurse"] = $true
            }
            $files = Get-ChildItem @gciParams | Where-Object {
                $parts = $_.FullName.Split(
                    [char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar),
                    [StringSplitOptions]::RemoveEmptyEntries)
                $skip = $false
                foreach ($ex in $ExcludeDirectory) {
                    if ($parts -contains $ex) { $skip = $true; break }
                }
                -not $skip
            }
            foreach ($f in $files) { $list.Add($f.FullName) }
        }
        else {
            $list.Add($item.FullName)
        }
    }
    $list | Select-Object -Unique | Sort-Object
}

$files = Get-FilesFromInput -Inputs $Path -Filter $Filter -Recurse:$Recurse -ExcludeDirectory $ExcludeDirectory
if ($files.Count -eq 0) {
    Write-Warning "Нет файлов для экспорта."
    exit 1
}

$enc = if ($Encoding -eq "UTF8BOM") {
    New-Object System.Text.UTF8Encoding($true)
}
else {
    New-Object System.Text.UTF8Encoding($false)
}

$fullOut = [IO.Path]::GetFullPath($OutFile)
$outDir = Split-Path -Parent $fullOut
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$sb = [System.Text.StringBuilder]::new()
$null = $sb.AppendLine("EXPORT: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$null = $sb.AppendLine("FILES: $($files.Count)")
$null = $sb.AppendLine($sep)
$null = $sb.AppendLine()

foreach ($filePath in $files) {
    $null = $sb.AppendLine($sep)
    $null = $sb.AppendLine("PATH: $filePath")
    $null = $sb.AppendLine($sep)
    $null = $sb.AppendLine()
    try {
        $content = [System.IO.File]::ReadAllText($filePath, $enc)
        $null = $sb.AppendLine($content)
    }
    catch {
        $null = $sb.AppendLine("[READ ERROR: $($_.Exception.Message)]")
    }
    $null = $sb.AppendLine()
    $null = $sb.AppendLine()
}

[System.IO.File]::WriteAllText($fullOut, $sb.ToString(), $enc)

Write-Host "Готово: $fullOut ($($files.Count) файлов)"
