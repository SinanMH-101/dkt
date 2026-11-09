<#
.SYNOPSIS
Train the DKT model on the prepared datasets and collect test AUCs.
.EXAMPLE
.\run-experiments.ps1
.EXAMPLE
.\run-experiments.ps1 -Dataset idetrace -Epochs 1
.EXAMPLE
.\run-experiments.ps1 -Dataset comp1350,idetrace -Epochs 10,20,50
.EXAMPLE
.\run-experiments.ps1 -Check
#>
[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [ValidateRange(1, [int]::MaxValue)]
    [int[]]$Epochs = @(50),
    [string[]]$Dataset,
    [switch]$Check,
    [switch]$NoCompiler,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$epochCheckpoints = @($Epochs | Sort-Object -Unique)
$maxEpoch = $epochCheckpoints[-1]
$experiments = Get-Content -LiteralPath "$PSScriptRoot/experiments.json" -Raw |
    ConvertFrom-Json
if ($Dataset) {
    $unknown = @($Dataset | Where-Object { $_ -notin $experiments.name })
    if ($unknown.Count) {
        throw "Unknown dataset: $($unknown -join ', '). Choose from $($experiments.name -join ', ')."
    }
    $experiments = @($experiments | Where-Object { $_.name -in $Dataset })
}

# Check every selected dataset before starting a potentially long training run.
$missing = foreach ($experiment in $experiments) {
    foreach ($file in @($experiment.dataset, $experiment.split)) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $file) -PathType Leaf)) {
            $file
        }
    }
}
if ($missing) {
    throw "Missing prepared files: $($missing -join ', '). Check your checkout or see data/README.md."
}
if ($Check) {
    Write-Host "Found prepared data and splits for $($experiments.name -join ', ')."
    return
}

& $Python -c "import sys; sys.exit(0 if sys.version_info[:2] == (2, 7) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw 'DKT requires Python 2.7. Run conda activate dkt-legacy first.'
}

$runId = '{0:yyyy-MM-dd_HH-mm-ss-fff}' -f (Get-Date)
$runDirectory = New-Item -ItemType Directory -Path "$PSScriptRoot/output/runs/$runId"
$summary = Join-Path $runDirectory.FullName 'summary.csv'
$previousBackend = $env:KERAS_BACKEND
$previousFlags = $env:THEANO_FLAGS
try {
    $env:KERAS_BACKEND = 'theano'
    $cache = (Join-Path $PSScriptRoot 'output/theano').Replace('\', '/')
    $env:THEANO_FLAGS = "device=cpu,floatX=float32,base_compiledir=$cache,gcc.cxxflags=-march=core2"
    if ($NoCompiler) {
        $env:THEANO_FLAGS = "device=cpu,floatX=float32,base_compiledir=$cache,cxx="
    }

    foreach ($experiment in $experiments) {
        # dkt.py writes beside its input, so train on a copy in the run folder.
        $folder = New-Item -ItemType Directory -Path (Join-Path $runDirectory.FullName $experiment.name)
        $source = Join-Path $PSScriptRoot $experiment.dataset
        $localDataset = Join-Path $folder.FullName (Split-Path $source -Leaf)
        Copy-Item -LiteralPath $source -Destination $localDataset
        $split = Join-Path $PSScriptRoot $experiment.split
        $trainingArgs = @(
            "$PSScriptRoot/dkt.py", '--dataset', $localDataset, '--splitfile', $split,
            '--epochs', $maxEpoch, '--hiddenunits', $experiment.hiddenUnits,
            '--batchsize', $experiment.batchSize, '--timewindow', $experiment.timeWindow
        )

        Write-Host "`nRunning $($experiment.name) for $maxEpoch epoch(s); reporting epoch(s) $($epochCheckpoints -join ', ')"
        & $Python -u @trainingArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $($experiment.name). Run folder: $($runDirectory.FullName)"
        }

        $history = @(Get-Content -LiteralPath "$localDataset.history")
        if ($history.Count -ne $maxEpoch) {
            throw "Expected $maxEpoch history rows for $($experiment.name), found $($history.Count). Run folder: $($runDirectory.FullName)"
        }
        $rows = foreach ($checkpoint in $epochCheckpoints) {
            $loss, $testAuc = $history[$checkpoint - 1] -split "`t"
            [pscustomobject]@{
                dataset = $experiment.name
                epochs = $checkpoint
                test_auc = $testAuc
                recorded_auc_epoch_50 = $experiment.recordedAuc
                training_loss = $loss
            }
        }
        $rows | Export-Csv -LiteralPath $summary -NoTypeInformation -Append
    }
}
finally {
    $env:KERAS_BACKEND = $previousBackend
    $env:THEANO_FLAGS = $previousFlags
}

Write-Host "`nResults: $summary"
$datasetFiles = @{}
foreach ($experiment in $experiments) {
    $datasetFiles[$experiment.name] = Split-Path $experiment.dataset -Leaf
}

$headers = @('Dataset', 'Completed epochs', 'Final test AUC')
$widths = @($headers | ForEach-Object { $_.Length })
$results = @(Import-Csv -LiteralPath $summary)
$tableRows = foreach ($result in $results) {
    $auc = [double]::Parse($result.test_auc, [Globalization.CultureInfo]::InvariantCulture)
    $cells = @($datasetFiles[$result.dataset], $result.epochs,
        $auc.ToString('F6', [Globalization.CultureInfo]::InvariantCulture))
    for ($i = 0; $i -lt $headers.Count; $i++) {
        $widths[$i] = [Math]::Max($widths[$i], $cells[$i].Length)
    }
    ,$cells
}

$border = '+' + (($widths | ForEach-Object { '-' * ($_ + 2) }) -join '+') + '+'
Write-Host $border
foreach ($row in (,@($headers) + @($tableRows))) {
    $padded = for ($i = 0; $i -lt $headers.Count; $i++) {
        $row[$i].PadRight($widths[$i])
    }
    Write-Host ('| ' + ($padded -join ' | ') + ' |')
    Write-Host $border
}
