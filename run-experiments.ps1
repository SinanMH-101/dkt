<#
.SYNOPSIS
Train the DKT model on the prepared datasets and collect test AUCs.
.EXAMPLE
.\run-experiments.ps1
.EXAMPLE
.\run-experiments.ps1 -Dataset idetrace -Epochs 1
.EXAMPLE
.\run-experiments.ps1 -Dataset idetrace -Epochs 1 --gemini-judge
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
    [switch]$GeminiJudge,
    [string]$Python = 'python',
    [Parameter(ValueFromRemainingArguments = $true)]
    [ValidateSet('--gemini-judge')]
    [string[]]$ExtraOptions
)

$ErrorActionPreference = 'Stop'
$reviewWithGemini = $GeminiJudge -or ($ExtraOptions -contains '--gemini-judge')
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

# Check inputs before training. A default run can skip data absent from a fresh checkout.
$missing = @()
$experiments = @(foreach ($experiment in $experiments) {
    $missingInputs = @(@($experiment.dataset, $experiment.split) | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $_) -PathType Leaf)
    })
    if (-not $Dataset -and $experiment.requiresLocalPreparation -and $missingInputs.Count -eq 2) {
        Write-Host "Skipping $($experiment.name). See data/README.md to prepare its files locally."
        continue
    }
    $missing += $missingInputs
    $experiment
})
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
        $datasetArgs = @(
            '--dataset', $localDataset, '--splitfile', $split,
            '--hiddenunits', $experiment.hiddenUnits,
            '--batchsize', $experiment.batchSize, '--timewindow', $experiment.timeWindow
        )

        Write-Host "`nRunning $($experiment.name) for $maxEpoch epoch(s); reporting epoch(s) $($epochCheckpoints -join ', ')"
        & $Python -u "$PSScriptRoot/dkt.py" --epochs $maxEpoch @datasetArgs
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

        if ($reviewWithGemini) {
            try {
                & py -3 "$PSScriptRoot/scripts/judge_gemini.py" @datasetArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Judge exited with code $LASTEXITCODE."
                }
            }
            catch {
                Write-Warning "Gemini review failed for $($experiment.name). Training results are saved in $($folder.FullName). $($_.Exception.Message)"
            }
        }
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
