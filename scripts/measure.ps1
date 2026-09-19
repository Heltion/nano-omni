# cspell:ignore LASTEXITCODE
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param([Parameter(Mandatory)][string[]] $Arguments)

    & uv @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv exited with code $LASTEXITCODE"
    }
}

$instances = @(
    "h3-t2va-official-dense-bf16-4step",
    "h3-fl2va-official-dense-bf16-4step",
    "h3-ref2va-0.4mp-dense-bf16-4step"
)
$attentionStudies = @(
    "h3-fl2va-official-sol-bf16-4step",
    "h3-fl2va-official-dense-int8-fp8-4step",
    "h3-fl2va-official-sol-int8-fp8-4step",
    "h3-fl2va-official-dense-nvfp4-4step",
    "h3-fl2va-official-sol-nvfp4-4step"
)

Push-Location $root
try {
    & git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        throw "git submodule update exited with code $LASTEXITCODE"
    }

    foreach ($instance in $instances) {
        $config = "perf/$instance/config.yaml"
        Invoke-Checked -Arguments @("run", "scripts/run/nano-omni.py", "--config", $config)
        Invoke-Checked -Arguments @(
            "run", "--group", "experiment", "scripts/run/comfy-ui.py", "--config", $config
        )
        Invoke-Checked -Arguments @("run", "scripts/run/nano-omni.py", "--config", $config, "--nsys")
    }

    foreach ($instance in $attentionStudies) {
        $config = "perf/$instance/config.yaml"
        Invoke-Checked -Arguments @("run", "scripts/run/nano-omni.py", "--config", $config)
        Invoke-Checked -Arguments @("run", "scripts/run/nano-omni.py", "--config", $config, "--nsys")
    }

    Invoke-Checked -Arguments @("run", "python", "hooks/readme.py")
}
finally {
    Pop-Location
}
