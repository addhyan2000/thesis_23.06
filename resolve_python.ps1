# Resolve a usable Python command on Windows (python OR py launcher).
# Dot-source from other scripts:  . "$PSScriptRoot\resolve_python.ps1"

function Get-MerPythonCommand {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            Executable = $python.Source
            ArgsPrefix = @()
            Display    = "python"
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($ver in @("-3.11", "-3.10", "-3.12", "-3")) {
            & py $ver -c "import sys" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return @{
                    Executable = "py"
                    ArgsPrefix = @($ver)
                    Display    = "py $ver"
                }
            }
        }
    }

    return $null
}

function Invoke-MerPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArgs
    )

    $cmd = Get-MerPythonCommand
    if (-not $cmd) {
        throw "Python not found. Install Python 3.11 or ensure 'py' launcher works."
    }

    if ($cmd.Executable -eq "py") {
        & py @($cmd.ArgsPrefix + $PythonArgs)
    } else {
        & $cmd.Executable @PythonArgs
    }
    return $LASTEXITCODE
}
