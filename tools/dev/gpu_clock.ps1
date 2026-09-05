<#
.SYNOPSIS
    Pin this machine's GPU clocks so measure_gpu.py's drift gate can pass.

.DESCRIPTION
    tools/dev/measure_gpu.py refuses an A/B comparison when the repeated control
    moves more than 1% across the series. On this laptop RTX 3080 Ti it moves far
    more than that on its own: the graphics clock ranges from 405 MHz to 2100 MHz
    and the memory clock hops between 6001, 7001 and 8001 MHz, with no
    user-settable power limit. A sustained load throttles rather than settling at
    a steady boost -- and a throttling clock IS a drifting control. Heavier scenes
    make it worse, not better; see docs/profiling.md for the measurements.

    Pinning both clocks removes the drift at its source. That needs administrator
    rights, so this script re-launches itself elevated and asks once via UAC
    rather than making you open an admin shell by hand.

    A pin is a CEILING, not a floor. Measured on this card: with the graphics
    clock pinned at 1400 MHz a `--scene stress` A/B passed the gate at 0.30%
    drift -- the first time it ever had. The same A/B on `--scene gpu-stress`
    still drifted 9.0%, because that scene drives the card to 87 C,
    `sw_thermal_slowdown` goes Active, and the clock falls below the pin anyway
    (1282-1402 MHz observed). Heavier load therefore needs a LOWER pin, not a
    higher one.

    The pin does NOT survive a reboot or a driver restart. Re-run it after
    either, and run `unlock` when you are done measuring -- pinned clocks cap the
    card for everything else too, games included.

.PARAMETER Action
    lock    Pin the graphics and memory clocks (see -Mhz and -MemMhz).
    unlock  Return the card to its normal boost behaviour.
    status  Print the current clocks, temperature, and whether a pin is holding.

.PARAMETER Mhz
    Graphics clock to pin, in MHz. Default 1400.

    The value wants to be one the card can hold indefinitely under the load you
    are measuring -- high enough that the frame is representative, low enough
    that it never reaches the thermal threshold. Too high and you are back to a
    drifting control; too low and you are measuring a card that is not the one
    you ship on. 1400 holds for `--scene stress`; it does not hold for
    `--scene gpu-stress`, which needs roughly 1100. Run `status` during a
    measurement: if sw_thermal_slowdown is Active, come down a step.

.PARAMETER MemMhz
    Memory clock to pin, in MHz. Default 7001, the middle of the three clocks
    this card actually uses under load. Supported here: 405, 810, 6001, 7001,
    8001 -- nvidia-smi rejects anything else. This one matters more than it
    looks on bandwidth-bound scenes, where an unpinned 6001<->8001 hop is a 33%
    swing in the only resource the frame is waiting on.

.EXAMPLE
    powershell -File tools/dev/gpu_clock.ps1 lock
    powershell -File tools/dev/gpu_clock.ps1 lock -Mhz 1100
    powershell -File tools/dev/gpu_clock.ps1 status
    python tools/dev/measure_gpu.py ab --b-set ... --args --scene gpu-stress
    powershell -File tools/dev/gpu_clock.ps1 unlock
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('lock', 'unlock', 'status')]
    [string]$Action = 'status',

    [ValidateRange(200, 3000)]
    [int]$Mhz = 1400,

    [ValidateRange(200, 12000)]
    [int]$MemMhz = 7001
)

$ErrorActionPreference = 'Stop'

function Get-NvidiaSmi {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $fallback = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path $fallback) { return $fallback }
    throw 'nvidia-smi not found. This script only applies to NVIDIA GPUs.'
}

function Test-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

$smi = Get-NvidiaSmi

# status needs no privileges, so it never prompts.
if ($Action -eq 'status') {
    & $smi --query-gpu=name,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw `
           --format=csv
    Write-Host ''
    $reasons = & $smi --query-gpu=clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sw_power_cap `
                      --format=csv,noheader
    Write-Host "Throttle reasons (hw_thermal, sw_thermal, sw_power_cap): $reasons"
    if ($reasons -match 'Active' -and $reasons -notmatch '^(Not Active, )*Not Active$') {
        Write-Host 'A throttle reason is Active: the card is being held below whatever you pinned.' -ForegroundColor Yellow
        Write-Host 'Re-pin lower (-Mhz) or the drift gate will still refuse the comparison.' -ForegroundColor Yellow
    }
    return
}

# lock and unlock write to the device, so they need administrator. Re-launch
# self rather than telling the reader to go find an admin shell; -Verb RunAs is
# what raises the UAC prompt.
if (-not (Test-Elevated)) {
    Write-Host "Re-launching elevated to change GPU clocks (one UAC prompt)..."
    $self = $PSCommandPath
    $arguments = @(
        '-NoProfile'
        '-ExecutionPolicy', 'Bypass'
        '-File', "`"$self`""
        $Action
    )
    if ($Action -eq 'lock') { $arguments += @('-Mhz', $Mhz, '-MemMhz', $MemMhz) }

    # The elevated child's output cannot be captured: -Verb and
    # -RedirectStandardOutput are different Start-Process parameter sets
    # (elevation needs UseShellExecute, redirection forbids it), and asking for
    # both fails to bind at all. Its console closes on exit, so read the result
    # back from the device afterwards instead -- which is what we wanted to
    # know anyway.
    $process = Start-Process -FilePath (Get-Process -Id $PID).Path `
                             -ArgumentList $arguments -Verb RunAs -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw ("Elevated run failed with exit code $($process.ExitCode). Its window " +
               "closed too fast to show why; re-run the same command from an " +
               "Administrator PowerShell to see the error.")
    }

    & $PSCommandPath status
    return
}

if ($Action -eq 'lock') {
    Write-Host "Pinning graphics clock to $Mhz MHz and memory clock to $MemMhz MHz..."
    & $smi --lock-gpu-clocks=$Mhz,$Mhz
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi --lock-gpu-clocks failed ($LASTEXITCODE)." }
    & $smi --lock-memory-clocks=$MemMhz,$MemMhz
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi --lock-memory-clocks failed ($LASTEXITCODE)." }
    Start-Sleep -Seconds 1
    Write-Host 'Pinned. This does not survive a reboot or a driver restart, and it caps the card for everything else until unlocked.'
}
else {
    Write-Host 'Releasing both pins...'
    & $smi --reset-gpu-clocks
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi --reset-gpu-clocks failed ($LASTEXITCODE)." }
    & $smi --reset-memory-clocks
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi --reset-memory-clocks failed ($LASTEXITCODE)." }
    Write-Host 'Released.'
}
