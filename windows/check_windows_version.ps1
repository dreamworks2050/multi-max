# Windows Version Check Script
# This script checks if the installed Multi-Max version is properly configured for Windows

Write-Host "==== Multi-Max Windows Version Check ====" -ForegroundColor Cyan
Write-Host

# Verify we're running on Windows
if (-not ($env:OS -like "*Windows*")) {
    Write-Host "WARNING: This script is designed for Windows but appears to be running on a different platform." -ForegroundColor Yellow
    Write-Host "Some functionality may not work as expected." -ForegroundColor Yellow
    Write-Host
}

# Get the proper parent directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ScriptDir
$mainFile = Join-Path -Path $ParentDir -ChildPath "main.py"

# Check if main.py exists
if (-not (Test-Path $mainFile)) {
    Write-Host "ERROR: main.py not found. Multi-Max may not be installed correctly." -ForegroundColor Red
    Write-Host "Please run the Windows installer first." -ForegroundColor Yellow
    exit 1
}

# Check if main.py contains the Windows marker
$windowsMarker = Select-String -Path $mainFile -Pattern "__windows_specific_version__" -Quiet
if ($windowsMarker) {
    Write-Host "SUCCESS: Windows-specific version is correctly installed." -ForegroundColor Green
} else {
    Write-Host "ERROR: The installed version does not appear to be the Windows-specific version." -ForegroundColor Red
    Write-Host "This may cause issues when running on Windows." -ForegroundColor Red
    
    # Check if we have a Windows version available
    $windowsMainFile = Join-Path -Path $ScriptDir -ChildPath "main.py"
    if (Test-Path $windowsMainFile) {
        Write-Host "Found Windows-specific version in the windows folder." -ForegroundColor Yellow
        Write-Host "Would you like to install it now? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            # Backup original file if it exists
            if (Test-Path $mainFile) {
                $backupFile = "$mainFile.backup"
                Copy-Item -Path $mainFile -Destination $backupFile -Force
                Write-Host "Backed up existing main.py to main.py.backup" -ForegroundColor Green
            }
            Copy-Item -Path $windowsMainFile -Destination $mainFile -Force
            Write-Host "Windows version installed successfully." -ForegroundColor Green
        }
    } else {
        Write-Host "No Windows-specific version found in the windows folder." -ForegroundColor Red
        Write-Host "Please download the correct version from the repository." -ForegroundColor Yellow
    }
}

# Check environment configuration
$envFile = Join-Path -Path $ParentDir -ChildPath ".env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
    
    # Check for Windows settings
    $hardwareAcceleration = $envContent -match "FORCE_HARDWARE_ACCELERATION=false"
    $softwareFallback = $envContent -match "ALLOW_SOFTWARE_FALLBACK=true"
    $windowsMode = $envContent -match "WINDOWS_MODE=true"
    
    if ($hardwareAcceleration -and $softwareFallback -and $windowsMode) {
        Write-Host "Environment configuration is correctly set up for Windows." -ForegroundColor Green
    } else {
        Write-Host "WARNING: Environment configuration may not be optimized for Windows." -ForegroundColor Yellow
        Write-Host "Would you like to apply the Windows-specific environment settings? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            $windowsEnvFile = Join-Path -Path $ScriptDir -ChildPath ".env"
            if (Test-Path $windowsEnvFile) {
                # Backup original env if it exists
                if (Test-Path $envFile) {
                    $backupEnvFile = "$envFile.backup"
                    Copy-Item -Path $envFile -Destination $backupEnvFile -Force
                    Write-Host "Backed up existing .env to .env.backup" -ForegroundColor Green
                }
                Copy-Item -Path $windowsEnvFile -Destination $envFile -Force
                Write-Host "Windows environment settings applied successfully." -ForegroundColor Green
            } else {
                Write-Host "No Windows-specific .env file found in the windows folder." -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "No .env file found. Environment may not be configured correctly." -ForegroundColor Yellow
    
    # Try to create one from the template in the windows folder
    $windowsEnvFile = Join-Path -Path $ScriptDir -ChildPath ".env"
    if (Test-Path $windowsEnvFile) {
        Write-Host "Would you like to create a new .env file from the Windows template? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -eq "Y" -or $response -eq "y") {
            Copy-Item -Path $windowsEnvFile -Destination $envFile -Force
            Write-Host "Created new .env file from Windows template." -ForegroundColor Green
        }
    }
}

Write-Host
Write-Host "==== Check Complete ====" -ForegroundColor Cyan
Write-Host

# Pause so the user can read the results
Write-Host "Press any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 