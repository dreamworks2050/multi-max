# Multi-Max Installation Fix Script for Windows
# This script fixes common issues with the Multi-Max installation

$ErrorActionPreference = "Stop"

# Banner
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  Multi-Max Windows Installation Fix Tool" -ForegroundColor Cyan
Write-Host "=================================================`n" -ForegroundColor Cyan

# Helper functions
function Write-Step {
    param([string]$message)
    Write-Host "`n>> $message" -ForegroundColor Green
}

function Test-Command {
    param([string]$command)
    return (Get-Command $command -ErrorAction SilentlyContinue) -ne $null
}

# Get current directory and set destination paths
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
$parentDir = Split-Path -Parent $scriptDir
$repoDir = $parentDir

Write-Host "Script directory: $scriptDir"
Write-Host "Repository directory: $repoDir"

# Check prerequisites
Write-Step "Checking prerequisites..."

# Check for Python
$pythonInstalled = Test-Command "python"
if (-not $pythonInstalled) {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.7 or higher from https://www.python.org/downloads/"
    Write-Host "Be sure to check 'Add Python to PATH' during installation."
    exit 1
}

$pythonVersion = python -c "import platform; print(platform.python_version())"
Write-Host "Python version: $pythonVersion" -ForegroundColor Green

# Check for Git (optional)
$gitInstalled = Test-Command "git"
if (-not $gitInstalled) {
    Write-Host "WARNING: Git is not installed. Update features will not work." -ForegroundColor Yellow
} else {
    Write-Host "Git is installed." -ForegroundColor Green
}

# Gather system information
Write-Step "Gathering system information..."
$os = [Environment]::OSVersion
$winVer = "$($os.Version.Major).$($os.Version.Minor).$($os.Version.Build)"
Write-Host "Windows version: $winVer" -ForegroundColor Green
Write-Host "Current directory: $pwd" -ForegroundColor Green

# Check installation structure
Write-Step "Checking installation structure..."

# Create logs directory
$logsDir = Join-Path -Path $repoDir -ChildPath "logs"
if (-not (Test-Path $logsDir)) {
    Write-Host "Logs directory not found. Creating..."
    New-Item -Path $logsDir -ItemType Directory | Out-Null
} else {
    Write-Host "Logs directory exists." -ForegroundColor Green
}

# Create VERSION file
$versionFile = Join-Path -Path $repoDir -ChildPath "VERSION"
if (-not (Test-Path $versionFile)) {
    Write-Host "VERSION file not found. Creating..."
    "1.0.0" | Out-File -FilePath $versionFile -Encoding utf8
} else {
    Write-Host "VERSION file exists." -ForegroundColor Green
}

# Repair main.py
$mainPy = Join-Path -Path $repoDir -ChildPath "main.py"
$windowsMainPy = Join-Path -Path $scriptDir -ChildPath "main.py"

if (-not (Test-Path $mainPy)) {
    Write-Host "ERROR: main.py not found in repository root." -ForegroundColor Red
    
    # Try to repair by copying from windows directory
    if (Test-Path $windowsMainPy) {
        Write-Host "Found Windows main.py. Copying to repository root..."
        Copy-Item -Path $windowsMainPy -Destination $mainPy
        Write-Host "main.py copied successfully." -ForegroundColor Green
    } else {
        Write-Host "ERROR: Windows main.py not found either. Cannot repair." -ForegroundColor Red
    }
} else {
    # Check if it contains Windows marker
    $content = Get-Content -Path $mainPy -Raw
    if (-not ($content -match "__windows_specific_version__")) {
        Write-Host "main.py doesn't contain Windows marker." -ForegroundColor Yellow
        
        # Backup and replace
        if (Test-Path $windowsMainPy) {
            Write-Host "Backing up current main.py and replacing with Windows version..."
            Copy-Item -Path $mainPy -Destination "$mainPy.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
            Copy-Item -Path $windowsMainPy -Destination $mainPy
            Write-Host "main.py replaced with Windows version." -ForegroundColor Green
        } else {
            Write-Host "ERROR: Windows main.py not found. Cannot replace." -ForegroundColor Red
        }
    } else {
        Write-Host "main.py contains Windows marker. Looks good." -ForegroundColor Green
    }
}

# Repair update checker
$updateCheckerPy = Join-Path -Path $repoDir -ChildPath "update_checker.py"
$simpleUpdateCheckerPy = Join-Path -Path $scriptDir -ChildPath "simple_update_checker.py"

if (-not (Test-Path $updateCheckerPy)) {
    Write-Host "update_checker.py not found in repository root." -ForegroundColor Yellow
    
    # Try to copy the simplified version
    if (Test-Path $simpleUpdateCheckerPy) {
        Write-Host "Found simplified update checker. Copying to repository root..."
        Copy-Item -Path $simpleUpdateCheckerPy -Destination $updateCheckerPy
        Write-Host "update_checker.py copied successfully." -ForegroundColor Green
    } else {
        Write-Host "WARNING: simplified update checker not found. Updates may not work." -ForegroundColor Yellow
    }
} else {
    Write-Host "update_checker.py exists. Backing up and replacing with simplified version..."
    Copy-Item -Path $updateCheckerPy -Destination "$updateCheckerPy.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    if (Test-Path $simpleUpdateCheckerPy) {
        Copy-Item -Path $simpleUpdateCheckerPy -Destination $updateCheckerPy
        Write-Host "update_checker.py replaced with simplified version." -ForegroundColor Green
    } else {
        Write-Host "WARNING: simplified update checker not found. Keeping existing version." -ForegroundColor Yellow
    }
}

# Fix virtual environment
Write-Step "Checking Python environment..."

# Try to find virtual environment
$venvFound = $false
$venvPath = $null

foreach ($dir in @("venv", ".venv", "multi-max")) {
    $testPath = Join-Path -Path $repoDir -ChildPath $dir
    $activateScript = Join-Path -Path $testPath -ChildPath "Scripts\activate.ps1"
    
    if (Test-Path $activateScript) {
        $venvFound = $true
        $venvPath = $testPath
        break
    }
}

if ($venvFound) {
    Write-Host "Found virtual environment at: $venvPath" -ForegroundColor Green
    
    # Test activation
    Write-Host "Testing virtual environment activation..."
    try {
        & $activateScript
        Write-Host "Virtual environment activation successful." -ForegroundColor Green
        
        # Check for missing packages
        Write-Host "Checking for required packages..."
        $packages = @("numpy", "opencv-python", "python-dotenv", "pygame", "psutil")
        $missingPackages = @()
        
        foreach ($package in $packages) {
            $packageName = $package.Split('-')[0]  # Handle packages like opencv-python
            try {
                $null = python -c "import $packageName"
            } catch {
                $missingPackages += $package
            }
        }
        
        if ($missingPackages.Count -gt 0) {
            Write-Host "Missing packages detected. Installing..." -ForegroundColor Yellow
            foreach ($package in $missingPackages) {
                Write-Host "Installing $package..."
                python -m pip install $package
            }
            Write-Host "Packages installed successfully." -ForegroundColor Green
        } else {
            Write-Host "All required packages are installed." -ForegroundColor Green
        }
        
        # Deactivate
        deactivate
    } catch {
        Write-Host "Error activating virtual environment: $_" -ForegroundColor Red
        Write-Host "Creating a new virtual environment..." -ForegroundColor Yellow
        
        # Rename broken venv directory
        Rename-Item -Path $venvPath -NewName "$($venvPath)_broken"
        
        # Create a new one
        $venvDir = Join-Path -Path $repoDir -ChildPath "venv"
        python -m venv $venvDir
        
        # Activate and install packages
        & (Join-Path -Path $venvDir -ChildPath "Scripts\activate.ps1")
        python -m pip install --upgrade pip
        python -m pip install numpy opencv-python python-dotenv pygame psutil
        deactivate
        
        Write-Host "New virtual environment created and packages installed." -ForegroundColor Green
    }
} else {
    Write-Host "No virtual environment found. Creating one..." -ForegroundColor Yellow
    
    # Create a new virtual environment
    $venvDir = Join-Path -Path $repoDir -ChildPath "venv"
    python -m venv $venvDir
    
    # Activate and install packages
    & (Join-Path -Path $venvDir -ChildPath "Scripts\activate.ps1")
    python -m pip install --upgrade pip
    python -m pip install numpy opencv-python python-dotenv pygame psutil
    deactivate
    
    Write-Host "New virtual environment created and packages installed." -ForegroundColor Green
}

# Test launching the application
Write-Step "Testing application launch capability..."

$simpleRunBat = Join-Path -Path $scriptDir -ChildPath "Simple-Run.bat"
if (-not (Test-Path $simpleRunBat)) {
    Write-Host "Simple-Run.bat not found. Checking if it needs to be created..." -ForegroundColor Yellow
    
    # TODO: If needed, create the batch file from scratch
    Write-Host "Please run the full installer to recreate Simple-Run.bat" -ForegroundColor Yellow
} else {
    Write-Host "Simple-Run.bat exists. Launch this file to start the application." -ForegroundColor Green
}

# Finish
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  Installation Fix Completed!" -ForegroundColor Cyan
Write-Host "=================================================`n" -ForegroundColor Cyan
Write-Host "The Multi-Max installation has been repaired." -ForegroundColor White
Write-Host "To run the application, use 'Simple-Run.bat' in the windows directory." -ForegroundColor White
Write-Host "`nIf you still encounter issues, please:" -ForegroundColor White
Write-Host "1. Check the logs in the 'logs' directory" -ForegroundColor White
Write-Host "2. Run the full installer again" -ForegroundColor White
Write-Host "3. Report the issue with logs attached" -ForegroundColor White
Write-Host "`n=================================================`n" -ForegroundColor Cyan

# Ask if user wants to run the application
$runNow = Read-Host "Would you like to try running Multi-Max now? (y/n)"
if ($runNow -eq "y") {
    Write-Host "Launching Multi-Max..."
    if (Test-Path $simpleRunBat) {
        & $simpleRunBat
    } else {
        Write-Host "ERROR: Simple-Run.bat not found. Cannot launch application." -ForegroundColor Red
    }
} 