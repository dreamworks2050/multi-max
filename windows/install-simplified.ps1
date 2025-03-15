# Multi-Max Simplified Installation Script for Windows
# This script provides a more reliable installation process that addresses common path and environment issues

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Banner
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  Multi-Max Simplified Windows Installation" -ForegroundColor Cyan
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
    Write-Host "You can install Git from https://git-scm.com/download/win"
} else {
    Write-Host "Git is installed." -ForegroundColor Green
}

# Create key directories
Write-Step "Setting up directories..."

# Create logs directory
$logsDir = Join-Path -Path $repoDir -ChildPath "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -Path $logsDir -ItemType Directory | Out-Null
    Write-Host "Created logs directory." -ForegroundColor Green
} else {
    Write-Host "Logs directory already exists." -ForegroundColor Green
}

# Create VERSION file
$versionFile = Join-Path -Path $repoDir -ChildPath "VERSION"
if (-not (Test-Path $versionFile)) {
    "1.0.0" | Out-File -FilePath $versionFile -Encoding utf8
    Write-Host "Created VERSION file." -ForegroundColor Green
} else {
    Write-Host "VERSION file already exists." -ForegroundColor Green
}

# Setup Python environment
Write-Step "Setting up Python environment..."

# Create virtual environment
$venvDir = Join-Path -Path $repoDir -ChildPath "venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvDir
    if (-not $?) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
$activateScript = Join-Path -Path $venvDir -ChildPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..."
    & $activateScript
    if (-not $?) {
        Write-Host "ERROR: Failed to activate virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ERROR: Activation script not found at: $activateScript" -ForegroundColor Red
    exit 1
}

# Install required packages
Write-Step "Installing required packages..."
python -m pip install --upgrade pip
python -m pip install numpy opencv-python python-dotenv pygame psutil yt-dlp

# Ensure main.py is in the correct location
Write-Step "Setting up application files..."

# Copy windows/main.py to parent directory if needed
$windowsMainPy = Join-Path -Path $scriptDir -ChildPath "main.py"
$mainPy = Join-Path -Path $repoDir -ChildPath "main.py"

if (-not (Test-Path $mainPy) -and (Test-Path $windowsMainPy)) {
    Write-Host "Copying Windows-specific main.py to parent directory..."
    Copy-Item -Path $windowsMainPy -Destination $mainPy
    Write-Host "main.py copied successfully." -ForegroundColor Green
} elseif (Test-Path $mainPy) {
    # Check if it's the Windows version
    $content = Get-Content -Path $mainPy -Raw
    if (-not ($content -match "__windows_specific_version__")) {
        Write-Host "Backing up original main.py..."
        Copy-Item -Path $mainPy -Destination "$mainPy.backup"
        
        if (Test-Path $windowsMainPy) {
            Write-Host "Installing Windows-specific main.py..."
            Copy-Item -Path $windowsMainPy -Destination $mainPy
        } else {
            Write-Host "WARNING: Windows-specific main.py not found in windows directory." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Windows-specific main.py is already installed." -ForegroundColor Green
    }
} else {
    Write-Host "ERROR: main.py not found in either location." -ForegroundColor Red
    exit 1
}

# Copy update checker to parent directory for easier imports
$updateCheckerPy = Join-Path -Path $scriptDir -ChildPath "simple_update_checker.py"
$destUpdateCheckerPy = Join-Path -Path $repoDir -ChildPath "update_checker.py"

if (Test-Path $updateCheckerPy) {
    Write-Host "Installing simplified update checker..."
    Copy-Item -Path $updateCheckerPy -Destination $destUpdateCheckerPy
    Write-Host "Update checker installed successfully." -ForegroundColor Green
}

# Create desktop shortcut
Write-Step "Creating desktop shortcut..."

$simpleRunBat = Join-Path -Path $scriptDir -ChildPath "Simple-Run.bat"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path -Path $desktopPath -ChildPath "Multi-Max.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $simpleRunBat
$Shortcut.WorkingDirectory = $scriptDir
$Shortcut.Description = "Launch Multi-Max application"
$Shortcut.Save()

Write-Host "Desktop shortcut created: $shortcutPath" -ForegroundColor Green

# Deactivate virtual environment
deactivate

# Finish
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  Multi-Max Installation Complete!" -ForegroundColor Cyan
Write-Host "=================================================`n" -ForegroundColor Cyan
Write-Host "To run Multi-Max, you can either:" -ForegroundColor White
Write-Host "1. Double-click the desktop shortcut" -ForegroundColor White
Write-Host "2. Run 'Simple-Run.bat' from the windows directory" -ForegroundColor White
Write-Host "3. Run 'Simple-Install.bat' to repair any issues" -ForegroundColor White
Write-Host "`nIf you encounter any problems, check the logs directory for error details." -ForegroundColor White
Write-Host "`n=================================================`n" -ForegroundColor Cyan

# Ask if user wants to run now
$runNow = Read-Host "Would you like to run Multi-Max now? (y/n)"
if ($runNow -eq "y") {
    Write-Host "Starting Multi-Max..."
    & $simpleRunBat
} 