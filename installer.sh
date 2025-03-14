#!/bin/bash

# Multi-Max Installer Script
# This script checks for and installs all necessary dependencies to run the Multi-Max application
# Installation order: Homebrew → Python → FFmpeg → Python packages

set -e  # Exit immediately if a command exits with a non-zero status

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Multi-Max Installation Script         ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Installation order:                   ${NC}"
echo -e "${BLUE}  1. Homebrew (macOS only)              ${NC}"
echo -e "${BLUE}  2. Python                             ${NC}"
echo -e "${BLUE}  3. FFmpeg                             ${NC}"
echo -e "${BLUE}  4. Python packages                    ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Detect OS and Architecture
OS="unknown"
ARCH="unknown"

if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    if [[ $(uname -m) == "arm64" ]]; then
        ARCH="arm64"
        echo -e "${GREEN}Detected: macOS on Apple Silicon (M1/M2/M3)${NC}"
    else
        ARCH="x86_64"
        echo -e "${GREEN}Detected: macOS on Intel${NC}"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    ARCH=$(uname -m)
    echo -e "${GREEN}Detected: Linux on $ARCH${NC}"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
    if [[ $(uname -m) == "x86_64" ]]; then
        ARCH="x86_64"
    else
        ARCH="x86"
    fi
    echo -e "${GREEN}Detected: Windows on $ARCH${NC}"
    echo -e "${RED}Note: Windows support is limited - manual installation may be required${NC}"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Check for Homebrew on macOS
install_homebrew() {
    echo -e "${YELLOW}Step 1: Installing Homebrew (required for Python and FFmpeg)...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH based on architecture
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${YELLOW}Adding Homebrew to PATH for Apple Silicon...${NC}"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo -e "${YELLOW}Adding Homebrew to PATH for Intel Mac...${NC}"
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    # Verify Homebrew installation
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew installation failed. Cannot continue.${NC}"
        exit 1
    fi
    
    # Update Homebrew
    echo -e "${YELLOW}Updating Homebrew...${NC}"
    brew update
}

# Install Python using the appropriate method for each OS
install_python() {
    echo -e "${YELLOW}Step 2: Installing Python...${NC}"
    
    if [[ "$OS" == "macos" ]]; then
        brew install python
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip
        else
            echo -e "${RED}Unsupported Linux distribution. Please install Python 3 manually.${NC}"
            exit 1
        fi
    elif [[ "$OS" == "windows" ]]; then
        echo -e "${YELLOW}Please install Python from https://www.python.org/downloads/ manually.${NC}"
        echo -e "${YELLOW}Then run this script again.${NC}"
        exit 1
    fi
    
    # Verify Python installation
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}Python installation failed. Cannot continue.${NC}"
        exit 1
    fi
}

# Install FFmpeg using the appropriate method for each OS
install_ffmpeg() {
    echo -e "${YELLOW}Step 3: Installing FFmpeg...${NC}"
    
    if [[ "$OS" == "macos" ]]; then
        brew install ffmpeg
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        else
            echo -e "${RED}Unsupported Linux distribution. Please install FFmpeg manually.${NC}"
            exit 1
        fi
    elif [[ "$OS" == "windows" ]]; then
        echo -e "${YELLOW}Please install FFmpeg from https://ffmpeg.org/download.html manually.${NC}"
        echo -e "${YELLOW}Then add it to your PATH environment variable.${NC}"
    fi
    
    # Verify FFmpeg installation
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}FFmpeg installation failed. Cannot continue.${NC}"
        exit 1
    fi
}

# Install python dependencies from requirements.txt
install_python_deps() {
    echo -e "${YELLOW}Step 4: Installing Python dependencies...${NC}"
    
    # Create and activate virtual environment
    if [[ ! -d "venv" ]]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        $PYTHON -m venv venv
    fi
    
    # Determine activation script based on OS
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Initialize UV_INSTALLED flag
    UV_INSTALLED=false
    
    # Install UV (faster package manager)
    echo -e "${YELLOW}Installing UV package manager (faster than pip)...${NC}"
    if ! command -v uv &> /dev/null; then
        if [[ "$OS" == "macos" || "$OS" == "linux" ]]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh || {
                echo -e "${YELLOW}UV installation failed, falling back to pip.${NC}"
                UV_INSTALLED=false
            }
            
            # Source updated profile to get uv in path if it was installed
            if [[ -f "$HOME/.bashrc" ]]; then
                source "$HOME/.bashrc"
            elif [[ -f "$HOME/.zshrc" ]]; then
                source "$HOME/.zshrc"
            fi
            
            # Check if UV is now in PATH
            if command -v uv &> /dev/null; then
                echo -e "${GREEN}UV package manager installed successfully.${NC}"
                UV_INSTALLED=true
            else
                echo -e "${YELLOW}UV not found in PATH after installation, falling back to pip.${NC}"
                UV_INSTALLED=false
            fi
        elif [[ "$OS" == "windows" ]]; then
            echo -e "${YELLOW}UV not available for Windows through this installer, using pip instead.${NC}"
            UV_INSTALLED=false
        fi
    else
        echo -e "${GREEN}UV package manager already installed: $(uv --version 2>&1 | head -n 1)${NC}"
        UV_INSTALLED=true
    fi
    
    # Upgrade pip as fallback if UV not available
    if [[ "$UV_INSTALLED" == "false" ]]; then
        echo -e "${YELLOW}Upgrading pip...${NC}"
        $PIP install --upgrade pip
    fi
    
    # Install requirements using UV if available, otherwise use pip
    if [[ -f "requirements.txt" ]]; then
        if [[ "$UV_INSTALLED" == "true" ]]; then
            echo -e "${YELLOW}Installing packages from requirements.txt using UV (faster)...${NC}"
            uv pip install -r requirements.txt || {
                echo -e "${YELLOW}UV installation failed, falling back to pip...${NC}"
                $PIP install -r requirements.txt || {
                    echo -e "${YELLOW}Warning: Some packages failed to install. Continuing anyway...${NC}"
                }
            }
        else
            echo -e "${YELLOW}Installing packages from requirements.txt using pip...${NC}"
            $PIP install -r requirements.txt || {
                echo -e "${YELLOW}Warning: Some packages failed to install. Continuing anyway...${NC}"
            }
        fi
    else
        echo -e "${RED}requirements.txt not found! Cannot install required Python packages.${NC}"
        exit 1
    fi
    
    # Check if yt-dlp command is available outside of Python
    if ! command -v yt-dlp &> /dev/null; then
        echo -e "${YELLOW}Installing yt-dlp command line tool...${NC}"
        if [[ "$OS" == "macos" ]]; then
            brew install yt-dlp
        elif [[ "$OS" == "linux" ]]; then
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y yt-dlp
            elif command -v pip3 &> /dev/null; then
                sudo pip3 install --upgrade yt-dlp
            fi
        elif [[ "$OS" == "windows" ]]; then
            echo -e "${YELLOW}Please install yt-dlp manually: https://github.com/yt-dlp/yt-dlp#installation${NC}"
        fi
    else
        echo -e "${GREEN}yt-dlp command-line tool is already installed: $(yt-dlp --version 2>&1 | head -n 1)${NC}"
    fi
    
    # Install platform-specific dependencies
    if [[ "$OS" == "macos" ]]; then
        install_macos_deps
    fi
}

# Check for and install macOS-specific dependencies
install_macos_deps() {
    echo -e "${YELLOW}Installing macOS-specific dependencies...${NC}"
    
    # Install PyObjC components for hardware acceleration
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${YELLOW}Installing PyObjC for Apple Silicon hardware acceleration...${NC}"
        # Install the main PyObjC packages which include all frameworks
        echo -e "${YELLOW}Installing main PyObjC package (includes all frameworks)...${NC}"
        
        if [[ "$UV_INSTALLED" == "true" ]]; then
            uv pip install pyobjc || echo -e "${YELLOW}Warning: Full PyObjC installation failed, trying core components...${NC}"
            
            # Try installing individual components if main package fails
            echo -e "${YELLOW}Installing PyObjC core components...${NC}"
            uv pip install pyobjc-core || echo -e "${YELLOW}Warning: PyObjC core installation failed.${NC}"
            
            # Try installing Quartz framework which is needed for hardware acceleration
            echo -e "${YELLOW}Installing Quartz framework...${NC}"
            uv pip install pyobjc-framework-Quartz || echo -e "${YELLOW}Warning: PyObjC Quartz framework installation failed.${NC}"
        else
            $PIP install pyobjc || echo -e "${YELLOW}Warning: Full PyObjC installation failed, trying core components...${NC}"
            
            # Try installing individual components if main package fails
            echo -e "${YELLOW}Installing PyObjC core components...${NC}"
            $PIP install pyobjc-core || echo -e "${YELLOW}Warning: PyObjC core installation failed.${NC}"
            
            # Try installing Quartz framework which is needed for hardware acceleration
            echo -e "${YELLOW}Installing Quartz framework...${NC}"
            $PIP install pyobjc-framework-Quartz || echo -e "${YELLOW}Warning: PyObjC Quartz framework installation failed.${NC}"
        fi
        
        # Note: We no longer try to install CoreFoundation separately as it's included in pyobjc-core
        echo -e "${YELLOW}Note: CoreFoundation framework is included in pyobjc-core${NC}"
    elif [[ "$ARCH" == "x86_64" ]]; then
        echo -e "${YELLOW}Installing PyObjC for Intel Mac compatibility...${NC}"
        # Install specific version of PyObjC that works well with Intel Macs
        echo -e "${YELLOW}Installing Intel-compatible PyObjC packages...${NC}"
        
        if [[ "$UV_INSTALLED" == "true" ]]; then
            # Try installing main package first with version constraint
            uv pip install "pyobjc<9.0" || echo -e "${YELLOW}Warning: Full PyObjC installation failed, trying core components...${NC}"
            
            # Try installing individual components with version constraints
            echo -e "${YELLOW}Installing PyObjC core components...${NC}"
            uv pip install "pyobjc-core<9.0" || echo -e "${YELLOW}Warning: PyObjC core installation failed.${NC}"
            
            # Install Quartz framework for Intel Macs
            echo -e "${YELLOW}Installing Quartz framework...${NC}"
            uv pip install "pyobjc-framework-Quartz<9.0" || echo -e "${YELLOW}Warning: PyObjC Quartz framework installation failed.${NC}"
        else
            # Try installing main package first with version constraint
            $PIP install "pyobjc<9.0" || echo -e "${YELLOW}Warning: Full PyObjC installation failed, trying core components...${NC}"
            
            # Try installing individual components with version constraints
            echo -e "${YELLOW}Installing PyObjC core components...${NC}"
            $PIP install "pyobjc-core<9.0" || echo -e "${YELLOW}Warning: PyObjC core installation failed.${NC}"
            
            # Install Quartz framework for Intel Macs
            echo -e "${YELLOW}Installing Quartz framework...${NC}"
            $PIP install "pyobjc-framework-Quartz<9.0" || echo -e "${YELLOW}Warning: PyObjC Quartz framework installation failed.${NC}"
        fi
        
        echo -e "${YELLOW}Note: Hardware acceleration will be limited on Intel Macs${NC}"
    fi
    
    # Install pygame dependencies through Homebrew for both architectures
    echo -e "${YELLOW}Installing PyGame dependencies...${NC}"
    brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf portmidi
}

# Set up .env file from template
setup_env_file() {
    echo -e "${YELLOW}Setting up environment configuration...${NC}"
    
    if [[ -f ".env" ]]; then
        echo -e "${GREEN}.env file already exists. Keeping existing configuration.${NC}"
        echo -e "${YELLOW}To reset to defaults, delete .env and run the installer again.${NC}"
    elif [[ -f ".env.template" ]]; then
        echo -e "${YELLOW}Creating .env file from template...${NC}"
        cp .env.template .env
        echo -e "${GREEN}.env file created successfully.${NC}"
        
        # Customize .env file based on detected system
        if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then
            echo -e "${YELLOW}Configuring .env for Apple Silicon...${NC}"
            # No need to modify as the defaults are already set for Apple Silicon
        elif [[ "$OS" == "macos" && "$ARCH" == "x86_64" ]]; then
            echo -e "${YELLOW}Configuring .env for Intel Mac...${NC}"
            # Disable hardware acceleration for Intel Macs
            sed -i '' 's/FORCE_HARDWARE_ACCELERATION=true/FORCE_HARDWARE_ACCELERATION=false/' .env
            sed -i '' 's/ALLOW_SOFTWARE_FALLBACK=false/ALLOW_SOFTWARE_FALLBACK=true/' .env
        elif [[ "$OS" == "linux" ]]; then
            echo -e "${YELLOW}Configuring .env for Linux...${NC}"
            # Disable hardware acceleration for Linux
            sed -i 's/FORCE_HARDWARE_ACCELERATION=true/FORCE_HARDWARE_ACCELERATION=false/' .env
            sed -i 's/ALLOW_SOFTWARE_FALLBACK=false/ALLOW_SOFTWARE_FALLBACK=true/' .env
        elif [[ "$OS" == "windows" ]]; then
            echo -e "${YELLOW}Configuring .env for Windows...${NC}"
            # Disable hardware acceleration for Windows
            sed -i 's/FORCE_HARDWARE_ACCELERATION=true/FORCE_HARDWARE_ACCELERATION=false/' .env
            sed -i 's/ALLOW_SOFTWARE_FALLBACK=false/ALLOW_SOFTWARE_FALLBACK=true/' .env
        fi
        
        echo -e "${GREEN}.env configured for your system.${NC}"
    else
        echo -e "${RED}.env.template not found! Creating basic .env file...${NC}"
        cat > .env << EOL
# Multi-Max Environment Configuration - Auto-generated
FORCE_HARDWARE_ACCELERATION=$(if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then echo "true"; else echo "false"; fi)
ALLOW_SOFTWARE_FALLBACK=$(if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then echo "false"; else echo "true"; fi)
DEFAULT_VIDEO_URL=https://www.youtube.com/watch?v=dQw4w9WgXcQ
ENABLE_MEMORY_TRACING=false
LOG_LEVEL=INFO
FRAME_BUFFER_SIZE=60
EOL
        echo -e "${GREEN}Basic .env file created.${NC}"
    fi
}

# Create desktop shortcut for easy launching
create_desktop_shortcut() {
    echo -e "${YELLOW}Would you like to create a desktop shortcut for easy launching? (y/n)${NC}"
    read -r create_shortcut
    
    if [[ "$create_shortcut" != "y" && "$create_shortcut" != "Y" ]]; then
        echo -e "${YELLOW}Skipping desktop shortcut creation.${NC}"
        return
    fi
    
    # Get current directory as absolute path
    APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [[ "$OS" == "macos" ]]; then
        echo -e "${YELLOW}Creating macOS application launcher...${NC}"
        
        # Create the AppleScript
        DESKTOP_DIR="$HOME/Desktop"
        APP_NAME="Multi-Max"
        SCRIPT_PATH="$DESKTOP_DIR/$APP_NAME.app/Contents/MacOS"
        
        # Create directory structure
        mkdir -p "$SCRIPT_PATH"
        
        # Create the executable script
        cat > "$SCRIPT_PATH/$APP_NAME" << EOL
#!/bin/bash
cd "$APP_DIR"
source venv/bin/activate
python main.py
EOL
        chmod +x "$SCRIPT_PATH/$APP_NAME"
        
        # Create Info.plist
        mkdir -p "$DESKTOP_DIR/$APP_NAME.app/Contents/Resources"
        cat > "$DESKTOP_DIR/$APP_NAME.app/Contents/Info.plist" << EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.multimax.app</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>Multi-Max</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>
EOL

        echo -e "${GREEN}Created macOS launcher at $DESKTOP_DIR/$APP_NAME.app${NC}"
        echo -e "${YELLOW}Note: You can drag this to your Applications folder if desired${NC}"
        
    elif [[ "$OS" == "linux" ]]; then
        echo -e "${YELLOW}Creating Linux desktop shortcut...${NC}"
        
        DESKTOP_DIR="$HOME/Desktop"
        ICON_PATH="$APP_DIR/icon.png"
        
        # Create a basic icon if none exists
        if [[ ! -f "$ICON_PATH" ]]; then
            echo -e "${YELLOW}No icon found, using system default.${NC}"
            ICON_PATH="applications-multimedia"
        fi
        
        # Create desktop entry
        cat > "$DESKTOP_DIR/Multi-Max.desktop" << EOL
[Desktop Entry]
Version=1.0
Type=Application
Name=Multi-Max
Comment=Multi-Max Recursive Video Grid
Exec=bash -c "cd $APP_DIR && source venv/bin/activate && python main.py"
Icon=$ICON_PATH
Terminal=false
Categories=Video;AudioVideo;Graphics;
EOL
        chmod +x "$DESKTOP_DIR/Multi-Max.desktop"
        
        echo -e "${GREEN}Created Linux desktop shortcut at $DESKTOP_DIR/Multi-Max.desktop${NC}"
        
    elif [[ "$OS" == "windows" ]]; then
        echo -e "${YELLOW}Creating Windows shortcut...${NC}"
        
        DESKTOP_DIR="$HOME/Desktop"
        
        # Create batch file
        cat > "$DESKTOP_DIR/Multi-Max.bat" << EOL
@echo off
cd "$APP_DIR"
call venv\Scripts\activate.bat
python main.py
EOL
        
        echo -e "${GREEN}Created Windows batch file at $DESKTOP_DIR/Multi-Max.bat${NC}"
        
    else
        echo -e "${RED}Unsupported OS for desktop shortcut creation.${NC}"
    fi
}

# Main installation process
main() {
    echo -e "${GREEN}Starting installation process...${NC}"
    
    # Determine Python and pip commands
    if command -v python3 &> /dev/null; then
        PYTHON="python3"
    elif command -v python &> /dev/null; then
        PYTHON="python"
    else
        PYTHON=""
    fi
    
    if command -v pip3 &> /dev/null; then
        PIP="pip3"
    elif command -v pip &> /dev/null; then
        PIP="pip"
    else
        PIP=""
    fi
    
    # STEP 1: Check for and install Homebrew on macOS (REQUIRED FIRST)
    if [[ "$OS" == "macos" ]]; then
        echo -e "${BLUE}STEP 1: Checking for Homebrew (required for subsequent steps)${NC}"
        if ! command -v brew &> /dev/null; then
            install_homebrew
        else
            echo -e "${GREEN}Homebrew is already installed.${NC}"
            # Update Homebrew anyway
            echo -e "${YELLOW}Updating Homebrew...${NC}"
            brew update
        fi
    fi
    
    # STEP 2: Check for and install Python
    echo -e "${BLUE}STEP 2: Checking for Python${NC}"
    if [[ -z "$PYTHON" ]]; then
        install_python
        
        # Re-check for Python
        if command -v python3 &> /dev/null; then
            PYTHON="python3"
            PIP="pip3"
        elif command -v python &> /dev/null; then
            PYTHON="python"
            PIP="pip"
        else
            echo -e "${RED}Failed to install Python. Please install it manually.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Python is already installed: $($PYTHON --version)${NC}"
    fi
    
    # STEP 3: Check for and install FFmpeg
    echo -e "${BLUE}STEP 3: Checking for FFmpeg${NC}"
    if ! command -v ffmpeg &> /dev/null; then
        install_ffmpeg
    else
        echo -e "${GREEN}FFmpeg is already installed: $(ffmpeg -version | head -n 1)${NC}"
    fi
    
    # STEP 4: Install Python dependencies
    echo -e "${BLUE}STEP 4: Installing Python dependencies${NC}"
    install_python_deps
    
    # STEP 5: Set up .env configuration file
    echo -e "${BLUE}STEP 5: Setting up environment configuration${NC}"
    setup_env_file
    
    echo -e "${GREEN}Installation complete!${NC}"
    
    # Special note for Apple Silicon users about hardware acceleration
    if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then
        echo -e "${BLUE}=========================================${NC}"
        echo -e "${BLUE}  Apple Silicon Hardware Acceleration   ${NC}"
        echo -e "${BLUE}=========================================${NC}"
        echo -e "${GREEN}Your Mac has Apple Silicon (M1/M2/M3) which supports hardware acceleration.${NC}"
        echo -e "${GREEN}The PyObjC libraries have been installed to enable this feature.${NC}"
        echo -e "${YELLOW}Note: If you experience any issues with hardware acceleration,${NC}"
        echo -e "${YELLOW}edit your .env file and set:${NC}"
        echo -e "${YELLOW}  FORCE_HARDWARE_ACCELERATION=false${NC}"
        echo -e "${YELLOW}  ALLOW_SOFTWARE_FALLBACK=true${NC}"
    elif [[ "$OS" == "macos" && "$ARCH" == "x86_64" ]]; then
        echo -e "${BLUE}=========================================${NC}"
        echo -e "${BLUE}  Intel Mac Configuration               ${NC}"
        echo -e "${BLUE}=========================================${NC}"
        echo -e "${GREEN}Your Mac has an Intel processor.${NC}"
        echo -e "${GREEN}Hardware acceleration has been configured for compatibility.${NC}"
        echo -e "${YELLOW}Note: Performance may be limited compared to Apple Silicon.${NC}"
        echo -e "${YELLOW}The application is configured to use software rendering by default.${NC}"
    fi
    
    # STEP 6: Create desktop shortcut if user wants it
    echo -e "${BLUE}STEP 6: Desktop shortcut${NC}"
    create_desktop_shortcut
    
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}  To run the application:                ${NC}"
    echo -e "${BLUE}  1. source venv/bin/activate            ${NC}"
    echo -e "${BLUE}  2. python main.py                      ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}  Environment configuration is in .env   ${NC}"
    echo -e "${BLUE}  Edit this file to customize settings   ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    
    # Check if venv is already activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo -e "${YELLOW}Virtual environment is not activated. Activate it with:${NC}"
        echo -e "${YELLOW}  source venv/bin/activate${NC}"
    fi
}

# Run the main function
main 