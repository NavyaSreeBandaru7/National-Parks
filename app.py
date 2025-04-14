"""
National Park Sentiment Analysis Deployment System

Robust launch system with dependency management and environment configuration.
"""

import os
import platform
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, Tuple

# Package mapping with exact version specifications
PACKAGE_REQUIREMENTS: Dict[str, str] = {
    'streamlit': '1.29.0',
    'pandas': '2.0.3',
    'numpy': '1.24.3',
    'matplotlib': '3.7.2',
    'seaborn': '0.13.0',
    'plotly': '5.18.0',
    'wordcloud': '1.9.2',
    'requests': '2.31.0',
    'beautifulsoup4': '4.12.2',
    'transformers': '4.35.2',
    'torch': '2.1.0',
    'scikit-learn': '1.3.2',
    'nltk': '3.8.1',
    'tqdm': '4.66.1',
    'lxml': '4.9.3',
    'protobuf': '3.20.3',
    'sentencepiece': '0.1.99',
    'tokenizers': '0.14.1'
}

SYSTEM_CONFIG = {
    "required_ports": [8501],
    "min_python_version": (3, 8)
}

def check_system_compatibility() -> bool:
    """Verify system meets minimum requirements."""
    python_version = sys.version_info[:2]
    if python_version < SYSTEM_CONFIG["min_python_version"]:
        print(f"❌ Python {'.'.join(map(str, SYSTEM_CONFIG['min_python_version']))}+ required.")
        return False
    return True

def verify_dependencies() -> Tuple[bool, list]:
    """Check installed packages against requirements."""
    missing = []
    wrong_version = []

    for pkg, req_version in PACKAGE_REQUIREMENTS.items():
        try:
            imported = __import__(pkg)
            installed_version = getattr(imported, '__version__', None)
            
            if installed_version != req_version:
                wrong_version.append((pkg, installed_version, req_version))
        except ImportError:
            missing.append(pkg)

    return (len(missing) == 0 and len(wrong_version) == 0), missing, wrong_version

def setup_virtual_environment():
    """Create and configure a virtual environment."""
    venv_path = Path(__file__).parent / "sentiment_env"
    
    if platform.system() == "Windows":
        venv_script = str(venv_path / "Scripts" / "activate")
    else:
        venv_script = f"source {venv_path}/bin/activate"

    if not venv_path.exists():
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    
    print("🔧 Installing dependencies...")
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "-r", "requirements.txt"
    ]
    subprocess.run(install_cmd, check=True)
    print(f"✅ Virtual environment ready at: {venv_path}")

def configure_environment():
    """Set up runtime environment and paths."""
    # Create project-specific cache directories
    cache_paths = {
        "TRANSFORMERS_CACHE": "cache/transformers",
        "NLTK_DATA": "cache/nltk",
        "TORCH_HOME": "cache/torch"
    }

    for env_var, path in cache_paths.items():
        full_path = Path(__file__).parent / path
        full_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(full_path)
        print(f"📁 Configured {env_var} => {full_path}")

    # Configure essential environment variables
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def initialize_nltk():
    """Ensure NLTK resources are available."""
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }

    try:
        import nltk
        for resource, path in resources.items():
            try:
                nltk.data.find(path)
                print(f"✅ NLTK {resource} already available")
            except LookupError:
                print(f"📥 Downloading {resource}...")
                nltk.download(resource, quiet=True)
                nltk.data.find(path)  # Verify download
                print(f"✅ {resource} successfully installed")
    except Exception as e:
        print(f"❌ NLTK initialization failed: {str(e)}")
        sys.exit(1)

def launch_dashboard():
    """Start the Streamlit application with monitoring."""
    app_files = [
        "app.py",
        "dashboard.py",
        "national_park_sentiment.py"
    ]

    for app_file in app_files:
        if (Path(__file__).parent / app_file).exists():
            target_app = app_file
            break
    else:
        print("❌ No application file found. Tried:")
        print("\n".join(f"- {f}" for f in app_files))
        sys.exit(1)

    print(f"🚀 Launching {target_app}...")
    
    # Configure Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / target_app),
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.serverAddress", "0.0.0.0",
        "--logger.level", "info"
    ]

    # Start browser after delay
    def browser_launcher():
        time.sleep(5)
        webbrowser.open("http://localhost:8501")
    
    import threading
    threading.Thread(target=browser_launcher, daemon=True).start()

    # Run Streamlit with output capture
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Real-time output monitoring
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            print(line.strip())

        if process.returncode != 0:
            print(f"❌ Application exited with code {process.returncode}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal...")
        process.terminate()
        sys.exit(0)

def main():
    """Main deployment workflow."""
    print("\n" + "=" * 60)
    print("National Park Sentiment Analysis Deployment System")
    print("=" * 60 + "\n")

    if not check_system_compatibility():
        sys.exit(1)

    print("🔍 Checking system environment...")
    deps_ok, missing, wrong_versions = verify_dependencies()
    
    if not deps_ok:
        if missing:
            print("\n❌ Missing dependencies:")
            print("\n".join(f"- {pkg}" for pkg in missing))
        if wrong_versions:
            print("\n⚠️ Version mismatches:")
            for pkg, actual, required in wrong_versions:
                print(f"- {pkg}: Installed {actual or 'unknown'}, Required {required}")
        
        print("\n🔄 Attempting automatic repair...")
        setup_virtual_environment()
    
    configure_environment()
    initialize_nltk()
    
    print("\n" + "=" * 60)
    print("Starting Sentiment Analysis Dashboard")
    print("=" * 60 + "\n")
    
    try:
        launch_dashboard()
    except Exception as e:
        print(f"❌ Critical failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
