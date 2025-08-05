#!/usr/bin/env python3
"""
Local API Server Runner for Drug Discovery Compound Optimization

This script provides a convenient way to run the API server locally
with various configuration options.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import yaml
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.api_frontend import app, api_instance, FASTAPI_AVAILABLE
    from src.logging_config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Drug Discovery API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/run_api.py

  # Run in development mode with auto-reload
  python scripts/run_api.py --dev

  # Run on specific host and port
  python scripts/run_api.py --host 0.0.0.0 --port 8080

  # Run with specific configuration file
  python scripts/run_api.py --config config/production.yaml

  # Run with debug logging
  python scripts/run_api.py --log-level DEBUG

  # Run with workers (production)
  python scripts/run_api.py --workers 4
        """
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with auto-reload"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: config/config.yaml)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (if not specified, logs to console)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, only for production)"
    )

    parser.add_argument(
        "--reload-dirs",
        nargs="+",
        default=["src", "config"],
        help="Directories to watch for changes in dev mode"
    )

    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable access log"
    )

    parser.add_argument(
        "--no-access-log",
        action="store_true",
        help="Disable access log"
    )

    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        help="SSL key file for HTTPS"
    )

    parser.add_argument(
        "--ssl-certfile",
        type=str,
        help="SSL certificate file for HTTPS"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if API can be imported without starting server"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )

    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load configuration from file."""
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Configuration file {config_path} not found, using defaults")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def check_dependencies():
    """Check if required dependencies are available."""
    dependencies = {
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "pydantic": "Data validation",
        "numpy": "Numerical computing",
        "pandas": "Data manipulation"
    }
    
    missing = []
    for dep, description in dependencies.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{dep} ({description})")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_api_health():
    """Check if API can be imported and basic functionality works."""
    try:
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI not available")
            return False
        
        if app is None:
            print("❌ API app could not be created")
            return False
        
        # Try to get app info
        print("✅ API app created successfully")
        print(f"✅ App title: {app.title}")
        print(f"✅ App version: {app.version}")
        
        # Check routes
        route_count = len(app.routes)
        print(f"✅ {route_count} routes registered")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking API health: {e}")
        return False


def show_version():
    """Show version information."""
    print("Drug Discovery Compound Optimization API")
    print("=" * 45)
    print(f"API Version: 1.0.0")
    print(f"Python Version: {sys.version}")
    print(f"Project Path: {project_root}")
    
    try:
        import uvicorn
        print(f"Uvicorn Version: {uvicorn.__version__}")
    except:
        print("Uvicorn: Not available")
    
    try:
        import fastapi
        print(f"FastAPI Version: {fastapi.__version__}")
    except:
        print("FastAPI: Not available")
    
    try:
        import numpy
        print(f"NumPy Version: {numpy.__version__}")
    except:
        print("NumPy: Not available")
    
    try:
        import pandas
        print(f"Pandas Version: {pandas.__version__}")
    except:
        print("Pandas: Not available")


def run_server(args, config):
    """Run the API server with specified configuration."""
    # Setup logging
    log_file_path = None
    if args.log_file:
        log_file_path = setup_logging(
            log_level=args.log_level,
            log_dir=Path(args.log_file).parent,
            console_output=True
        )
    else:
        setup_logging(log_level=args.log_level, console_output=True)
    
    # Determine server configuration
    api_config = config.get('api', {})
    
    host = args.host or api_config.get('host', '127.0.0.1')
    port = args.port or api_config.get('port', 8000)
    
    # Enable reload in development mode
    reload = args.dev or api_config.get('reload', False)
    
    # Configure access log
    access_log = True
    if args.no_access_log:
        access_log = False
    elif args.access_log:
        access_log = True
    
    # Prepare uvicorn configuration
    uvicorn_config = {
        "app": "src.api_frontend:app",
        "host": host,
        "port": port,
        "log_level": args.log_level.lower(),
        "access_log": access_log,
    }
    
    # Development mode settings
    if args.dev or reload:
        uvicorn_config.update({
            "reload": True,
            "reload_dirs": [str(project_root / dir_name) for dir_name in args.reload_dirs],
            "workers": 1  # Force single worker in dev mode
        })
        print(f"🔄 Running in DEVELOPMENT mode with auto-reload")
        print(f"📁 Watching directories: {args.reload_dirs}")
    else:
        # Production mode settings
        uvicorn_config["workers"] = args.workers
        if args.workers > 1:
            print(f"🚀 Running in PRODUCTION mode with {args.workers} workers")
        else:
            print(f"🚀 Running in PRODUCTION mode")
    
    # SSL configuration
    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_config.update({
            "ssl_keyfile": args.ssl_keyfile,
            "ssl_certfile": args.ssl_certfile
        })
        protocol = "https"
    else:
        protocol = "http"
    
    # Print startup information
    print(f"\n🧬 Drug Discovery Compound Optimization API")
    print(f"📍 Server URL: {protocol}://{host}:{port}")
    print(f"📚 API Documentation: {protocol}://{host}:{port}/docs")
    print(f"🔍 Alternative docs: {protocol}://{host}:{port}/redoc")
    print(f"💚 Health check: {protocol}://{host}:{port}/health")
    print(f"📊 Metrics: {protocol}://{host}:{port}/metrics")
    
    if log_file_path:
        print(f"📝 Logs: {log_file_path}")
    
    print(f"\n🎯 Starting server...")
    print("Press CTRL+C to quit")
    
    try:
        # Import uvicorn here to avoid import errors
        import uvicorn
        # Start server
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return 1
    
    return 0


def main():
    """Main entry point."""
    args = parse_args()
    
    # Show version and exit
    if args.version:
        show_version()
        return 0
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check API health
    if args.check:
        if check_api_health():
            print("✅ API check passed")
            return 0
        else:
            print("❌ API check failed")
            return 1
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if we can start the API
    if not check_api_health():
        print("❌ Cannot start server - API health check failed")
        return 1
    
    # Run server
    return run_server(args, config)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)