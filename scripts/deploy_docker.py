#!/usr/bin/env python3
"""
Docker Deployment Script for Drug Discovery Compound Optimization API

This script provides comprehensive Docker deployment capabilities including
building, running, monitoring, and managing the containerized application.
"""

import argparse
import subprocess
import sys
import os
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent


def run_command(command: List[str], capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with proper error handling."""
    try:
        print(f"🔄 Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=check,
            cwd=project_root
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"❌ Command not found. Make sure Docker is installed and in your PATH.")
        raise


def check_docker():
    """Check if Docker is available and running."""
    try:
        result = run_command(["docker", "--version"], capture_output=True)
        print(f"✅ Found Docker: {result.stdout.strip()}")
        
        result = run_command(["docker", "info"], capture_output=True)
        print("✅ Docker daemon is running")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available or not running")
        print("Please install Docker and ensure the Docker daemon is running.")
        return False


def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        result = run_command(["docker", "compose", "version"], capture_output=True)
        print(f"✅ Found Docker Compose: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try old docker-compose command
            result = run_command(["docker-compose", "--version"], capture_output=True)
            print(f"✅ Found Docker Compose: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker Compose is not available")
            return False


def build_image(tag: str = "drug-discovery-api:latest", target: str = "production", no_cache: bool = False):
    """Build Docker image."""
    print(f"🔨 Building Docker image: {tag}")
    
    command = ["docker", "build"]
    
    if no_cache:
        command.append("--no-cache")
    
    command.extend([
        "--target", target,
        "--tag", tag,
        "."
    ])
    
    run_command(command)
    print(f"✅ Image {tag} built successfully")


def run_container(
    tag: str = "drug-discovery-api:latest",
    name: str = "drug-discovery-api",
    port: int = 8000,
    env_vars: Dict[str, str] = None,
    volumes: Dict[str, str] = None,
    detach: bool = True,
    remove: bool = True
):
    """Run Docker container."""
    print(f"🚀 Running container: {name}")
    
    command = ["docker", "run"]
    
    if detach:
        command.append("-d")
    
    if remove:
        command.append("--rm")
    
    command.extend(["--name", name])
    command.extend(["-p", f"{port}:8000"])
    
    # Add environment variables
    if env_vars:
        for key, value in env_vars.items():
            command.extend(["-e", f"{key}={value}"])
    
    # Add volume mounts
    if volumes:
        for host_path, container_path in volumes.items():
            command.extend(["-v", f"{host_path}:{container_path}"])
    
    command.append(tag)
    
    result = run_command(command, capture_output=True)
    if detach:
        container_id = result.stdout.strip()
        print(f"✅ Container started with ID: {container_id}")
        print(f"🌐 API available at: http://localhost:{port}")
        print(f"📚 Documentation at: http://localhost:{port}/docs")
        return container_id
    else:
        return None


def stop_container(name: str):
    """Stop Docker container."""
    print(f"🛑 Stopping container: {name}")
    try:
        run_command(["docker", "stop", name], capture_output=True)
        print(f"✅ Container {name} stopped")
    except subprocess.CalledProcessError:
        print(f"⚠️ Container {name} was not running or doesn't exist")


def remove_container(name: str):
    """Remove Docker container."""
    print(f"🗑️ Removing container: {name}")
    try:
        run_command(["docker", "rm", name], capture_output=True)
        print(f"✅ Container {name} removed")
    except subprocess.CalledProcessError:
        print(f"⚠️ Container {name} doesn't exist")


def list_containers():
    """List running containers."""
    print("📋 Listing containers:")
    run_command(["docker", "ps"])


def show_logs(name: str, follow: bool = False, tail: int = 100):
    """Show container logs."""
    print(f"📝 Showing logs for container: {name}")
    
    command = ["docker", "logs"]
    
    if follow:
        command.append("-f")
    
    command.extend(["--tail", str(tail)])
    command.append(name)
    
    try:
        run_command(command)
    except KeyboardInterrupt:
        print("\n👋 Log streaming stopped")


def docker_compose_up(
    compose_file: str = "docker-compose.yml",
    env_file: Optional[str] = None,
    profiles: List[str] = None,
    detach: bool = True,
    build: bool = False
):
    """Run Docker Compose."""
    print(f"🚀 Starting services with Docker Compose")
    
    command = ["docker", "compose", "-f", compose_file]
    
    if env_file:
        command.extend(["--env-file", env_file])
    
    if profiles:
        for profile in profiles:
            command.extend(["--profile", profile])
    
    command.append("up")
    
    if detach:
        command.append("-d")
    
    if build:
        command.append("--build")
    
    try:
        run_command(command)
        if detach:
            print("✅ Services started successfully")
            print("🌐 API available at: http://localhost:8000")
            print("📚 Documentation at: http://localhost:8000/docs")
    except KeyboardInterrupt:
        print("\n👋 Docker Compose stopped")


def docker_compose_down(compose_file: str = "docker-compose.yml", volumes: bool = False):
    """Stop Docker Compose services."""
    print("🛑 Stopping Docker Compose services")
    
    command = ["docker", "compose", "-f", compose_file, "down"]
    
    if volumes:
        command.append("-v")
    
    run_command(command)
    print("✅ Services stopped")


def docker_compose_logs(
    compose_file: str = "docker-compose.yml",
    service: Optional[str] = None,
    follow: bool = False
):
    """Show Docker Compose logs."""
    command = ["docker", "compose", "-f", compose_file, "logs"]
    
    if follow:
        command.append("-f")
    
    if service:
        command.append(service)
    
    try:
        run_command(command)
    except KeyboardInterrupt:
        print("\n👋 Log streaming stopped")


def health_check(url: str = "http://localhost:8000/health", timeout: int = 30):
    """Check API health."""
    print(f"🔍 Checking API health at {url}")
    
    try:
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ API is healthy")
                    print(f"   Status: {data.get('status', 'unknown')}")
                    print(f"   Version: {data.get('version', 'unknown')}")
                    print(f"   Uptime: {data.get('uptime', 'unknown')} seconds")
                    return True
                else:
                    print(f"⚠️ API returned status {response.status_code}")
            except requests.exceptions.RequestException:
                print("⏳ Waiting for API to start...")
                time.sleep(2)
        
        print(f"❌ API health check failed after {timeout} seconds")
        return False
        
    except ImportError:
        print("⚠️ requests library not available, cannot perform health check")
        print("Install with: pip install requests")
        return None


def cleanup_images(force: bool = False):
    """Clean up Docker images."""
    print("🧹 Cleaning up Docker images")
    
    if force:
        # Remove all unused images
        run_command(["docker", "image", "prune", "-a", "-f"])
    else:
        # Remove dangling images only
        run_command(["docker", "image", "prune", "-f"])
    
    print("✅ Cleanup completed")


def show_system_info():
    """Show Docker system information."""
    print("🔧 Docker system information:")
    run_command(["docker", "system", "df"])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy Drug Discovery API with Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build and run with default settings
  python scripts/deploy_docker.py build-and-run

  # Build development image
  python scripts/deploy_docker.py build --target development

  # Run with custom port and environment
  python scripts/deploy_docker.py run --port 8080 --env ENVIRONMENT=production

  # Start with Docker Compose
  python scripts/deploy_docker.py compose up --build

  # Start with development profile
  python scripts/deploy_docker.py compose up --profiles dev-tools

  # Show logs
  python scripts/deploy_docker.py logs --follow

  # Health check
  python scripts/deploy_docker.py health

  # Stop everything
  python scripts/deploy_docker.py stop
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build Docker image')
    build_parser.add_argument('--tag', default='drug-discovery-api:latest', help='Image tag')
    build_parser.add_argument('--target', default='production', choices=['production', 'development'], help='Build target')
    build_parser.add_argument('--no-cache', action='store_true', help='Build without cache')

    # Run command  
    run_parser = subparsers.add_parser('run', help='Run Docker container')
    run_parser.add_argument('--tag', default='drug-discovery-api:latest', help='Image tag')
    run_parser.add_argument('--name', default='drug-discovery-api', help='Container name')
    run_parser.add_argument('--port', type=int, default=8000, help='Host port')
    run_parser.add_argument('--env', action='append', default=[], help='Environment variables (KEY=VALUE)')
    run_parser.add_argument('--volume', action='append', default=[], help='Volume mounts (HOST:CONTAINER)')
    run_parser.add_argument('--no-detach', action='store_true', help='Run in foreground')
    run_parser.add_argument('--no-remove', action='store_true', help='Don\'t remove container on exit')

    # Build and run command
    build_run_parser = subparsers.add_parser('build-and-run', help='Build and run in one step')
    build_run_parser.add_argument('--tag', default='drug-discovery-api:latest', help='Image tag')
    build_run_parser.add_argument('--target', default='production', choices=['production', 'development'], help='Build target')
    build_run_parser.add_argument('--port', type=int, default=8000, help='Host port')
    build_run_parser.add_argument('--no-cache', action='store_true', help='Build without cache')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop containers')
    stop_parser.add_argument('name', nargs='?', default='drug-discovery-api', help='Container name')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove containers')
    remove_parser.add_argument('name', nargs='?', default='drug-discovery-api', help='Container name')

    # List command
    subparsers.add_parser('list', help='List running containers')

    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show container logs')
    logs_parser.add_argument('name', nargs='?', default='drug-discovery-api', help='Container name')
    logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow log output')
    logs_parser.add_argument('--tail', type=int, default=100, help='Number of lines to show')

    # Docker Compose commands
    compose_parser = subparsers.add_parser('compose', help='Docker Compose commands')
    compose_subparsers = compose_parser.add_subparsers(dest='compose_command', help='Compose commands')

    # Compose up
    compose_up_parser = compose_subparsers.add_parser('up', help='Start services')
    compose_up_parser.add_argument('--file', '-f', default='docker-compose.yml', help='Compose file')
    compose_up_parser.add_argument('--env-file', help='Environment file')
    compose_up_parser.add_argument('--profiles', nargs='+', help='Compose profiles to enable')
    compose_up_parser.add_argument('--no-detach', action='store_true', help='Run in foreground')
    compose_up_parser.add_argument('--build', action='store_true', help='Build before starting')

    # Compose down
    compose_down_parser = compose_subparsers.add_parser('down', help='Stop services')
    compose_down_parser.add_argument('--file', '-f', default='docker-compose.yml', help='Compose file')
    compose_down_parser.add_argument('--volumes', '-v', action='store_true', help='Remove volumes')

    # Compose logs
    compose_logs_parser = compose_subparsers.add_parser('logs', help='Show service logs')
    compose_logs_parser.add_argument('--file', '-f', default='docker-compose.yml', help='Compose file')
    compose_logs_parser.add_argument('--service', help='Specific service name')
    compose_logs_parser.add_argument('--follow', action='store_true', help='Follow log output')

    # Health check command
    health_parser = subparsers.add_parser('health', help='Check API health')
    health_parser.add_argument('--url', default='http://localhost:8000/health', help='Health check URL')
    health_parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up Docker resources')
    cleanup_parser.add_argument('--force', action='store_true', help='Remove all unused images')

    # System info command
    subparsers.add_parser('info', help='Show Docker system information')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Check Docker availability
    if not check_docker():
        return 1

    if not args.command:
        print("❌ No command specified. Use --help for usage information.")
        return 1

    try:
        # Handle commands
        if args.command == 'build':
            build_image(args.tag, args.target, args.no_cache)

        elif args.command == 'run':
            # Parse environment variables
            env_vars = {}
            for env in args.env:
                if '=' in env:
                    key, value = env.split('=', 1)
                    env_vars[key] = value

            # Parse volume mounts
            volumes = {}
            for volume in args.volume:
                if ':' in volume:
                    host_path, container_path = volume.split(':', 1)
                    volumes[host_path] = container_path

            container_id = run_container(
                args.tag,
                args.name,
                args.port,
                env_vars,
                volumes,
                not args.no_detach,
                not args.no_remove
            )

            if container_id and not args.no_detach:
                # Wait a moment then check health
                time.sleep(5)
                health_check(f"http://localhost:{args.port}/health")

        elif args.command == 'build-and-run':
            build_image(args.tag, args.target, args.no_cache)
            container_id = run_container(args.tag, port=args.port)
            if container_id:
                time.sleep(5)
                health_check(f"http://localhost:{args.port}/health")

        elif args.command == 'stop':
            stop_container(args.name)

        elif args.command == 'remove':
            stop_container(args.name)
            remove_container(args.name)

        elif args.command == 'list':
            list_containers()

        elif args.command == 'logs':
            show_logs(args.name, args.follow, args.tail)

        elif args.command == 'compose':
            if not check_docker_compose():
                return 1

            if args.compose_command == 'up':
                docker_compose_up(
                    args.file,
                    args.env_file,
                    args.profiles,
                    not args.no_detach,
                    args.build
                )
                if not args.no_detach:
                    time.sleep(5)
                    health_check()

            elif args.compose_command == 'down':
                docker_compose_down(args.file, args.volumes)

            elif args.compose_command == 'logs':
                docker_compose_logs(args.file, args.service, args.follow)

            else:
                print("❌ Unknown compose command")
                return 1

        elif args.command == 'health':
            if not health_check(args.url, args.timeout):
                return 1

        elif args.command == 'cleanup':
            cleanup_images(args.force)

        elif args.command == 'info':
            show_system_info()

        else:
            print(f"❌ Unknown command: {args.command}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)