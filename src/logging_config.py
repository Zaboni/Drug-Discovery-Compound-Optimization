#!/usr/bin/env python3
"""
Logging Configuration for Drug Discovery Compound Optimization

This module provides centralized logging configuration for the entire project.
"""

import logging
import logging.config
from pathlib import Path
import yaml
from datetime import datetime


def setup_logging(config_path: str = None, log_level: str = "INFO", 
                 log_dir: str = "logs", console_output: bool = True):
    """
    Set up logging configuration for the project.
    
    Args:
        config_path: Path to logging configuration file
        log_level: Default logging level
        log_dir: Directory for log files
        console_output: Whether to output logs to console
    """
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"drug_discovery_{timestamp}.log"
    
    # Default logging configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(log_filename),
                'mode': 'w',
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(log_dir / 'errors.log'),
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'data_processing': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'] if console_output else ['file'],
                'propagate': False
            },
            'models': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'] if console_output else ['file'],
                'propagate': False
            },
            'training': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'] if console_output else ['file'],
                'propagate': False
            },
            'api': {
                'level': 'INFO',
                'handlers': ['console', 'file'] if console_output else ['file'],
                'propagate': False
            },
            'utils': {
                'level': 'INFO',
                'handlers': ['console', 'file'] if console_output else ['file'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'file', 'error_file'] if console_output else ['file', 'error_file']
        }
    }
    
    # Load custom configuration if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    # Assume it's a Python logging config
                    import json
                    config = json.load(f)
            
            # Update file paths in the loaded config
            for handler in config.get('handlers', {}).values():
                if 'filename' in handler:
                    # Make paths relative to log directory
                    filename = Path(handler['filename'])
                    if not filename.is_absolute():
                        handler['filename'] = str(log_dir / filename)
                        
        except Exception as e:
            print(f"Warning: Could not load logging config from {config_path}: {e}")
            print("Using default logging configuration")
            config = default_config
    else:
        config = default_config
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured successfully")
    logger.info(f"Log files will be saved to: {log_dir}")
    logger.info(f"Main log file: {log_filename}")
    
    return str(log_filename)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed successfully in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_dataframe_info(df, name: str = "DataFrame", logger: logging.Logger = None):
    """
    Log information about a pandas DataFrame.
    
    Args:
        df: pandas DataFrame
        name: Name to use in log messages
        logger: Logger instance (if None, uses root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {list(df.columns)}")
    logger.info(f"{name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Log missing data
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if len(missing_cols) > 0:
        logger.warning(f"{name} missing data: {dict(missing_cols)}")
    else:
        logger.info(f"{name} has no missing data")


def log_model_metrics(metrics: dict, model_name: str = "Model", logger: logging.Logger = None):
    """
    Log model performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model
        logger: Logger instance (if None, uses root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"{model_name} Performance Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


def create_logging_config_file(output_path: str = "config/logging.yaml"):
    """
    Create a sample logging configuration file.
    
    Args:
        output_path: Path where to save the configuration file
    """
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'errors.log'
            }
        },
        'loggers': {
            'data_processing': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'models': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'training': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'api': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file', 'error_file']
        }
    }
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Logging configuration saved to {output_path}")


def main():
    """Example usage of logging configuration."""
    # Setup logging
    log_file = setup_logging(log_level="DEBUG")
    
    # Get logger
    logger = get_logger(__name__)
    
    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test function decorator
    @log_function_call
    def example_function(x, y):
        """Example function to test logging decorator."""
        return x + y
    
    result = example_function(1, 2)
    logger.info(f"Function result: {result}")
    
    # Test DataFrame logging
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, None, 6]})
        log_dataframe_info(df, "Example DataFrame", logger)
    except ImportError:
        logger.warning("Pandas not available for DataFrame logging test")
    
    # Test metrics logging
    metrics = {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88}
    log_model_metrics(metrics, "Example Model", logger)
    
    print(f"Logging test completed. Check log file: {log_file}")


if __name__ == "__main__":
    main()