import os
import sys
from pathlib import Path

def add_project_root_to_path(marker_files=None, max_depth=10):
    """
    Add the project root directory to sys.path so imports work from anywhere.
    
    Args:
        marker_files (list): Files/dirs that indicate project root 
                           (default: ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt'])
        max_depth (int): Maximum number of parent directories to search
        
    Returns:
        str: Path to project root, or None if not found
    """
    if marker_files is None:
        marker_files = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt', '.gitignore']
    
    current_path = Path(__file__).resolve().parent
    
    # Search up the directory tree
    for _ in range(max_depth):
        # Check if any marker files exist in current directory
        for marker in marker_files:
            if (current_path / marker).exists():
                project_root = str(current_path)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                return project_root
        
        # Move up one directory
        parent = current_path.parent
        if parent == current_path:  # Reached filesystem root
            break
        current_path = parent
    
    return None

def get_project_root(marker_files=None, max_depth=10):
    """
    Get the project root directory without modifying sys.path.
    
    Args:
        marker_files (list): Files/dirs that indicate project root
        max_depth (int): Maximum number of parent directories to search
        
    Returns:
        Path: Path object to project root, or None if not found
    """
    if marker_files is None:
        marker_files = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt', '.gitignore']
    
    current_path = Path(__file__).resolve().parent
    
    for _ in range(max_depth):
        for marker in marker_files:
            if (current_path / marker).exists():
                return current_path
        
        parent = current_path.parent
        if parent == current_path:
            break
        current_path = parent
    
    return None

# Automatically add project root when this module is imported
project_root = add_project_root_to_path()