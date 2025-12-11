import sys
import os

# Add parent directory to path so we can import dq_service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dq_service import app

# Vercel expects the app to be available as a variable
__all__ = ["app"]

