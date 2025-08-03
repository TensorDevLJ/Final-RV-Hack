"""
Authentication module for bearer token validation
"""
import os
from fastapi import HTTPException, status

def verify_bearer_token(token: str) -> bool:
    """
    Verify bearer token against environment variable
    """
    expected_token = os.getenv("BEARER_TOKEN")
    
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bearer token not configured"
        )
    
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True