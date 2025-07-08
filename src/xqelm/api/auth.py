"""
Authentication Module

JWT-based authentication for the XQELM API.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import secrets

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from loguru import logger

from ..utils.config import get_config


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Configuration
config = get_config()
SECRET_KEY = config.api.secret_key
ALGORITHM = config.api.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = config.api.access_token_expire_minutes


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token"
        )


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Username of authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        payload = verify_token(credentials.credentials)
        username = payload.get("sub")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return username
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


class UserManager:
    """
    Simple user management for demo purposes.
    In production, this would integrate with a proper user database.
    """
    
    def __init__(self):
        """Initialize user manager with demo users."""
        self.users = {
            "demo": {
                "username": "demo",
                "email": "demo@xqelm.com",
                "hashed_password": get_password_hash("demo123"),
                "full_name": "Demo User",
                "is_active": True,
                "roles": ["user"],
                "created_at": datetime.utcnow(),
                "last_login": None
            },
            "admin": {
                "username": "admin",
                "email": "admin@xqelm.com",
                "hashed_password": get_password_hash("admin123"),
                "full_name": "Admin User",
                "is_active": True,
                "roles": ["admin", "user"],
                "created_at": datetime.utcnow(),
                "last_login": None
            },
            "legal_expert": {
                "username": "legal_expert",
                "email": "expert@xqelm.com",
                "hashed_password": get_password_hash("expert123"),
                "full_name": "Legal Expert",
                "is_active": True,
                "roles": ["expert", "user"],
                "created_at": datetime.utcnow(),
                "last_login": None
            }
        }
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User data if authentication successful, None otherwise
        """
        user = self.users.get(username)
        
        if not user:
            return None
        
        if not user["is_active"]:
            return None
        
        if not verify_password(password, user["hashed_password"]):
            return None
        
        # Update last login
        user["last_login"] = datetime.utcnow()
        
        return user
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User data if found, None otherwise
        """
        return self.users.get(username)
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str,
        roles: Optional[List[str]] = None
    ) -> bool:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            full_name: Full name
            roles: User roles
            
        Returns:
            True if user created successfully, False otherwise
        """
        if username in self.users:
            return False
        
        if roles is None:
            roles = ["user"]
        
        self.users[username] = {
            "username": username,
            "email": email,
            "hashed_password": get_password_hash(password),
            "full_name": full_name,
            "is_active": True,
            "roles": roles,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        return True
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """
        Update user information.
        
        Args:
            username: Username
            updates: Dictionary of fields to update
            
        Returns:
            True if user updated successfully, False otherwise
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Handle password update
        if "password" in updates:
            user["hashed_password"] = get_password_hash(updates["password"])
            del updates["password"]
        
        # Update other fields
        for key, value in updates.items():
            if key in user and key not in ["username", "hashed_password", "created_at"]:
                user[key] = value
        
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """
        Deactivate a user.
        
        Args:
            username: Username
            
        Returns:
            True if user deactivated successfully, False otherwise
        """
        if username not in self.users:
            return False
        
        self.users[username]["is_active"] = False
        return True
    
    def has_role(self, username: str, role: str) -> bool:
        """
        Check if user has a specific role.
        
        Args:
            username: Username
            role: Role to check
            
        Returns:
            True if user has the role, False otherwise
        """
        user = self.users.get(username)
        if not user:
            return False
        
        return role in user.get("roles", [])
    
    def get_user_stats(self) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Returns:
            Dictionary with user statistics
        """
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user["is_active"])
        
        role_counts = {}
        for user in self.users.values():
            for role in user.get("roles", []):
                role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "role_distribution": role_counts
        }


# Global user manager instance
user_manager = UserManager()


def require_role(required_role: str):
    """
    Decorator to require a specific role for endpoint access.
    
    Args:
        required_role: Required role
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: str = Depends(get_current_user)) -> str:
        if not user_manager.has_role(current_user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return current_user
    
    return role_checker


def require_any_role(required_roles: List[str]):
    """
    Decorator to require any of the specified roles for endpoint access.
    
    Args:
        required_roles: List of acceptable roles
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: str = Depends(get_current_user)) -> str:
        user_has_role = any(
            user_manager.has_role(current_user, role) 
            for role in required_roles
        )
        
        if not user_has_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(required_roles)}"
            )
        return current_user
    
    return role_checker


class APIKeyManager:
    """
    API key management for service-to-service authentication.
    """
    
    def __init__(self):
        """Initialize API key manager."""
        self.api_keys = {
            "xqelm_demo_key_123": {
                "name": "Demo API Key",
                "created_at": datetime.utcnow(),
                "last_used": None,
                "is_active": True,
                "permissions": ["query", "bail_analysis", "cheque_bounce_analysis"],
                "rate_limit": 1000,  # requests per hour
                "usage_count": 0
            }
        }
    
    def generate_api_key(self, name: str, permissions: List[str], rate_limit: int = 1000) -> str:
        """
        Generate a new API key.
        
        Args:
            name: Name/description for the API key
            permissions: List of permissions
            rate_limit: Rate limit per hour
            
        Returns:
            Generated API key
        """
        # Generate secure random API key
        api_key = f"xqelm_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "is_active": True,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "usage_count": 0
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            API key data if valid, None otherwise
        """
        key_data = self.api_keys.get(api_key)
        
        if not key_data or not key_data["is_active"]:
            return None
        
        # Update usage
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return key_data
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        """
        Check if API key has a specific permission.
        
        Args:
            api_key: API key
            permission: Permission to check
            
        Returns:
            True if API key has permission, False otherwise
        """
        key_data = self.api_keys.get(api_key)
        if not key_data or not key_data["is_active"]:
            return False
        
        return permission in key_data.get("permissions", [])
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        if api_key not in self.api_keys:
            return False
        
        self.api_keys[api_key]["is_active"] = False
        return True


# Global API key manager instance
api_key_manager = APIKeyManager()


async def get_api_key_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Authenticate using API key.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        API key identifier
        
    Raises:
        HTTPException: If API key is invalid
    """
    api_key = credentials.credentials
    
    key_data = api_key_manager.validate_api_key(api_key)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key


def require_api_permission(permission: str):
    """
    Decorator to require a specific API permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def permission_checker(api_key: str = Depends(get_api_key_user)) -> str:
        if not api_key_manager.has_permission(api_key, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key does not have required permission: {permission}"
            )
        return api_key
    
    return permission_checker


def create_session_token(user_data: Dict[str, Any]) -> str:
    """
    Create a session token for web applications.
    
    Args:
        user_data: User data to encode
        
    Returns:
        Session token
    """
    session_data = {
        "sub": user_data["username"],
        "email": user_data["email"],
        "roles": user_data["roles"],
        "session_id": secrets.token_hex(16),
        "created_at": datetime.utcnow().isoformat()
    }
    
    return create_access_token(session_data, timedelta(hours=8))  # 8-hour session


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.
    
    Args:
        api_key: API key to hash
        
    Returns:
        Hashed API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_csrf_token() -> str:
    """
    Generate a CSRF token for web forms.
    
    Returns:
        CSRF token
    """
    return secrets.token_urlsafe(32)


def validate_csrf_token(token: str, expected_token: str) -> bool:
    """
    Validate a CSRF token.
    
    Args:
        token: Token to validate
        expected_token: Expected token value
        
    Returns:
        True if token is valid, False otherwise
    """
    return secrets.compare_digest(token, expected_token)