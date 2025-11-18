"""
Production JWT Authentication with OAuth2
HIPAA-compliant user authentication and authorization
"""

from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import os

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# Models
class Token(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    email: Optional[str] = None
    roles: List[str] = []
    scopes: List[str] = []


class User(BaseModel):
    """User model"""
    user_id: str
    email: str
    full_name: str
    roles: List[str]
    is_active: bool = True
    is_verified: bool = True


class UserInDB(User):
    """User in database with hashed password"""
    hashed_password: str


# Mock user database (replace with actual database)
fake_users_db = {
    "doctor@hospital.com": UserInDB(
        user_id="USR001",
        email="doctor@hospital.com",
        full_name="Dr. John Smith",
        roles=["physician", "neurologist"],
        hashed_password=pwd_context.hash("SecurePassword123!"),
        is_active=True,
        is_verified=True
    ),
    "nurse@hospital.com": UserInDB(
        user_id="USR002",
        email="nurse@hospital.com",
        full_name="Jane Doe",
        roles=["nurse"],
        hashed_password=pwd_context.hash("NursePass456!"),
        is_active=True,
        is_verified=True
    )
}


# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


# User utilities
def get_user(email: str) -> Optional[UserInDB]:
    """Get user from database"""
    return fake_users_db.get(email)


def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with email and password"""
    user = get_user(email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# Token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        token_data = TokenData(
            user_id=user_id,
            email=payload.get("email"),
            roles=payload.get("roles", []),
            scopes=payload.get("scopes", [])
        )
        return token_data

    except JWTError:
        raise credentials_exception


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user"""
    token_data = verify_token(token)

    # Get user from database
    user = get_user(token_data.email) if token_data.email else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return User(**user.dict())


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Role-based access control
class RoleChecker:
    """Check user roles for authorization"""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        for role in user.roles:
            if role in self.allowed_roles:
                return user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )


# Permission-based access control
def check_patient_access(user: User, patient_id: str) -> bool:
    """Check if user has access to patient data"""
    # Implement patient access control logic
    # For now, all physicians have access
    if "physician" in user.roles or "nurse" in user.roles:
        return True
    return False


def require_patient_access(patient_id: str):
    """Dependency to require patient access"""
    async def _check_access(user: User = Depends(get_current_active_user)):
        if not check_patient_access(user, patient_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No access to patient {patient_id}"
            )
        return user
    return _check_access
