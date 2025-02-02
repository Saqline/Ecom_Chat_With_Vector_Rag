
from datetime import timedelta
import datetime
import random
import string
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import sessionmaker, Session, relationship
from app.db_models.db import get_db
from app.db_models.models import User


SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3000
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)



def create_access_token(data: dict, expires_delta: timedelta = None, db_session: Session = None):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + expires_delta if expires_delta else datetime.datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    user_id = data["user_id"]
    user = db_session.query(User).filter(User.id == user_id).first()
    user.access_token = encoded_jwt
    db_session.commit()
    return encoded_jwt
    

# Define OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
        if user.access_token != token:
            raise HTTPException(status_code=400, detail="Your token does not match with the one stored in the database.")
        return user     
    except jwt.PyJWTError:
        raise credentials_exception





def generate_six_digit_code():
    chars = string.ascii_uppercase + string.digits 
    random_sys = random.SystemRandom()
    return ''.join(random_sys.choice(chars) for _ in range(6))


def generate_code_and_expiry():
    code = generate_six_digit_code()
    expiry = datetime.datetime.utcnow() + timedelta(minutes=5)
    return code, expiry
