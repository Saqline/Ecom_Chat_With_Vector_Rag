import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.db_models.db import get_db
from app.db_models.models import User
from app.utills.auth_utils import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, generate_code_and_expiry, get_current_user, get_password_hash, pwd_context, verify_password
from pydantic import BaseModel, EmailStr, Field
from fastapi import BackgroundTasks

from app.utills.email import send_verification_email

router = APIRouter()

## --Register--
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(...)
class UserResponse(BaseModel):
    id: int
    email: EmailStr
    class Config:
        orm_mode = True
@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, 
                  background_tasks: BackgroundTasks,
                  db: Session = Depends(get_db),
                  ):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already registered"
        )
    hashed_password = get_password_hash(user.password)
    ev_code, ev_code_expire = generate_code_and_expiry()
    new_user = User(
        email=user.email, hashed_password=hashed_password,
        ev_code=ev_code, ev_code_expire=ev_code_expire,role="customer",register_type="email"
        )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    background_tasks.add_task(send_verification_email, new_user.email, ev_code)
    return new_user

## --Login--
class UserLogin(BaseModel):
    email: EmailStr
    password: str
@router.post("/login")
def login_user(user_login: UserLogin, 
               background_tasks: BackgroundTasks ,
               db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_login.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if not verify_password(user_login.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Username and Password not match!")
    
    # if not user.email_verified:
    #     ev_code, ev_code_expire = generate_code_and_expiry()
    #     user.ev_code = ev_code
    #     user.ev_code_expire = ev_code_expire
    #     db.commit()
    #     #background_tasks.add_task(send_verification_email, user.email, user.ev_code)
    #     raise HTTPException(status_code=400, detail="Email not verified. A new verification code has been sent to your email.")
    
    access_token = create_access_token(data={"user_id": str(user.id)}, expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),db_session=db)
    return {"access_token": access_token, "token_type": "Bearer","user_id":user.id}


class EmailSchema(BaseModel):
    gmail: str
@router.post("/send-verification-email")
def verification_email(
    email_data: EmailSchema,  
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email_data.gmail).first()

    if not user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")

    if user.email_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    ev_code, ev_code_expire = generate_code_and_expiry()

    user.ev_code = ev_code
    user.ev_code_expire = ev_code_expire
    db.commit()
    

    background_tasks.add_task(send_verification_email, email_data.gmail, ev_code)

    return {"message": "Verification code sent to email"}

class VerifyEmailRequest(BaseModel):
    email: str
    ev_code: str
@router.post("/verify-email")
def verify_email(data: VerifyEmailRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")

    if user.ev_code != data.ev_code:
        raise HTTPException(status_code=400, detail="Invalid verification code")

    if user.ev_code_expire < datetime.datetime.utcnow():
        raise HTTPException(status_code=400, detail="Verification code has expired")

    user.email_verified = True
    user.ev_code = None  
    user.ev_code_expire = None  
    db.commit()
    access_token = create_access_token(data={"user_id": str(user.id)}, expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),db_session=db)

    return {"message": "Email verified successfully", "access_token": access_token,"user_id":user.id}


class ForgotPasswordRequest(BaseModel):
    email: str
@router.post("/send-forgot-password-otp")
def send_forgot_password_otp(data: ForgotPasswordRequest,background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")
    if user.email_verified == False:
        raise HTTPException(status_code=400, detail="User not found")
    
    fp_code, fp_code_expire = generate_code_and_expiry()

    user.fp_code = fp_code
    user.fp_code_expire = fp_code_expire 
    db.commit()

    background_tasks.add_task(send_verification_email, data.email, fp_code)

    return {"message": "Forgot password code sent to email"}

class ResetPasswordSchema(BaseModel):
    email: str
    fp_code: str
    new_password: str

class OtpVerifySchema(BaseModel):
    email: str
    fp_code: str
   

@router.post("/fp-otp-verify")
def otp_verify(
    reset_data: OtpVerifySchema,  
    background_tasks: BackgroundTasks,  
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == reset_data.email).first()

    if not user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")

    if user.fp_code != reset_data.fp_code:
        raise HTTPException(status_code=400, detail="Invalid forgot password code")

    if user.fp_code_expire < datetime.datetime.utcnow():
        raise HTTPException(status_code=400, detail="Forgot password code has expired")

    return {"message": "go for reset password"}

@router.post("/reset-password")
def reset_password(
    reset_data: ResetPasswordSchema,  
    background_tasks: BackgroundTasks,  
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == reset_data.email).first()

    if not user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")

    if user.fp_code != reset_data.fp_code:
        raise HTTPException(status_code=400, detail="Invalid forgot password code")

    if user.fp_code_expire < datetime.datetime.utcnow():
        raise HTTPException(status_code=400, detail="Forgot password code has expired")

    hashed_password = get_password_hash(reset_data.new_password)
    user.hashed_password = hashed_password
    user.fp_code = None  
    user.fp_code_expire = None  
    db.commit()

    # Add background task to send email
    #background_tasks.add_task(send_reset_email, reset_data.gmail)

    return {"message": "Password reset successfully"}


@router.post("/token")
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Username and Password not match!")
    
    if not user.email_verified:
        ev_code, ev_code_expire = generate_code_and_expiry()
        user.ev_code = ev_code
        user.ev_code_expire = ev_code_expire
        db.commit()

        raise HTTPException(status_code=400, detail="Email not verified. A new verification code has been sent to your email.")
    
    access_token = create_access_token(data={"user_id": str(user.id)}, expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),db_session=db)
    return {"access_token": access_token, "token_type": "Bearer","user_id":user.id}

@router.get("/user/me")
async def read_user_me(current_user: User = Depends(get_current_user)):
    return current_user