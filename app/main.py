from fastapi import FastAPI
from .db_models.db import Base, engine, get_db
from .views.auth import router as main_router
from .views.rag.api import router as rag_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)
Base.metadata.create_all(bind=engine)

app.include_router(main_router)
app.include_router(rag_router)