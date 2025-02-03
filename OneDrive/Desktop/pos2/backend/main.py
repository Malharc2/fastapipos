from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import auth_router

app = FastAPI(title="Employee Management System")

# List of allowed origins
origins = [
    "http://localhost:3000",      # Local React development
    "http://192.168.1.8:3000",   # React app on your network
    # Add other origins as needed
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allows specified origins
    allow_credentials=True,       # Allows cookies and credentials
    allow_methods=["*"],         # Allows all HTTP methods
    allow_headers=["*"],         # Allows all HTTP headers
)

# Include authentication router
app.include_router(auth_router, tags=["Authentication"], prefix="/auth")