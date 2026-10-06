from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Docker!", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}
