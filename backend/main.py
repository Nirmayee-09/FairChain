from fastapi import FastAPI

app = FastAPI(title="FairChain API")

@app.get("/")
def read_root():
    return {"message": "Welcome to FairChain API"}