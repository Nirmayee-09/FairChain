from fastapi import FastAPI
from routes.disruptions import router as disruptions_router
from routes.fairness import router as fairness_router

app = FastAPI(title="FairChain API")

app.include_router(disruptions_router)
app.include_router(fairness_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to FairChain API"}
