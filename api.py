from fastapi import FastAPI

root = FastAPI(title="mon api")


@root.get("/")
def saluer(name: int):
    return f"salut {name}"
