"""
Pydantic models to validate request and response
"""
from pydantic import BaseModel

class Request(BaseModel):
    """
    Pydantic model to validate the request
    """
    user_id: str
    message: str
    clear: bool

class Response(BaseModel):
    """
    Pydantic model to validate the response
    """
    status: int
    response: str
