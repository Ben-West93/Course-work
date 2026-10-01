# FastAPI Basics

FastAPI is a modern Python web framework for building APIs. It is built on top of Starlette for the web layer and Pydantic for data handling, and it uses standard Python type hints to describe what each endpoint expects and returns. Because the framework reads those type hints, a lot of boilerplate disappears: you declare a parameter as an int and FastAPI converts and validates it for you.

Pydantic models are the heart of request validation. When you define a class that inherits from BaseModel and use it as a parameter in a POST endpoint, FastAPI parses the JSON body, checks every field against its declared type, and returns a 422 Unprocessable Entity response with a detailed error message if anything is missing or malformed. This means invalid data never reaches your business logic.

FastAPI generates interactive documentation automatically. Visiting /docs opens Swagger UI, where you can try every endpoint from the browser, and /redoc offers an alternative read-only view. The documentation always stays in sync with the code because it is produced from the same type hints and models that drive validation.

Async support lets a single FastAPI server handle many requests concurrently. When an endpoint is declared with async def and awaits a slow operation such as a database query or an outbound HTTP call, the event loop is free to process other requests instead of blocking. Run the app with an ASGI server such as uvicorn: uvicorn main:app --reload starts a development server that restarts when files change.

Dependency injection is another core feature. A function passed to Depends() runs before the endpoint and its return value is injected as a parameter. Common uses include opening a database session, reading the current user from an authentication token, or enforcing pagination limits. Dependencies can depend on other dependencies, which keeps endpoint code short and testable.

Status codes and errors are handled with HTTPException. Raising HTTPException(status_code=404, detail="Item not found") stops the request and returns a JSON error body. Successful creation endpoints typically set status_code=201 in the route decorator so clients know a new resource was created.
