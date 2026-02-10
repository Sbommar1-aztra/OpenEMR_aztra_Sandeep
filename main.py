"""
FastAPI Application for OpenEMR API Integration

This application provides a FastAPI interface to interact with OpenEMR's
REST API and FHIR API endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
import httpx
import os
from datetime import datetime, timedelta
import secrets

# Security scheme
security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="OpenEMR API Interface",
    description="""
    FastAPI interface for interacting with OpenEMR Electronic Health Records system.
    
    ## Features
    - **OAuth 2.0 Authentication** - Secure token-based authentication
    - **FHIR R4 API** - Full FHIR Release 4 support
    - **Standard OpenEMR API** - Native OpenEMR REST endpoints
    - **Interactive Swagger UI** - Test endpoints directly in your browser
    
    ## Quick Start
    
    1. **Configure Environment Variables:**
       ```bash
       OPENEMR_BASE_URL=https://your-openemr-server
       OPENEMR_CLIENT_ID=your_client_id
       OPENEMR_CLIENT_SECRET=your_client_secret
       ```
    
    2. **Register OAuth Client:**
       Use the `/oauth/register` endpoint to register your application
    
    3. **Get Access Token:**
       Use `/oauth/token` endpoint with authorization code
    
    4. **Access Protected Endpoints:**
       Include `Authorization: Bearer <token>` header in requests
    
    ## API Documentation
    
    - **Swagger UI:** Available at `/docs` (interactive testing)
    - **ReDoc:** Available at `/redoc` (alternative documentation)
    - **OpenAPI JSON:** Available at `/openapi.json`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    tags_metadata=[
        {
            "name": "Authentication",
            "description": "OAuth 2.0 authentication endpoints for token management and client registration.",
        },
        {
            "name": "FHIR",
            "description": "FHIR R4 API endpoints for healthcare data interoperability.",
        },
        {
            "name": "Standard API",
            "description": "Standard OpenEMR REST API endpoints for native operations.",
        },
    ],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENEMR_BASE_URL = os.getenv("OPENEMR_BASE_URL", "https://localhost:9300")
OPENEMR_API_BASE = f"{OPENEMR_BASE_URL}/apis/default"
OPENEMR_OAUTH_BASE = f"{OPENEMR_BASE_URL}/oauth2/default"
CLIENT_ID = os.getenv("OPENEMR_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("OPENEMR_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("OPENEMR_REDIRECT_URI", "http://localhost:8000/oauth/callback")

# In-memory token storage (use database in production)
token_storage: Dict[str, Dict[str, Any]] = {}

# OAuth2 scheme (using HTTPBearer for Swagger UI compatibility)
# Note: oauth2_scheme is defined but not currently used - kept for potential future use


# Pydantic Models
class TokenRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "grant_type": "authorization_code",
                "code": "abc123xyz",
                "redirect_uri": "http://localhost:8000/oauth/callback",
                "code_verifier": "optional_pkce_verifier"
            }
        }
    )
    grant_type: str = Field(..., description="Grant type: 'authorization_code' or 'refresh_token'", examples=["authorization_code", "refresh_token"])
    code: Optional[str] = Field(None, description="Authorization code (required for authorization_code grant)")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI (must match registered URI)")
    refresh_token: Optional[str] = Field(None, description="Refresh token (required for refresh_token grant)")
    code_verifier: Optional[str] = Field(None, description="PKCE code verifier (for public clients)")


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "def456uvw",
                "scope": "openid api:fhir patient/Patient.rs",
                "id_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    )
    access_token: str = Field(..., description="OAuth 2.0 access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: Optional[int] = Field(None, description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token for obtaining new access tokens")
    scope: Optional[str] = Field(None, description="Granted scopes")
    id_token: Optional[str] = Field(None, description="OpenID Connect ID token")


class ClientRegistration(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "client_name": "My Healthcare App",
                "redirect_uris": ["http://localhost:8000/oauth/callback"],
                "scope": "openid api:fhir patient/Patient.rs user/Patient.rs",
                "token_endpoint_auth_method": "client_secret_basic"
            }
        }
    )
    client_name: str = Field(..., description="Name of your application", examples=["My Healthcare App"])
    redirect_uris: List[str] = Field(..., description="List of allowed redirect URIs", examples=[["http://localhost:8000/oauth/callback"]])
    scope: Optional[str] = Field(None, description="Space-separated list of requested scopes", examples=["openid api:fhir patient/Patient.rs"])
    token_endpoint_auth_method: Optional[str] = Field(default="client_secret_basic", description="Client authentication method", examples=["client_secret_basic", "client_secret_post", "none"])


class PatientSearch(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "John Doe",
                "birthdate": "1990-01-01",
                "identifier": "12345",
                "_count": 10,
                "_sort": "name"
            }
        }
    )
    name: Optional[str] = Field(None, description="Patient name to search", examples=["John Doe"])
    birthdate: Optional[str] = Field(None, description="Patient birthdate (YYYY-MM-DD)", examples=["1990-01-01"])
    identifier: Optional[str] = Field(None, description="Patient identifier", examples=["12345"])
    id: Optional[str] = Field(None, alias="_id", description="Patient resource UUID")
    count: Optional[int] = Field(default=10, alias="_count", description="Number of results to return", examples=[10, 20, 50])
    sort: Optional[str] = Field(None, alias="_sort", description="Sort criteria (comma-separated, use - for descending)", examples=["name", "-birthdate"])


class PatientCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fname": "John",
                "lname": "Doe",
                "dob": "1990-01-01",
                "sex": "Male",
                "street": "123 Main St",
                "city": "Springfield",
                "state": "IL",
                "postal_code": "62701",
                "phone_cell": "555-1234",
                "email": "john.doe@example.com"
            }
        }
    )
    fname: str = Field(..., description="First name", examples=["John"])
    lname: str = Field(..., description="Last name", examples=["Doe"])
    dob: str = Field(..., description="Date of birth (YYYY-MM-DD)", examples=["1990-01-01"])
    sex: Optional[str] = Field(default="Unknown", description="Sex", examples=["Male", "Female", "Other", "Unknown"])
    street: Optional[str] = Field(None, description="Street address", examples=["123 Main St"])
    city: Optional[str] = Field(None, description="City", examples=["Springfield"])
    state: Optional[str] = Field(None, description="State/Province", examples=["IL"])
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code", examples=["62701"])
    phone_cell: Optional[str] = Field(None, description="Cell phone number", examples=["555-1234"])
    email: Optional[str] = Field(None, description="Email address", examples=["john.doe@example.com"])


class ObservationSearch(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient": "Patient/123",
                "category": "vital-signs",
                "code": "85354-9",
                "_count": 10
            }
        }
    )
    patient: Optional[str] = Field(None, description="Patient reference (e.g., Patient/123)", examples=["Patient/123"])
    category: Optional[str] = Field(None, description="Observation category", examples=["vital-signs", "laboratory"])
    code: Optional[str] = Field(None, description="LOINC code for the observation", examples=["85354-9"])
    count: Optional[int] = Field(default=10, alias="_count", description="Number of results to return")
    sort: Optional[str] = Field(None, alias="_sort", description="Sort criteria")


class EncounterSearch(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient": "Patient/123",
                "status": "finished",
                "date": "2024-01-01",
                "_count": 10
            }
        }
    )
    patient: Optional[str] = Field(None, description="Patient reference", examples=["Patient/123"])
    status: Optional[str] = Field(None, description="Encounter status", examples=["planned", "in-progress", "finished"])
    date: Optional[str] = Field(None, description="Encounter date (FHIR date format)", examples=["2024-01-01"])
    count: Optional[int] = Field(default=10, alias="_count", description="Number of results to return")
    sort: Optional[str] = Field(None, alias="_sort", description="Sort criteria")


# Helper Functions
async def get_access_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract access token from Authorization header"""
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None


async def make_openemr_request(
    method: str,
    endpoint: str,
    token: Optional[str] = None,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    headers: Optional[Dict] = None
) -> Dict[str, Any]:
    """Make HTTP request to OpenEMR API"""
    url = f"{OPENEMR_API_BASE}{endpoint}"
    
    request_headers = headers or {}
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    request_headers.setdefault("Accept", "application/fhir+json")
    request_headers.setdefault("Content-Type", "application/json")
    
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                json=json_data
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else {"error": str(e)}
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_detail
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")


# Authentication Endpoints
@app.get(
    "/",
    tags=["Authentication"],
    summary="API Information",
    description="Root endpoint with API information and links to documentation.",
    responses={
        200: {
            "description": "API information",
            "content": {
                "application/json": {
                    "example": {
                        "name": "OpenEMR API Interface",
                        "version": "1.0.0",
                        "openemr_url": "https://localhost:9300",
                        "docs": "/docs",
                        "redoc": "/redoc"
                    }
                }
            }
        }
    }
)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenEMR API Interface",
        "version": "1.0.0",
        "openemr_url": OPENEMR_BASE_URL,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }


@app.get(
    "/oauth/authorize",
    tags=["Authentication"],
    summary="OAuth 2.0 Authorization",
    description="""
    Initiates the OAuth 2.0 authorization code flow.
    
    Redirects users to OpenEMR's authorization endpoint to begin authentication.
    After authorization, users will be redirected back to your `redirect_uri` with
    an authorization code that can be exchanged for an access token.
    
    **Query Parameters:**
    - `response_type`: Must be "code" (default)
    - `client_id`: Your OAuth client ID
    - `redirect_uri`: Your registered redirect URI
    - `scope`: Space-separated list of requested scopes
    - `state`: Optional state parameter for CSRF protection
    """,
    responses={
        302: {"description": "Redirects to OpenEMR authorization page"},
        400: {"description": "Invalid request (missing client_id)"}
    }
)
async def oauth_authorize(
    response_type: str = Query("code", description="Response type (must be 'code')"),
    client_id: Optional[str] = Query(None, description="OAuth client ID"),
    redirect_uri: Optional[str] = Query(None, description="Redirect URI"),
    scope: Optional[str] = Query(None, description="Requested scopes (space-separated)"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection")
):
    """
    OAuth 2.0 Authorization Endpoint
    
    Redirects to OpenEMR's authorization endpoint.
    """
    client = client_id or CLIENT_ID
    redirect = redirect_uri or REDIRECT_URI
    
    if not client:
        raise HTTPException(status_code=400, detail="client_id is required")
    
    if not scope:
        scope = "openid api:fhir patient/Patient.rs user/Patient.rs"
    
    # Generate state if not provided
    if not state:
        state = secrets.token_urlsafe(32)
        token_storage[state] = {"type": "oauth_state"}
    
    auth_url = (
        f"{OPENEMR_OAUTH_BASE}/authorize"
        f"?response_type={response_type}"
        f"&client_id={client}"
        f"&redirect_uri={redirect}"
        f"&scope={scope}"
        f"&state={state}"
    )
    
    return RedirectResponse(url=auth_url)


@app.get(
    "/oauth/callback",
    tags=["Authentication"],
    summary="OAuth 2.0 Callback",
    description="""
    OAuth 2.0 callback endpoint that receives the authorization code
    and automatically exchanges it for an access token.
    
    This endpoint is called by OpenEMR after user authorization.
    """,
    responses={
        200: {
            "description": "Token response",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "def456uvw"
                    }
                }
            }
        },
        400: {"description": "Invalid authorization code"},
        401: {"description": "Authentication failed"}
    }
)
async def oauth_callback(
    code: str = Query(..., description="Authorization code from OpenEMR"),
    state: Optional[str] = Query(None, description="State parameter (if provided)")
):
    """
    OAuth 2.0 Callback Endpoint
    
    Receives authorization code and exchanges it for access token.
    """
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OPENEMR_OAUTH_BASE}/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            token_response = response.json()
            
            # Store token (in production, use secure storage)
            if "access_token" in token_response:
                token_storage[token_response["access_token"]] = {
                    "token_data": token_response,
                    "expires_at": datetime.now() + timedelta(seconds=token_response.get("expires_in", 3600))
                }
            
            return token_response
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else {"error": str(e)}
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_detail
            )


@app.post(
    "/oauth/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    tags=["Authentication"],
    summary="Get OAuth 2.0 Access Token",
    description="""
    Exchange authorization code for access token or refresh an existing access token.
    
    **Grant Types:**
    - `authorization_code`: Exchange authorization code for access token
    - `refresh_token`: Refresh an expired access token
    
    **Response:**
    Returns an access token that can be used to authenticate API requests.
    Include the token in the `Authorization` header as `Bearer <token>`.
    """,
    responses={
        200: {
            "description": "Token response",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "def456uvw",
                        "scope": "openid api:fhir patient/Patient.rs"
                    }
                }
            }
        },
        400: {"description": "Invalid request (missing required fields)"},
        401: {"description": "Invalid authorization code or credentials"}
    }
)
async def get_token(token_request: TokenRequest):
    """
    OAuth 2.0 Token Endpoint
    
    Exchange authorization code for access token or refresh access token.
    """
    data = {
        "grant_type": token_request.grant_type,
    }
    
    if token_request.grant_type == "authorization_code":
        if not token_request.code:
            raise HTTPException(status_code=400, detail="code is required for authorization_code grant")
        data["code"] = token_request.code
        data["redirect_uri"] = token_request.redirect_uri or REDIRECT_URI
        if token_request.code_verifier:
            data["code_verifier"] = token_request.code_verifier
    
    elif token_request.grant_type == "refresh_token":
        if not token_request.refresh_token:
            raise HTTPException(status_code=400, detail="refresh_token is required for refresh_token grant")
        data["refresh_token"] = token_request.refresh_token
    
    if CLIENT_ID:
        data["client_id"] = CLIENT_ID
    if CLIENT_SECRET:
        data["client_secret"] = CLIENT_SECRET
    
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OPENEMR_OAUTH_BASE}/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else {"error": str(e)}
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_detail
            )


@app.post(
    "/oauth/register",
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register OAuth 2.0 Client",
    description="""
    Register a new OAuth 2.0 client application with OpenEMR.
    
    This endpoint creates a new OAuth client that can be used to authenticate
    and access OpenEMR APIs. The response includes `client_id` and `client_secret`
    which should be stored securely.
    
    **Important:** The `client_secret` is only shown once. Save it securely!
    """,
    responses={
        201: {
            "description": "Client registered successfully",
            "content": {
                "application/json": {
                    "example": {
                        "client_id": "abc123xyz",
                        "client_secret": "secret456",
                        "client_id_issued_at": 1234567890,
                        "client_secret_expires_at": 0,
                        "redirect_uris": ["http://localhost:8000/oauth/callback"],
                        "grant_types": ["authorization_code", "refresh_token"],
                        "response_types": ["code"],
                        "client_name": "My Healthcare App"
                    }
                }
            }
        },
        400: {"description": "Invalid client metadata"},
        422: {"description": "Validation error"}
    }
)
async def register_client(registration: ClientRegistration):
    """
    Register OAuth 2.0 Client
    
    Register a new OAuth client application with OpenEMR.
    """
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OPENEMR_OAUTH_BASE}/registration",
                json=registration.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else {"error": str(e)}
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_detail
            )


# FHIR API Endpoints
@app.get(
    "/fhir/metadata",
    tags=["FHIR"],
    summary="Get FHIR Capability Statement",
    description="""
    Returns the FHIR metadata/capability statement.
    
    **No Authentication Required**
    
    This endpoint returns information about the FHIR server's capabilities,
    including supported resources, operations, and search parameters.
    """,
    responses={
        200: {
            "description": "FHIR CapabilityStatement resource",
            "content": {
                "application/fhir+json": {
                    "example": {
                        "resourceType": "CapabilityStatement",
                        "status": "active",
                        "kind": "instance"
                    }
                }
            }
        }
    }
)
async def get_capability_statement():
    """
    Get FHIR Capability Statement
    
    Returns the FHIR metadata/capability statement (no authentication required).
    """
    return await make_openemr_request("GET", "/fhir/metadata")


@app.get(
    "/fhir/Patient",
    tags=["FHIR"],
    summary="Search Patients (FHIR)",
    description="""
    Search for patient resources using FHIR R4 search parameters.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Search Parameters:**
    - `name`: Patient name
    - `birthdate`: Date of birth
    - `identifier`: Patient identifier
    - `_id`: Resource UUID
    - `_count`: Number of results (default: 10)
    - `_sort`: Sort criteria
    """,
    responses={
        200: {
            "description": "Bundle of Patient resources",
            "content": {
                "application/fhir+json": {
                    "example": {
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "total": 1,
                        "entry": [{
                            "resource": {
                                "resourceType": "Patient",
                                "id": "123",
                                "name": [{"family": "Doe", "given": ["John"]}]
                            }
                        }]
                    }
                }
            }
        },
        401: {"description": "Unauthorized - Missing or invalid token"}
    }
)
async def search_patients(
    search: PatientSearch = Depends(),
    authorization: Optional[str] = Header(None, description="Bearer token for authentication")
):
    """
    Search for Patients (FHIR)
    
    Search for patient resources using FHIR search parameters.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in search.model_dump(by_alias=True).items() if v is not None}
    return await make_openemr_request("GET", "/fhir/Patient", token=token, params=params)


@app.get(
    "/fhir/Patient/{patient_id}",
    tags=["FHIR"],
    summary="Get Patient by ID (FHIR)",
    description="""
    Retrieve a specific patient resource by ID.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Path Parameters:**
    - `patient_id`: Patient resource ID (UUID)
    """,
    responses={
        200: {
            "description": "Patient resource",
            "content": {
                "application/fhir+json": {
                    "example": {
                        "resourceType": "Patient",
                        "id": "123",
                        "name": [{"family": "Doe", "given": ["John"]}],
                        "birthDate": "1990-01-01",
                        "gender": "male"
                    }
                }
            }
        },
        401: {"description": "Unauthorized - Missing or invalid token"},
        404: {"description": "Patient not found"}
    }
)
async def get_patient(
    patient_id: str,
    authorization: Optional[str] = Header(None, description="Bearer token for authentication")
):
    """
    Get Patient by ID (FHIR)
    
    Retrieve a specific patient resource by ID.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("GET", f"/fhir/Patient/{patient_id}", token=token)


@app.post(
    "/fhir/Patient",
    status_code=status.HTTP_201_CREATED,
    tags=["FHIR"],
    summary="Create Patient (FHIR)",
    description="""
    Create a new patient resource using FHIR R4 format.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Request Body:** FHIR Patient resource (JSON)
    
    **Example Patient Resource:**
    ```json
    {
      "resourceType": "Patient",
      "name": [{
        "family": "Doe",
        "given": ["John"]
      }],
      "birthDate": "1990-01-01",
      "gender": "male",
      "identifier": [{
        "system": "http://hospital.example.org",
        "value": "123456"
      }]
    }
    ```
    """,
    responses={
        201: {
            "description": "Patient created successfully",
            "content": {
                "application/fhir+json": {
                    "example": {
                        "resourceType": "Patient",
                        "id": "123",
                        "name": [{"family": "Doe", "given": ["John"]}],
                        "birthDate": "1990-01-01",
                        "gender": "male"
                    }
                }
            }
        },
        401: {"description": "Unauthorized - Missing or invalid token"},
        400: {"description": "Bad request - Invalid patient data"}
    }
)
async def create_patient(
    patient: Dict[str, Any] = Body(
        ...,
        examples=[{
            "resourceType": "Patient",
            "name": [{"family": "Doe", "given": ["John"]}],
            "birthDate": "1990-01-01",
            "gender": "male",
            "identifier": [{"system": "http://hospital.example.org", "value": "123456"}]
        }]
    ),
    authorization: Optional[str] = Header(None, description="Bearer token for authentication")
):
    """
    Create Patient (FHIR)
    
    Create a new patient resource.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("POST", "/fhir/Patient", token=token, json_data=patient)


@app.get("/fhir/Observation")
async def search_observations(
    search: ObservationSearch = Depends(),
    authorization: Optional[str] = Header(None)
):
    """
    Search Observations (FHIR)
    
    Search for observation resources (vital signs, lab results, etc.).
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in search.model_dump(by_alias=True).items() if v is not None}
    return await make_openemr_request("GET", "/fhir/Observation", token=token, params=params)


@app.get("/fhir/Encounter")
async def search_encounters(
    search: EncounterSearch = Depends(),
    authorization: Optional[str] = Header(None)
):
    """
    Search Encounters (FHIR)
    
    Search for encounter/visit resources.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in search.model_dump(by_alias=True).items() if v is not None}
    return await make_openemr_request("GET", "/fhir/Encounter", token=token, params=params)


@app.get("/fhir/MedicationRequest")
async def search_medication_requests(
    patient: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    _count: Optional[int] = Query(10),
    authorization: Optional[str] = Header(None)
):
    """
    Search Medication Requests (FHIR)
    
    Search for medication prescription/order resources.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/fhir/MedicationRequest", token=token, params=params)


@app.get("/fhir/Condition")
async def search_conditions(
    patient: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    _count: Optional[int] = Query(10),
    authorization: Optional[str] = Header(None)
):
    """
    Search Conditions (FHIR)
    
    Search for condition/problem resources.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/fhir/Condition", token=token, params=params)


@app.get("/fhir/Procedure")
async def search_procedures(
    patient: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    _count: Optional[int] = Query(10),
    authorization: Optional[str] = Header(None)
):
    """
    Search Procedures (FHIR)
    
    Search for procedure resources.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/fhir/Procedure", token=token, params=params)


@app.get("/fhir/Appointment")
async def search_appointments(
    patient: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    _count: Optional[int] = Query(10),
    authorization: Optional[str] = Header(None)
):
    """
    Search Appointments (FHIR)
    
    Search for appointment resources.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/fhir/Appointment", token=token, params=params)


@app.get("/fhir/DocumentReference/$docref")
async def generate_document(
    patient: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None)
):
    """
    Generate Clinical Document (FHIR)
    
    Generates a Clinical Summary of Care Document (CCD) for a patient.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/fhir/DocumentReference/$docref", token=token, params=params)


# Standard OpenEMR API Endpoints
@app.get("/api/patient")
async def list_patients(
    name: Optional[str] = Query(None),
    dob: Optional[str] = Query(None),
    pid: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None)
):
    """
    List Patients (Standard API)
    
    Get a list of patients using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/api/patient", token=token, params=params)


@app.post(
    "/api/patient",
    status_code=status.HTTP_201_CREATED,
    tags=["Standard API"],
    summary="Create Patient (Standard API)",
    description="""
    Create a new patient using the Standard OpenEMR API format.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Required Fields:**
    - `fname`: First name
    - `lname`: Last name
    - `dob`: Date of birth (YYYY-MM-DD)
    
    **Optional Fields:**
    - `sex`: Male, Female, Other, or Unknown
    - `street`, `city`, `state`, `postal_code`: Address information
    - `phone_cell`: Phone number
    - `email`: Email address
    """,
    responses={
        201: {
            "description": "Patient created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "validationErrors": {},
                        "internalErrors": [],
                        "data": {
                            "pid": "123",
                            "fname": "John",
                            "lname": "Doe",
                            "dob": "1990-01-01",
                            "sex": "Male"
                        }
                    }
                }
            }
        },
        401: {"description": "Unauthorized - Missing or invalid token"},
        422: {"description": "Validation error - Missing required fields"}
    }
)
async def create_patient_standard(
    patient: PatientCreate,
    authorization: Optional[str] = Header(None, description="Bearer token for authentication")
):
    """
    Create Patient (Standard API)
    
    Create a new patient using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("POST", "/api/patient", token=token, json_data=patient.model_dump())


@app.get("/api/patient/{pid}")
async def get_patient_standard(
    pid: str,
    authorization: Optional[str] = Header(None)
):
    """
    Get Patient by ID (Standard API)
    
    Get patient details by ID using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("GET", f"/api/patient/{pid}", token=token)


@app.get("/api/patient/{pid}/encounter")
async def get_patient_encounters(
    pid: str,
    authorization: Optional[str] = Header(None)
):
    """
    Get Patient Encounters (Standard API)
    
    Get all encounters for a patient using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("GET", f"/api/patient/{pid}/encounter", token=token)


@app.get("/api/encounter")
async def list_encounters(
    pid: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None)
):
    """
    List Encounters (Standard API)
    
    Get a list of encounters using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/api/encounter", token=token, params=params)


@app.get("/api/appointment")
async def list_appointments(
    pid: Optional[str] = Query(None),
    pc_eid: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None)
):
    """
    List Appointments (Standard API)
    
    Get a list of appointments using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in locals().items() if v is not None and k != "authorization" and k != "token"}
    return await make_openemr_request("GET", "/api/appointment", token=token, params=params)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
