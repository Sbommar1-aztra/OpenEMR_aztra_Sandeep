"""
FastAPI Application for OpenEMR API Integration

This application provides a FastAPI interface to interact with OpenEMR's
REST API and FHIR API endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import os
from datetime import datetime, timedelta
import secrets

app = FastAPI(
    title="OpenEMR API Interface",
    description="""
    FastAPI interface for interacting with OpenEMR Electronic Health Records system.
    
    ## Features
    - **OAuth 2.0 Authentication** - Secure token-based authentication
    - **FHIR R4 API** - Full FHIR Release 4 support
    - **Standard OpenEMR API** - Native OpenEMR REST endpoints
    - **Automatic Documentation** - Interactive API docs (Swagger UI)
    
    ## Setup
    1. Configure your OpenEMR server URL in environment variables
    2. Register your OAuth client in OpenEMR
    3. Use the authentication endpoints to get access tokens
    4. Access protected endpoints with Bearer tokens
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# Pydantic Models
class TokenRequest(BaseModel):
    grant_type: str = Field(..., description="Grant type: 'authorization_code' or 'refresh_token'")
    code: Optional[str] = Field(None, description="Authorization code")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    code_verifier: Optional[str] = Field(None, description="PKCE code verifier")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None


class ClientRegistration(BaseModel):
    client_name: str
    redirect_uris: List[str]
    scope: Optional[str] = None
    token_endpoint_auth_method: Optional[str] = "client_secret_basic"


class PatientSearch(BaseModel):
    name: Optional[str] = None
    birthdate: Optional[str] = None
    identifier: Optional[str] = None
    _id: Optional[str] = None
    _count: Optional[int] = 10
    _sort: Optional[str] = None


class PatientCreate(BaseModel):
    fname: str
    lname: str
    dob: str
    sex: Optional[str] = "Unknown"
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    phone_cell: Optional[str] = None
    email: Optional[str] = None


class ObservationSearch(BaseModel):
    patient: Optional[str] = None
    category: Optional[str] = None
    code: Optional[str] = None
    _count: Optional[int] = 10
    _sort: Optional[str] = None


class EncounterSearch(BaseModel):
    patient: Optional[str] = None
    status: Optional[str] = None
    date: Optional[str] = None
    _count: Optional[int] = 10
    _sort: Optional[str] = None


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
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenEMR API Interface",
        "version": "1.0.0",
        "openemr_url": OPENEMR_BASE_URL,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/oauth/authorize")
async def oauth_authorize(
    response_type: str = Query("code"),
    client_id: Optional[str] = Query(None),
    redirect_uri: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
    state: Optional[str] = Query(None)
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


@app.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(...),
    state: Optional[str] = Query(None)
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


@app.post("/oauth/token", response_model=TokenResponse)
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


@app.post("/oauth/register")
async def register_client(registration: ClientRegistration):
    """
    Register OAuth 2.0 Client
    
    Register a new OAuth client application with OpenEMR.
    """
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OPENEMR_OAUTH_BASE}/registration",
                json=registration.dict(),
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
@app.get("/fhir/metadata")
async def get_capability_statement():
    """
    Get FHIR Capability Statement
    
    Returns the FHIR metadata/capability statement (no authentication required).
    """
    return await make_openemr_request("GET", "/fhir/metadata")


@app.get("/fhir/Patient")
async def search_patients(
    search: PatientSearch = Depends(),
    authorization: Optional[str] = Header(None)
):
    """
    Search for Patients (FHIR)
    
    Search for patient resources using FHIR search parameters.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    params = {k: v for k, v in search.dict().items() if v is not None}
    return await make_openemr_request("GET", "/fhir/Patient", token=token, params=params)


@app.get("/fhir/Patient/{patient_id}")
async def get_patient(
    patient_id: str,
    authorization: Optional[str] = Header(None)
):
    """
    Get Patient by ID (FHIR)
    
    Retrieve a specific patient resource by ID.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("GET", f"/fhir/Patient/{patient_id}", token=token)


@app.post("/fhir/Patient")
async def create_patient(
    patient: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None)
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
    
    params = {k: v for k, v in search.dict().items() if v is not None}
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
    
    params = {k: v for k, v in search.dict().items() if v is not None}
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


@app.post("/api/patient")
async def create_patient_standard(
    patient: PatientCreate,
    authorization: Optional[str] = Header(None)
):
    """
    Create Patient (Standard API)
    
    Create a new patient using the Standard OpenEMR API.
    """
    token = await get_access_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token required")
    
    return await make_openemr_request("POST", "/api/patient", token=token, json_data=patient.dict())


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
