"""
Test suite for all POST methods in main.py

Run with: pytest test_post_methods.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json
from main import app

client = TestClient(app)


class TestPostMethods:
    """Test all POST endpoints in the FastAPI application"""

    @pytest.fixture
    def mock_token_response(self):
        """Mock OAuth token response"""
        return {
            "access_token": "test_access_token_12345",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test_refresh_token_12345",
            "scope": "openid api:fhir patient/Patient.rs",
            "id_token": "test_id_token"
        }

    @pytest.fixture
    def mock_client_registration_response(self):
        """Mock OAuth client registration response"""
        return {
            "client_id": "test_client_id_12345",
            "client_secret": "test_client_secret_67890",
            "client_id_issued_at": 1234567890,
            "client_secret_expires_at": 0,
            "redirect_uris": ["http://localhost:8000/oauth/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "client_name": "Test FastAPI App",
            "scope": "openid api:fhir patient/Patient.rs"
        }

    @pytest.fixture
    def mock_fhir_patient_response(self):
        """Mock FHIR Patient resource response"""
        return {
            "resourceType": "Patient",
            "id": "test-patient-123",
            "identifier": [
                {
                    "system": "http://hospital.example.org",
                    "value": "123456"
                }
            ],
            "name": [
                {
                    "family": "Doe",
                    "given": ["John"]
                }
            ],
            "birthDate": "1990-01-01",
            "gender": "male"
        }

    @pytest.fixture
    def mock_standard_patient_response(self):
        """Mock Standard API patient response"""
        return {
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

    # ============================================
    # Test POST /oauth/token
    # ============================================
    
    def test_post_oauth_token_authorization_code(self, mock_token_response):
        """Test POST /oauth/token with authorization_code grant"""
        async def mock_post(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_token_response
            mock_resp.raise_for_status = AsyncMock()
            return mock_resp
        
        with patch('main.httpx.AsyncClient') as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=mock_post)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance
            
            response = client.post(
                "/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "code": "test_auth_code_123",
                    "redirect_uri": "http://localhost:8000/oauth/callback"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "Bearer"
            assert "expires_in" in data

    def test_post_oauth_token_refresh_token(self, mock_token_response):
        """Test POST /oauth/token with refresh_token grant"""
        async def mock_post(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_token_response
            mock_resp.raise_for_status = AsyncMock()
            return mock_resp
        
        with patch('main.httpx.AsyncClient') as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=mock_post)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance
            
            response = client.post(
                "/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": "test_refresh_token_12345"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data

    def test_post_oauth_token_missing_code(self):
        """Test POST /oauth/token with missing code for authorization_code grant"""
        response = client.post(
            "/oauth/token",
            json={
                "grant_type": "authorization_code",
                "redirect_uri": "http://localhost:8000/oauth/callback"
            }
        )
        
        assert response.status_code == 400
        assert "code is required" in response.json()["detail"].lower()

    def test_post_oauth_token_missing_refresh_token(self):
        """Test POST /oauth/token with missing refresh_token for refresh_token grant"""
        response = client.post(
            "/oauth/token",
            json={
                "grant_type": "refresh_token"
            }
        )
        
        assert response.status_code == 400
        assert "refresh_token is required" in response.json()["detail"].lower()

    def test_post_oauth_token_with_pkce(self, mock_token_response):
        """Test POST /oauth/token with PKCE code verifier"""
        async def mock_post(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_token_response
            mock_resp.raise_for_status = AsyncMock()
            return mock_resp
        
        with patch('main.httpx.AsyncClient') as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=mock_post)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance
            
            response = client.post(
                "/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "code": "test_auth_code_123",
                    "redirect_uri": "http://localhost:8000/oauth/callback",
                    "code_verifier": "test_code_verifier_12345"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data

    # ============================================
    # Test POST /oauth/register
    # ============================================
    
    def test_post_oauth_register_success(self, mock_client_registration_response):
        """Test POST /oauth/register - successful client registration"""
        async def mock_post(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_client_registration_response
            mock_resp.raise_for_status = AsyncMock()
            return mock_resp
        
        with patch('main.httpx.AsyncClient') as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=mock_post)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance
            
            response = client.post(
                "/oauth/register",
                json={
                    "client_name": "Test FastAPI App",
                    "redirect_uris": ["http://localhost:8000/oauth/callback"],
                    "scope": "openid api:fhir patient/Patient.rs",
                    "token_endpoint_auth_method": "client_secret_basic"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "client_id" in data
            assert "client_secret" in data
            assert data["client_name"] == "Test FastAPI App"

    def test_post_oauth_register_minimal(self, mock_client_registration_response):
        """Test POST /oauth/register with minimal required fields"""
        async def mock_post(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_client_registration_response
            mock_resp.raise_for_status = AsyncMock()
            return mock_resp
        
        with patch('main.httpx.AsyncClient') as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=mock_post)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance
            
            response = client.post(
                "/oauth/register",
                json={
                    "client_name": "Minimal App",
                    "redirect_uris": ["http://localhost:8000/callback"]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "client_id" in data

    def test_post_oauth_register_validation_error(self):
        """Test POST /oauth/register with validation error (missing required fields)"""
        response = client.post(
            "/oauth/register",
            json={
                "client_name": "Test App"
                # Missing redirect_uris
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_post_oauth_register_openemr_error(self):
        """Test POST /oauth/register when OpenEMR returns an error"""
        with patch('main.httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = Exception("HTTP 400")
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "invalid_client_metadata"}
            mock_response.content = b'{"error": "invalid_client_metadata"}'
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            # Mock HTTPStatusError
            from httpx import HTTPStatusError
            import httpx
            
            with patch.object(httpx, 'HTTPStatusError', side_effect=HTTPStatusError("Bad Request", request=None, response=mock_response)):
                response = client.post(
                    "/oauth/register",
                    json={
                        "client_name": "Test App",
                        "redirect_uris": ["http://localhost:8000/callback"]
                    }
                )
                
                # Should handle the error gracefully
                assert response.status_code in [400, 500]

    # ============================================
    # Test POST /fhir/Patient
    # ============================================
    
    def test_post_fhir_patient_success(self, mock_fhir_patient_response):
        """Test POST /fhir/Patient - create patient successfully"""
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_fhir_patient_response
            
            patient_data = {
                "resourceType": "Patient",
                "name": [
                    {
                        "family": "Doe",
                        "given": ["John"]
                    }
                ],
                "birthDate": "1990-01-01",
                "gender": "male"
            }
            
            response = client.post(
                "/fhir/Patient",
                json=patient_data,
                headers={"Authorization": "Bearer test_token_123"}
            )
            
            assert response.status_code == 201  # Created
            data = response.json()
            assert data["resourceType"] == "Patient"
            assert "id" in data
            mock_request.assert_called_once()

    def test_post_fhir_patient_missing_auth(self):
        """Test POST /fhir/Patient without authorization header"""
        patient_data = {
            "resourceType": "Patient",
            "name": [{"family": "Doe", "given": ["John"]}]
        }
        
        response = client.post(
            "/fhir/Patient",
            json=patient_data
        )
        
        assert response.status_code == 401
        assert "Authorization" in response.json()["detail"]

    def test_post_fhir_patient_invalid_token(self):
        """Test POST /fhir/Patient with invalid token"""
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            from fastapi import HTTPException
            mock_request.side_effect = HTTPException(status_code=401, detail="Invalid token")
            
            patient_data = {
                "resourceType": "Patient",
                "name": [{"family": "Doe", "given": ["John"]}]
            }
            
            response = client.post(
                "/fhir/Patient",
                json=patient_data,
                headers={"Authorization": "Bearer invalid_token"}
            )
            
            assert response.status_code == 401

    def test_post_fhir_patient_complete_data(self, mock_fhir_patient_response):
        """Test POST /fhir/Patient with complete patient data"""
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_fhir_patient_response
            
            patient_data = {
                "resourceType": "Patient",
                "identifier": [
                    {
                        "system": "http://hospital.example.org",
                        "value": "123456"
                    }
                ],
                "name": [
                    {
                        "family": "Smith",
                        "given": ["Jane", "Marie"]
                    }
                ],
                "birthDate": "1985-05-15",
                "gender": "female",
                "telecom": [
                    {
                        "system": "phone",
                        "value": "555-1234"
                    },
                    {
                        "system": "email",
                        "value": "jane.smith@example.com"
                    }
                ],
                "address": [
                    {
                        "line": ["123 Main St"],
                        "city": "Springfield",
                        "state": "IL",
                        "postalCode": "62701",
                        "country": "USA"
                    }
                ]
            }
            
            response = client.post(
                "/fhir/Patient",
                json=patient_data,
                headers={"Authorization": "Bearer test_token_123"}
            )
            
            assert response.status_code == 201  # Created
            # Verify the request was made with correct data
            call_args = mock_request.call_args
            assert call_args[1]["json_data"] == patient_data

    # ============================================
    # Test POST /api/patient
    # ============================================
    
    def test_post_api_patient_success(self, mock_standard_patient_response):
        """Test POST /api/patient - create patient successfully (Standard API)"""
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_standard_patient_response
            
            patient_data = {
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
            
            response = client.post(
                "/api/patient",
                json=patient_data,
                headers={"Authorization": "Bearer test_token_123"}
            )
            
            assert response.status_code == 201  # Created
            data = response.json()
            assert "data" in data
            mock_request.assert_called_once()

    def test_post_api_patient_minimal_required(self, mock_standard_patient_response):
        """Test POST /api/patient with only required fields"""
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_standard_patient_response
            
            patient_data = {
                "fname": "Jane",
                "lname": "Smith",
                "dob": "1985-05-15"
            }
            
            response = client.post(
                "/api/patient",
                json=patient_data,
                headers={"Authorization": "Bearer test_token_123"}
            )
            
            assert response.status_code == 201  # Created
            # Verify required fields were sent
            call_args = mock_request.call_args
            sent_data = call_args[1]["json_data"]
            assert sent_data["fname"] == "Jane"
            assert sent_data["lname"] == "Smith"
            assert sent_data["dob"] == "1985-05-15"

    def test_post_api_patient_missing_required_field(self):
        """Test POST /api/patient with missing required field"""
        patient_data = {
            "fname": "John"
            # Missing lname and dob
        }
        
        response = client.post(
            "/api/patient",
            json=patient_data,
            headers={"Authorization": "Bearer test_token_123"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_post_api_patient_missing_auth(self):
        """Test POST /api/patient without authorization header"""
        patient_data = {
            "fname": "John",
            "lname": "Doe",
            "dob": "1990-01-01"
        }
        
        response = client.post(
            "/api/patient",
            json=patient_data
        )
        
        assert response.status_code == 401
        assert "Authorization" in response.json()["detail"]

    def test_post_api_patient_all_fields(self, mock_standard_patient_response):
        """Test POST /api/patient with all optional fields"""
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_standard_patient_response
            
            patient_data = {
                "fname": "Robert",
                "lname": "Johnson",
                "dob": "1975-03-20",
                "sex": "Male",
                "street": "456 Oak Avenue",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60601",
                "phone_cell": "555-9876",
                "email": "robert.johnson@example.com"
            }
            
            response = client.post(
                "/api/patient",
                json=patient_data,
                headers={"Authorization": "Bearer test_token_123"}
            )
            
            assert response.status_code == 201  # Created
            # Verify all fields were sent
            call_args = mock_request.call_args
            sent_data = call_args[1]["json_data"]
            assert sent_data == patient_data

    def test_post_api_patient_invalid_sex(self):
        """Test POST /api/patient with invalid sex value"""
        patient_data = {
            "fname": "Test",
            "lname": "User",
            "dob": "1990-01-01",
            "sex": "InvalidSex"  # Should still work, just stored as-is
        }
        
        # Note: The model allows any string, so this should pass validation
        # but might fail at OpenEMR level
        with patch('main.make_openemr_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {"pid": "123"}}
            
            response = client.post(
                "/api/patient",
                json=patient_data,
                headers={"Authorization": "Bearer test_token_123"}
            )
            
            # Should pass validation (422) but might fail at API level
            assert response.status_code in [201, 400, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
