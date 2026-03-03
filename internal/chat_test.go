package internal

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func assertOpenAIInvalidAPIKeyResponse(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected %d, got %d", http.StatusUnauthorized, rec.Code)
	}

	var got map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("expected JSON error body, got unmarshal error: %v", err)
	}
	errorObj, ok := got["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error object in response body")
	}
	if errorObj["code"] != "invalid_api_key" {
		t.Fatalf("expected code invalid_api_key, got %v", errorObj["code"])
	}
	if errorObj["type"] != "invalid_request_error" {
		t.Fatalf("expected type invalid_request_error, got %v", errorObj["type"])
	}
	if errorObj["message"] != "Incorrect API key provided." {
		t.Fatalf("unexpected error message: %v", errorObj["message"])
	}
	if _, exists := errorObj["param"]; !exists {
		t.Fatalf("expected param field in error object")
	}
	if errorObj["param"] != nil {
		t.Fatalf("expected param to be null, got %v", errorObj["param"])
	}
}

func assertAnthropicInvalidAPIKeyResponse(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected %d, got %d", http.StatusUnauthorized, rec.Code)
	}

	var got map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("expected JSON error body, got unmarshal error: %v", err)
	}
	if got["type"] != "error" {
		t.Fatalf("expected top-level type error, got %v", got["type"])
	}
	errorObj, ok := got["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error object in response body")
	}
	if errorObj["type"] != "authentication_error" {
		t.Fatalf("expected authentication_error, got %v", errorObj["type"])
	}
	if errorObj["message"] != "Invalid API key" {
		t.Fatalf("unexpected error message: %v", errorObj["message"])
	}
}

func TestRedactSensitiveURL(t *testing.T) {
	input := `Post "https://chat.z.ai/api/v2/chat/completions?token=abc123&user_id=u-1&current_url=https://chat.z.ai/c/xyz&pathname=/c/xyz&requestId=req-1": EOF`
	got := redactSensitiveURL(input)

	expects := []string{
		"token=REDACTED",
		"user_id=REDACTED",
		"current_url=REDACTED",
		"pathname=REDACTED",
		"requestId=req-1",
	}
	for _, expect := range expects {
		if !strings.Contains(got, expect) {
			t.Fatalf("expected redacted output to contain %q, got: %s", expect, got)
		}
	}

	if strings.Contains(got, "abc123") || strings.Contains(got, "u-1") || strings.Contains(got, "/c/xyz") {
		t.Fatalf("sensitive values should be redacted, got: %s", got)
	}
}

func TestHandleChatCompletionsRejectsMissingAuthorization(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"GLM-4.7","messages":[]}`))
	rec := httptest.NewRecorder()

	HandleChatCompletions(rec, req)

	assertOpenAIInvalidAPIKeyResponse(t, rec)
}

func TestHandleChatCompletionsRejectsNonBearerAuthorization(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"GLM-4.7","messages":[]}`))
	req.Header.Set("Authorization", "Basic abc")
	rec := httptest.NewRecorder()

	HandleChatCompletions(rec, req)

	assertOpenAIInvalidAPIKeyResponse(t, rec)
}

func TestHandleChatCompletionsRejectsBearerWithEmptyToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"GLM-4.7","messages":[]}`))
	req.Header.Set("Authorization", "Bearer ")
	rec := httptest.NewRecorder()

	HandleChatCompletions(rec, req)

	assertOpenAIInvalidAPIKeyResponse(t, rec)
}

func TestHandleChatCompletionsRejectsBearerWithSpaceInToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"GLM-4.7","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token has-space")
	rec := httptest.NewRecorder()

	HandleChatCompletions(rec, req)

	assertOpenAIInvalidAPIKeyResponse(t, rec)
}

func TestHandleChatCompletionsRejectsBearerWithTabInToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"GLM-4.7","messages":[]}`))
	req.Header.Set("Authorization", "Bearer token\tbad")
	rec := httptest.NewRecorder()

	HandleChatCompletions(rec, req)

	assertOpenAIInvalidAPIKeyResponse(t, rec)
}

func TestHandleChatCompletionsRequestBodyTooLarge(t *testing.T) {
	largeContent := strings.Repeat("a", 4*1024*1024)
	payload := map[string]interface{}{
		"model": "GLM-4.7",
		"messages": []map[string]string{
			{"role": "user", "content": largeContent},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer valid.token.value")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	HandleChatCompletions(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected %d, got %d", http.StatusRequestEntityTooLarge, rec.Code)
	}
}
