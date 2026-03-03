package internal

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandleAnthropicMessagesRejectsMissingAuthorization(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()

	HandleAnthropicMessages(rec, req)

	assertAnthropicInvalidAPIKeyResponse(t, rec)
}

func TestHandleAnthropicMessagesRejectsNonBearerAuthorization(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Basic abc")
	rec := httptest.NewRecorder()

	HandleAnthropicMessages(rec, req)

	assertAnthropicInvalidAPIKeyResponse(t, rec)
}

func TestHandleAnthropicMessagesRejectsBearerWithEmptyToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer ")
	rec := httptest.NewRecorder()

	HandleAnthropicMessages(rec, req)

	assertAnthropicInvalidAPIKeyResponse(t, rec)
}

func TestHandleAnthropicMessagesRejectsBearerWithSpaceInToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer token with-space")
	rec := httptest.NewRecorder()

	HandleAnthropicMessages(rec, req)

	assertAnthropicInvalidAPIKeyResponse(t, rec)
}

func TestHandleAnthropicMessagesRejectsBearerWithTabInToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer token\tbad")
	rec := httptest.NewRecorder()

	HandleAnthropicMessages(rec, req)

	assertAnthropicInvalidAPIKeyResponse(t, rec)
}

func TestHandleAnthropicMessagesRequestBodyTooLarge(t *testing.T) {
	largeContent := strings.Repeat("a", 4*1024*1024)
	payload := map[string]interface{}{
		"model": "claude-sonnet-4-5",
		"messages": []map[string]string{
			{"role": "user", "content": largeContent},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	req.Header.Set("x-api-key", "free")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	HandleAnthropicMessages(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected %d, got %d", http.StatusRequestEntityTooLarge, rec.Code)
	}
}
