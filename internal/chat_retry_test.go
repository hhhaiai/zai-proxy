package internal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

type closeCounterReadCloser struct {
	reader  io.Reader
	onClose func()
}

func (r *closeCounterReadCloser) Read(p []byte) (int, error) {
	return r.reader.Read(p)
}

func (r *closeCounterReadCloser) Close() error {
	if r.onClose != nil {
		r.onClose()
	}
	return nil
}

func newMockResponse(status int, onClose func()) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body: &closeCounterReadCloser{
			reader:  strings.NewReader(""),
			onClose: onClose,
		},
	}
}

func TestDoUpstreamRequestWithRetry_405Then200(t *testing.T) {
	attempts := 0
	refreshCalls := 0
	closedBodies := 0

	doRequest := func() (*http.Response, error) {
		attempts++
		switch attempts {
		case 1:
			return newMockResponse(http.StatusMethodNotAllowed, func() { closedBodies++ }), nil
		case 2:
			return newMockResponse(http.StatusOK, func() { closedBodies++ }), nil
		default:
			return nil, fmt.Errorf("unexpected extra attempt")
		}
	}

	refresh := func(force bool) {
		refreshCalls++
	}

	resp, err := doUpstreamRequestWithRetry(context.Background(), doRequest, refresh)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp == nil {
		t.Fatalf("expected non-nil response")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts)
	}
	if refreshCalls != 0 {
		t.Fatalf("expected 0 refresh calls, got %d", refreshCalls)
	}
	if closedBodies != 1 {
		t.Fatalf("expected 1 discarded body closed, got %d", closedBodies)
	}
}

func TestDoUpstreamRequestWithRetry_405RetriesExhausted(t *testing.T) {
	attempts := 0
	refreshCalls := 0
	closedBodies := 0

	doRequest := func() (*http.Response, error) {
		attempts++
		return newMockResponse(http.StatusMethodNotAllowed, func() { closedBodies++ }), nil
	}

	refresh := func(force bool) {
		refreshCalls++
	}

	resp, err := doUpstreamRequestWithRetry(context.Background(), doRequest, refresh)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp == nil {
		t.Fatalf("expected non-nil response")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", resp.StatusCode)
	}
	if attempts != maxStatus405Retries+1 {
		t.Fatalf("expected %d attempts, got %d", maxStatus405Retries+1, attempts)
	}
	if refreshCalls != 0 {
		t.Fatalf("expected 0 refresh calls, got %d", refreshCalls)
	}
	if closedBodies != maxStatus405Retries {
		t.Fatalf("expected %d discarded body close calls, got %d", maxStatus405Retries, closedBodies)
	}
}

func TestDoUpstreamRequestWithRetry_403Then200RefreshOnce(t *testing.T) {
	attempts := 0
	refreshCalls := 0
	closedBodies := 0

	doRequest := func() (*http.Response, error) {
		attempts++
		switch attempts {
		case 1:
			return newMockResponse(http.StatusForbidden, func() { closedBodies++ }), nil
		case 2:
			return newMockResponse(http.StatusOK, func() { closedBodies++ }), nil
		default:
			return nil, fmt.Errorf("unexpected extra attempt")
		}
	}

	refresh := func(force bool) {
		if !force {
			t.Fatalf("expected force refresh=true")
		}
		refreshCalls++
	}

	resp, err := doUpstreamRequestWithRetry(context.Background(), doRequest, refresh)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp == nil {
		t.Fatalf("expected non-nil response")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts)
	}
	if refreshCalls != 1 {
		t.Fatalf("expected 1 refresh call, got %d", refreshCalls)
	}
	if closedBodies != 1 {
		t.Fatalf("expected 1 discarded body close call, got %d", closedBodies)
	}
}

func TestDoUpstreamRequestWithRetry_403Then405Then200(t *testing.T) {
	attempts := 0
	refreshCalls := 0
	closedBodies := 0

	doRequest := func() (*http.Response, error) {
		attempts++
		switch attempts {
		case 1:
			return newMockResponse(http.StatusForbidden, func() { closedBodies++ }), nil
		case 2:
			return newMockResponse(http.StatusMethodNotAllowed, func() { closedBodies++ }), nil
		case 3:
			return newMockResponse(http.StatusOK, func() { closedBodies++ }), nil
		default:
			return nil, fmt.Errorf("unexpected extra attempt")
		}
	}

	refresh := func(force bool) {
		refreshCalls++
	}

	resp, err := doUpstreamRequestWithRetry(context.Background(), doRequest, refresh)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp == nil {
		t.Fatalf("expected non-nil response")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
	if refreshCalls != 1 {
		t.Fatalf("expected 1 refresh call, got %d", refreshCalls)
	}
	if closedBodies != 2 {
		t.Fatalf("expected 2 discarded body close calls, got %d", closedBodies)
	}
}

func TestDoUpstreamRequestWithRetry_RequestError(t *testing.T) {
	expectedErr := fmt.Errorf("network failure")
	attempts := 0
	refreshCalls := 0

	doRequest := func() (*http.Response, error) {
		attempts++
		return nil, expectedErr
	}

	refresh := func(force bool) {
		refreshCalls++
	}

	resp, err := doUpstreamRequestWithRetry(context.Background(), doRequest, refresh)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if err != expectedErr {
		t.Fatalf("expected error %v, got %v", expectedErr, err)
	}
	if resp != nil {
		t.Fatalf("expected nil response on error")
	}
	if attempts != 1 {
		t.Fatalf("expected 1 attempt, got %d", attempts)
	}
	if refreshCalls != 0 {
		t.Fatalf("expected 0 refresh calls, got %d", refreshCalls)
	}
}

func TestDoUpstreamRequestWithRetry_ContextCancelledBeforeRetry(t *testing.T) {
	attempts := 0
	refreshCalls := 0
	closedBodies := 0
	ctx, cancel := context.WithCancel(context.Background())

	doRequest := func() (*http.Response, error) {
		attempts++
		if attempts == 1 {
			cancel()
			return newMockResponse(http.StatusMethodNotAllowed, func() { closedBodies++ }), nil
		}
		return nil, fmt.Errorf("unexpected extra attempt")
	}

	refresh := func(force bool) {
		refreshCalls++
	}

	resp, err := doUpstreamRequestWithRetry(ctx, doRequest, refresh)
	if err == nil {
		t.Fatalf("expected context canceled error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context canceled error, got %v", err)
	}
	if resp != nil {
		t.Fatalf("expected nil response when context canceled")
	}
	if attempts != 1 {
		t.Fatalf("expected 1 attempt, got %d", attempts)
	}
	if refreshCalls != 0 {
		t.Fatalf("expected 0 refresh calls, got %d", refreshCalls)
	}
	if closedBodies != 1 {
		t.Fatalf("expected discarded body to be closed once, got %d", closedBodies)
	}
}

func TestDoUpstreamRequestWithRetry_ConcurrentRequests(t *testing.T) {
	const goroutines = 16

	errCh := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			attempts := 0
			refreshCalls := 0
			closedBodies := 0

			doRequest := func() (*http.Response, error) {
				attempts++
				switch attempts {
				case 1:
					return newMockResponse(http.StatusMethodNotAllowed, func() { closedBodies++ }), nil
				case 2:
					return newMockResponse(http.StatusMethodNotAllowed, func() { closedBodies++ }), nil
				case 3:
					return newMockResponse(http.StatusOK, func() { closedBodies++ }), nil
				default:
					return nil, fmt.Errorf("unexpected extra attempt")
				}
			}

			refresh := func(force bool) {
				refreshCalls++
			}

			resp, err := doUpstreamRequestWithRetry(context.Background(), doRequest, refresh)
			if err != nil {
				errCh <- fmt.Errorf("unexpected error: %w", err)
				return
			}
			if resp == nil {
				errCh <- fmt.Errorf("nil response")
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				errCh <- fmt.Errorf("status = %d, want 200", resp.StatusCode)
				return
			}
			if attempts != 3 {
				errCh <- fmt.Errorf("attempts = %d, want 3", attempts)
				return
			}
			if refreshCalls != 0 {
				errCh <- fmt.Errorf("refreshCalls = %d, want 0", refreshCalls)
				return
			}
			if closedBodies != 2 {
				errCh <- fmt.Errorf("closedBodies = %d, want 2", closedBodies)
				return
			}
			errCh <- nil
		}()
	}

	for i := 0; i < goroutines; i++ {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}
