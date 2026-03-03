package internal

import "testing"

func TestGetRequestVersionDefault(t *testing.T) {
	versionLock.Lock()
	oldRequestVersion := requestVersion
	requestVersion = defaultRequestVersion
	versionLock.Unlock()
	defer func() {
		versionLock.Lock()
		requestVersion = oldRequestVersion
		versionLock.Unlock()
	}()

	got := GetRequestVersion()
	if got == "" {
		t.Fatalf("expected non-empty default request version")
	}
	if got != defaultRequestVersion {
		t.Fatalf("expected default request version %q, got %q", defaultRequestVersion, got)
	}
}
