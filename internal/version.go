package internal

import (
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	defaultRequestVersion  = "0.0.1"
	feVersionRefreshWindow = 30 * time.Second
	feVersionRetryWindow   = 2 * time.Second
)

var (
	feVersion          string
	requestVersion     = defaultRequestVersion
	lastRefreshAt      time.Time
	lastRefreshAttempt time.Time
	versionLock        sync.RWMutex
	refreshLock        sync.Mutex
	refreshCond        = sync.NewCond(&refreshLock)
	feVersionRe        = regexp.MustCompile(`prod-fe-[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*`)
	requestVersionRe   = regexp.MustCompile(`^\d+(?:\.\d+)+$`)
	versionClient      = &http.Client{Timeout: 10 * time.Second}
	refreshInProgress  bool
)

func GetFeVersion() string {
	versionLock.RLock()
	defer versionLock.RUnlock()
	return feVersion
}

func GetRequestVersion() string {
	versionLock.RLock()
	defer versionLock.RUnlock()
	return requestVersion
}

func fetchFeVersion() {
	resp, err := versionClient.Get("https://chat.z.ai/")
	if err != nil {
		LogError("Failed to fetch fe version: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		LogError("Failed to fetch fe version: status=%d", resp.StatusCode)
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		LogError("Failed to read fe version response: %v", err)
		return
	}

	match := feVersionRe.FindString(string(body))
	if match == "" {
		LogWarn("No fe version found in homepage")
		return
	}

	version := strings.TrimPrefix(match, "prod-fe-")
	if !requestVersionRe.MatchString(version) {
		LogWarn("Invalid request version format: %s, fallback to default", version)
		version = defaultRequestVersion
	}

	versionLock.Lock()
	feVersion = match
	requestVersion = version
	lastRefreshAt = time.Now()
	versionLock.Unlock()
	LogInfo("Updated fe version: %s, request version: %s", match, version)
}

func RefreshFeVersionIfNeeded(force bool) {
	now := time.Now()

	versionLock.RLock()
	recentlyRefreshed := !lastRefreshAt.IsZero() && now.Sub(lastRefreshAt) < feVersionRefreshWindow
	versionLock.RUnlock()
	if recentlyRefreshed && !force {
		return
	}

	refreshLock.Lock()
	for refreshInProgress {
		refreshCond.Wait()
	}

	versionLock.RLock()
	recentlyRefreshed = !lastRefreshAt.IsZero() && time.Since(lastRefreshAt) < feVersionRefreshWindow
	versionLock.RUnlock()
	if recentlyRefreshed && !force {
		refreshLock.Unlock()
		return
	}

	versionLock.RLock()
	recentlyAttempted := !lastRefreshAttempt.IsZero() && time.Since(lastRefreshAttempt) < feVersionRetryWindow
	versionLock.RUnlock()
	if recentlyAttempted && !force {
		refreshLock.Unlock()
		return
	}

	versionLock.Lock()
	lastRefreshAttempt = time.Now()
	versionLock.Unlock()

	refreshInProgress = true
	refreshLock.Unlock()

	defer func() {
		refreshLock.Lock()
		refreshInProgress = false
		refreshCond.Broadcast()
		refreshLock.Unlock()
	}()

	fetchFeVersion()
}

func StartVersionUpdater() {
	fetchFeVersion()

	ticker := time.NewTicker(1 * time.Hour)
	go func() {
		for range ticker.C {
			RefreshFeVersionIfNeeded(false)
		}
	}()
}
