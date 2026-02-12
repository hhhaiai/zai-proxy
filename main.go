package main

import (
	"net/http"

	"zai-proxy/internal"
)

func main() {
	internal.LoadConfig()
	internal.InitLogger()
	internal.StartVersionUpdater()

	http.HandleFunc("/v1/models", internal.HandleModels)
	http.HandleFunc("/v1/chat/completions", internal.HandleChatCompletions)
	http.HandleFunc("/v1/messages", internal.HandleAnthropicMessages)

	addr := ":" + internal.Cfg.Port
	internal.LogInfo("Server starting on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		internal.LogError("Server failed: %v", err)
	}
}
