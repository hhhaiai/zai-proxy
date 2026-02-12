package internal

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/google/uuid"
)

// ==================== Anthropic Request Types ====================

type AnthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	MaxTokens     int                `json:"max_tokens"`
	System        interface{}        `json:"system,omitempty"` // string or []AnthropicContentBlock
	Stream        bool               `json:"stream,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []AnthropicContentBlock
}

type AnthropicContentBlock struct {
	Type   string                `json:"type"`
	Text   string                `json:"text,omitempty"`
	Source *AnthropicImageSource `json:"source,omitempty"`
}

type AnthropicImageSource struct {
	Type      string `json:"type"`       // "base64" or "url"
	MediaType string `json:"media_type"` // e.g. "image/jpeg"
	Data      string `json:"data"`       // base64 data or URL
}

// ==================== Anthropic Response Types ====================

type AnthropicResponse struct {
	ID           string                     `json:"id"`
	Type         string                     `json:"type"`
	Role         string                     `json:"role"`
	Content      []AnthropicContentBlockOut `json:"content"`
	Model        string                     `json:"model"`
	StopReason   *string                    `json:"stop_reason"`
	StopSequence *string                    `json:"stop_sequence"`
	Usage        AnthropicUsage             `json:"usage"`
}

type AnthropicContentBlockOut struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ==================== Anthropic SSE Event Types ====================

type AnthropicStreamEvent struct {
	Type    string      `json:"type"`
	Message interface{} `json:"message,omitempty"`
	Index   *int        `json:"index,omitempty"`

	ContentBlock interface{} `json:"content_block,omitempty"`
	Delta        interface{} `json:"delta,omitempty"`
	Usage        interface{} `json:"usage,omitempty"`
}

// ==================== Request Conversion ====================

// convertAnthropicMessages converts Anthropic messages to internal Message format
func convertAnthropicMessages(req *AnthropicRequest) []Message {
	var messages []Message

	// Handle system prompt
	if req.System != nil {
		systemText := extractSystemText(req.System)
		if systemText != "" {
			messages = append(messages, Message{
				Role:    "system",
				Content: systemText,
			})
		}
	}

	// Convert each message
	for _, msg := range req.Messages {
		text, imageURLs := parseAnthropicContent(msg.Content)

		if len(imageURLs) == 0 {
			messages = append(messages, Message{
				Role:    msg.Role,
				Content: text,
			})
		} else {
			// Build multimodal content
			var parts []interface{}
			if text != "" {
				parts = append(parts, map[string]interface{}{
					"type": "text",
					"text": text,
				})
			}
			for _, imgURL := range imageURLs {
				parts = append(parts, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": imgURL,
					},
				})
			}
			messages = append(messages, Message{
				Role:    msg.Role,
				Content: parts,
			})
		}
	}

	return messages
}

func extractSystemText(system interface{}) string {
	switch s := system.(type) {
	case string:
		return s
	case []interface{}:
		var parts []string
		for _, item := range s {
			if block, ok := item.(map[string]interface{}); ok {
				if t, ok := block["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func parseAnthropicContent(content interface{}) (text string, imageURLs []string) {
	switch c := content.(type) {
	case string:
		return c, nil
	case []interface{}:
		var textParts []string
		for _, item := range c {
			if block, ok := item.(map[string]interface{}); ok {
				blockType, _ := block["type"].(string)
				switch blockType {
				case "text":
					if t, ok := block["text"].(string); ok {
						textParts = append(textParts, t)
					}
				case "image":
					if source, ok := block["source"].(map[string]interface{}); ok {
						srcType, _ := source["type"].(string)
						if srcType == "base64" {
							mediaType, _ := source["media_type"].(string)
							data, _ := source["data"].(string)
							if mediaType != "" && data != "" {
								imageURLs = append(imageURLs, fmt.Sprintf("data:%s;base64,%s", mediaType, data))
							}
						} else if srcType == "url" {
							if url, ok := source["url"].(string); ok {
								imageURLs = append(imageURLs, url)
							}
						}
					}
				}
			}
		}
		return strings.Join(textParts, "\n"), imageURLs
	}
	return "", nil
}

// resolveAnthropicModel always uses GLM-5-thinking-search for Anthropic endpoint
func resolveAnthropicModel(model string) string {
	return "GLM-5-thinking-search"
}

// ==================== Handler ====================

func HandleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	// Extract token from x-api-key or Authorization header
	token := r.Header.Get("x-api-key")
	if token == "" {
		token = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}
	if token == "" {
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "Missing API key")
		return
	}

	if token == "free" {
		anonymousToken, err := GetAnonymousToken()
		if err != nil {
			LogError("[Anthropic] Failed to get anonymous token: %v", err)
			writeAnthropicError(w, http.StatusInternalServerError, "api_error", "Failed to get anonymous token")
			return
		}
		token = anonymousToken
	}

	var req AnthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Invalid JSON request body")
		return
	}

	if len(req.Messages) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "messages is required")
		return
	}

	// Save original model name for response (echo back what client sent)
	clientModel := req.Model

	// Resolve model name (internally always uses GLM-5-thinking-search)
	req.Model = resolveAnthropicModel(req.Model)

	// Convert to internal message format
	messages := convertAnthropicMessages(&req)

	isGLM5 := IsGLM5Model(req.Model)

	// Make upstream request (reuse existing logic)
	resp, _, err := makeUpstreamRequest(token, messages, req.Model)
	if err != nil {
		LogError("[Anthropic] Upstream request failed: %v", err)
		writeAnthropicError(w, http.StatusBadGateway, "api_error", "Upstream service error")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)
		if len(bodyStr) > 500 {
			bodyStr = bodyStr[:500]
		}
		LogError("[Anthropic] Upstream error: status=%d, body=%s", resp.StatusCode, bodyStr)
		writeAnthropicError(w, resp.StatusCode, "api_error", "Upstream error")
		return
	}

	msgID := fmt.Sprintf("msg_%s", uuid.New().String()[:24])

	if req.Stream {
		handleAnthropicStream(w, resp.Body, msgID, clientModel, isGLM5)
	} else {
		handleAnthropicNonStream(w, resp.Body, msgID, clientModel, isGLM5)
	}
}

// ==================== Streaming Response ====================

func sendAnthropicSSE(w http.ResponseWriter, flusher http.Flusher, eventType string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, jsonData)
	flusher.Flush()
}

func handleAnthropicStream(w http.ResponseWriter, body io.ReadCloser, msgID, modelName string, isGLM5 bool) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// 1. message_start
	sendAnthropicSSE(w, flusher, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":            msgID,
			"type":          "message",
			"role":          "assistant",
			"content":       []interface{}{},
			"model":         modelName,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]interface{}{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})

	// Track content block indices
	blockIndex := 0
	thinkingBlockStarted := false
	textBlockStarted := false
	outputTokens := 0

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	searchRefFilter := NewSearchRefFilter()
	thinkingFilter := &ThinkingFilter{}
	totalContentOutputLength := 0

	for scanner.Scan() {
		line := scanner.Text()
		LogDebug("[Anthropic][Upstream] %s", line)

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var upstream UpstreamData
		if err := json.Unmarshal([]byte(payload), &upstream); err != nil {
			continue
		}

		if upstream.Data.Phase == "done" {
			break
		}

		// Skip usage-only data
		if upstream.Data.Phase == "other" && upstream.Data.DeltaContent == "" && upstream.GetEditContent() == "" {
			continue
		}

		// Handle thinking content
		if upstream.Data.Phase == "thinking" && upstream.Data.DeltaContent != "" {
			if !thinkingBlockStarted {
				thinkingBlockStarted = true
				idx := blockIndex
				sendAnthropicSSE(w, flusher, "content_block_start", map[string]interface{}{
					"type":  "content_block_start",
					"index": idx,
					"content_block": map[string]interface{}{
						"type":     "thinking",
						"thinking": "",
					},
				})
				blockIndex++
			}

			var reasoningContent string
			if isGLM5 {
				reasoningContent = searchRefFilter.Process(upstream.Data.DeltaContent)
			} else {
				if thinkingFilter.lastPhase != "" && thinkingFilter.lastPhase != "thinking" {
					thinkingFilter.ResetForNewRound()
				}
				thinkingFilter.lastPhase = "thinking"
				reasoningContent = thinkingFilter.ProcessThinking(upstream.Data.DeltaContent)
				if reasoningContent != "" {
					thinkingFilter.lastOutputChunk = reasoningContent
					reasoningContent = searchRefFilter.Process(reasoningContent)
				}
			}

			if reasoningContent != "" {
				outputTokens += len(reasoningContent) / 4 // rough estimate
				idx := blockIndex - 1
				sendAnthropicSSE(w, flusher, "content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": idx,
					"delta": map[string]interface{}{
						"type":     "thinking_delta",
						"thinking": reasoningContent,
					},
				})
			}
			continue
		}

		if upstream.Data.Phase != "" {
			thinkingFilter.lastPhase = upstream.Data.Phase
		}

		// Handle search results, image search, mcp, tool_call (skip like OpenAI handler)
		editContent := upstream.GetEditContent()
		if editContent != "" && IsSearchResultContent(editContent) {
			if results := ParseSearchResults(editContent); len(results) > 0 {
				searchRefFilter.AddSearchResults(results)
			}
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"search_image"`) {
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"mcp"`) {
			textBeforeBlock := ExtractTextBeforeGlmBlock(editContent)
			if textBeforeBlock != "" {
				textBeforeBlock = searchRefFilter.Process(textBeforeBlock)
				if textBeforeBlock != "" {
					ensureTextBlock(w, flusher, &textBlockStarted, &blockIndex, &thinkingBlockStarted)
					outputTokens += len(textBeforeBlock) / 4
					idx := blockIndex - 1
					sendAnthropicSSE(w, flusher, "content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": idx,
						"delta": map[string]interface{}{
							"type": "text_delta",
							"text": textBeforeBlock,
						},
					})
				}
			}
			continue
		}
		if editContent != "" && IsSearchToolCall(editContent, upstream.Data.Phase) {
			continue
		}

		// Flush thinking buffer
		if thinkingRemaining := thinkingFilter.Flush(); thinkingRemaining != "" {
			thinkingFilter.lastOutputChunk = thinkingRemaining
			processedRemaining := searchRefFilter.Process(thinkingRemaining)
			if processedRemaining != "" && thinkingBlockStarted {
				idx := blockIndex - 1
				if textBlockStarted {
					idx = blockIndex - 2
				}
				sendAnthropicSSE(w, flusher, "content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": idx,
					"delta": map[string]interface{}{
						"type":     "thinking_delta",
						"thinking": processedRemaining,
					},
				})
			}
		}

		// Extract content
		content := ""
		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			content = upstream.Data.DeltaContent
		} else if upstream.Data.Phase == "answer" && editContent != "" {
			if strings.Contains(editContent, "</details>") {
				if idx := strings.Index(editContent, "</details>"); idx != -1 {
					afterDetails := editContent[idx+len("</details>"):]
					if strings.HasPrefix(afterDetails, "\n") {
						content = afterDetails[1:]
					} else {
						content = afterDetails
					}
					totalContentOutputLength = len([]rune(content))
				}
			}
		} else if (upstream.Data.Phase == "other" || upstream.Data.Phase == "tool_call") && editContent != "" {
			fullContent := editContent
			fullContentRunes := []rune(fullContent)
			if len(fullContentRunes) > totalContentOutputLength {
				content = string(fullContentRunes[totalContentOutputLength:])
				totalContentOutputLength = len(fullContentRunes)
			} else {
				content = fullContent
			}
		}

		if content == "" {
			continue
		}

		content = searchRefFilter.Process(content)
		if content == "" {
			continue
		}

		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			totalContentOutputLength += len([]rune(content))
		}

		// Close thinking block if needed, start text block
		ensureTextBlock(w, flusher, &textBlockStarted, &blockIndex, &thinkingBlockStarted)

		outputTokens += len(content) / 4
		idx := blockIndex - 1
		sendAnthropicSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": idx,
			"delta": map[string]interface{}{
				"type": "text_delta",
				"text": content,
			},
		})
	}

	// Flush remaining search ref content
	if remaining := searchRefFilter.Flush(); remaining != "" {
		ensureTextBlock(w, flusher, &textBlockStarted, &blockIndex, &thinkingBlockStarted)
		idx := blockIndex - 1
		sendAnthropicSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": idx,
			"delta": map[string]interface{}{
				"type": "text_delta",
				"text": remaining,
			},
		})
	}

	// If no text block was started, start one with empty content
	if !textBlockStarted && !thinkingBlockStarted {
		ensureTextBlock(w, flusher, &textBlockStarted, &blockIndex, &thinkingBlockStarted)
	}

	// Close thinking block if still open
	if thinkingBlockStarted && !textBlockStarted {
		// Close thinking block
		thinkingIdx := blockIndex - 1
		sendAnthropicSSE(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": thinkingIdx,
		})
		// Start and close a text block (Anthropic always expects at least one text block)
		ensureTextBlock(w, flusher, &textBlockStarted, &blockIndex, &thinkingBlockStarted)
	}

	// Close the last content block
	lastIdx := blockIndex - 1
	sendAnthropicSSE(w, flusher, "content_block_stop", map[string]interface{}{
		"type":  "content_block_stop",
		"index": lastIdx,
	})

	// message_delta
	sendAnthropicSSE(w, flusher, "message_delta", map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": outputTokens,
		},
	})

	// message_stop
	sendAnthropicSSE(w, flusher, "message_stop", map[string]interface{}{
		"type": "message_stop",
	})
}

// ensureTextBlock closes thinking block if needed and starts a text block
func ensureTextBlock(w http.ResponseWriter, flusher http.Flusher, textBlockStarted *bool, blockIndex *int, thinkingBlockStarted *bool) {
	if *textBlockStarted {
		return
	}

	// Close thinking block first if open
	if *thinkingBlockStarted {
		thinkingIdx := *blockIndex - 1
		sendAnthropicSSE(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": thinkingIdx,
		})
	}

	// Start text block
	idx := *blockIndex
	sendAnthropicSSE(w, flusher, "content_block_start", map[string]interface{}{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]interface{}{
			"type": "text",
			"text": "",
		},
	})
	*blockIndex++
	*textBlockStarted = true
}

// ==================== Non-Streaming Response ====================

func handleAnthropicNonStream(w http.ResponseWriter, body io.ReadCloser, msgID, modelName string, isGLM5 bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	var chunks []string
	var reasoningChunks []string
	thinkingFilter := &ThinkingFilter{}
	searchRefFilter := NewSearchRefFilter()
	hasThinking := false

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var upstream UpstreamData
		if err := json.Unmarshal([]byte(payload), &upstream); err != nil {
			continue
		}

		if upstream.Data.Phase == "done" {
			break
		}

		// Skip usage-only data
		if upstream.Data.Phase == "other" && upstream.Data.DeltaContent == "" && upstream.GetEditContent() == "" {
			continue
		}

		if upstream.Data.Phase == "thinking" && upstream.Data.DeltaContent != "" {
			hasThinking = true
			if isGLM5 {
				reasoningChunks = append(reasoningChunks, upstream.Data.DeltaContent)
			} else {
				if thinkingFilter.lastPhase != "" && thinkingFilter.lastPhase != "thinking" {
					thinkingFilter.ResetForNewRound()
				}
				thinkingFilter.lastPhase = "thinking"
				reasoningContent := thinkingFilter.ProcessThinking(upstream.Data.DeltaContent)
				if reasoningContent != "" {
					thinkingFilter.lastOutputChunk = reasoningContent
					reasoningChunks = append(reasoningChunks, reasoningContent)
				}
			}
			continue
		}

		if upstream.Data.Phase != "" {
			thinkingFilter.lastPhase = upstream.Data.Phase
		}

		editContent := upstream.GetEditContent()
		if editContent != "" && IsSearchResultContent(editContent) {
			if results := ParseSearchResults(editContent); len(results) > 0 {
				searchRefFilter.AddSearchResults(results)
			}
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"search_image"`) {
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"mcp"`) {
			textBeforeBlock := ExtractTextBeforeGlmBlock(editContent)
			if textBeforeBlock != "" {
				chunks = append(chunks, textBeforeBlock)
			}
			continue
		}
		if editContent != "" && IsSearchToolCall(editContent, upstream.Data.Phase) {
			continue
		}

		content := ""
		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			content = upstream.Data.DeltaContent
		} else if upstream.Data.Phase == "answer" && editContent != "" {
			if strings.Contains(editContent, "</details>") {
				reasoningContent := thinkingFilter.ExtractIncrementalThinking(editContent)
				if reasoningContent != "" {
					reasoningChunks = append(reasoningChunks, reasoningContent)
				}
				if idx := strings.Index(editContent, "</details>"); idx != -1 {
					afterDetails := editContent[idx+len("</details>"):]
					if strings.HasPrefix(afterDetails, "\n") {
						content = afterDetails[1:]
					} else {
						content = afterDetails
					}
				}
			}
		} else if (upstream.Data.Phase == "other" || upstream.Data.Phase == "tool_call") && editContent != "" {
			content = editContent
		}

		if content != "" {
			chunks = append(chunks, content)
		}
	}

	fullContent := strings.Join(chunks, "")
	fullContent = searchRefFilter.Process(fullContent) + searchRefFilter.Flush()
	fullReasoning := strings.Join(reasoningChunks, "")
	fullReasoning = searchRefFilter.Process(fullReasoning) + searchRefFilter.Flush()

	// Build content blocks
	var contentBlocks []AnthropicContentBlockOut
	if hasThinking && fullReasoning != "" {
		contentBlocks = append(contentBlocks, AnthropicContentBlockOut{
			Type:     "thinking",
			Thinking: fullReasoning,
		})
	}
	contentBlocks = append(contentBlocks, AnthropicContentBlockOut{
		Type: "text",
		Text: fullContent,
	})

	stopReason := "end_turn"
	outputTokens := (len(fullContent) + len(fullReasoning)) / 4 // rough estimate

	response := AnthropicResponse{
		ID:         msgID,
		Type:       "message",
		Role:       "assistant",
		Content:    contentBlocks,
		Model:      modelName,
		StopReason: &stopReason,
		Usage: AnthropicUsage{
			InputTokens:  0,
			OutputTokens: outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ==================== Error Response ====================

func writeAnthropicError(w http.ResponseWriter, statusCode int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    errType,
			"message": message,
		},
	})
}

func HandleModelsAnthropic(w http.ResponseWriter, r *http.Request) {
	// Return models in a format Claude Code might expect
	HandleModels(w, r)
}
