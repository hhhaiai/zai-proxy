package internal

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/corpix/uarand"
	"github.com/google/uuid"
)

func extractLatestUserContent(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			text, _ := messages[i].ParseContent()
			return text
		}
	}
	return ""
}

func extractAllImageURLs(messages []Message) []string {
	var allImageURLs []string
	for _, msg := range messages {
		_, imageURLs := msg.ParseContent()
		allImageURLs = append(allImageURLs, imageURLs...)
	}
	return allImageURLs
}

func makeUpstreamRequest(token string, messages []Message, model string) (*http.Response, string, error) {
	payload, err := DecodeJWTPayload(token)
	if err != nil || payload == nil {
		return nil, "", fmt.Errorf("invalid token")
	}

	userID := payload.ID
	chatID := uuid.New().String()
	timestamp := time.Now().UnixMilli()
	requestID := uuid.New().String()
	userMsgID := uuid.New().String()

	targetModel := GetTargetModel(model)
	latestUserContent := extractLatestUserContent(messages)
	imageURLs := extractAllImageURLs(messages)

	signature := GenerateSignature(userID, requestID, latestUserContent, timestamp)

	url := fmt.Sprintf("https://chat.z.ai/api/v2/chat/completions?timestamp=%d&requestId=%s&user_id=%s&version=0.0.1&platform=web&token=%s&current_url=%s&pathname=%s&signature_timestamp=%d",
		timestamp, requestID, userID, token,
		fmt.Sprintf("https://chat.z.ai/c/%s", chatID),
		fmt.Sprintf("/c/%s", chatID),
		timestamp)

	enableThinking := IsThinkingModel(model)
	autoWebSearch := IsSearchModel(model)
	isGLM5 := IsGLM5Model(model)
	if targetModel == "glm-4.5v" || targetModel == "glm-4.6v" {
		autoWebSearch = false
	}

	var mcpServers []string
	if targetModel == "glm-4.6v" {
		mcpServers = []string{"vlm-image-search", "vlm-image-recognition", "vlm-image-processing"}
	}
	if isGLM5 {
		mcpServers = []string{"advanced-search"}
	}

	urlToFileID := make(map[string]string)
	var filesData []map[string]interface{}
	if len(imageURLs) > 0 {
		files, _ := UploadImages(token, imageURLs)
		for i, f := range files {
			if i < len(imageURLs) {
				urlToFileID[imageURLs[i]] = f.ID
			}
			filesData = append(filesData, map[string]interface{}{
				"type":            f.Type,
				"file":            f.File,
				"id":              f.ID,
				"url":             f.URL,
				"name":            f.Name,
				"status":          f.Status,
				"size":            f.Size,
				"error":           f.Error,
				"itemId":          f.ItemID,
				"media":           f.Media,
				"ref_user_msg_id": userMsgID,
			})
		}
	}

	var upstreamMessages []map[string]interface{}
	for _, msg := range messages {
		upstreamMessages = append(upstreamMessages, msg.ToUpstreamMessage(urlToFileID))
	}

	features := map[string]interface{}{
		"image_generation": false,
		"web_search":       false,
		"auto_web_search":  autoWebSearch,
		"preview_mode":     true,
		"enable_thinking":  enableThinking,
	}
	if isGLM5 {
		features["flags"] = []interface{}{}
	}

	body := map[string]interface{}{
		"stream":                         true,
		"model":                          targetModel,
		"messages":                       upstreamMessages,
		"signature_prompt":               latestUserContent,
		"params":                         map[string]interface{}{},
		"extra":                          map[string]interface{}{},
		"features":                       features,
		"chat_id":                        chatID,
		"id":                             uuid.New().String(),
		"current_user_message_id":        userMsgID,
		"current_user_message_parent_id": nil,
		"background_tasks": map[string]interface{}{
			"title_generation": true,
			"tags_generation":  true,
		},
	}

	if len(mcpServers) > 0 {
		body["mcp_servers"] = mcpServers
	}

	if isGLM5 {
		now := time.Now()
		loc, _ := time.LoadLocation("Asia/Shanghai")
		if loc != nil {
			now = now.In(loc)
		}
		body["variables"] = map[string]interface{}{
			"{{USER_NAME}}":        "Guest",
			"{{USER_LOCATION}}":    "Unknown",
			"{{CURRENT_DATETIME}}": now.Format("2006-01-02 15:04:05"),
			"{{CURRENT_DATE}}":     now.Format("2006-01-02"),
			"{{CURRENT_TIME}}":     now.Format("15:04:05"),
			"{{CURRENT_WEEKDAY}}":  now.Weekday().String(),
			"{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
			"{{USER_LANGUAGE}}":    "en-US",
		}
	}

	if len(filesData) > 0 {
		body["files"] = filesData
	}

	bodyBytes, _ := json.Marshal(body)

	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, "", err
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("X-FE-Version", GetFeVersion())
	req.Header.Set("X-Signature", signature)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Origin", "https://chat.z.ai")
	req.Header.Set("Referer", fmt.Sprintf("https://chat.z.ai/c/%s", uuid.New().String()))
	req.Header.Set("User-Agent", uarand.GetRandom())

	LogInfo("[Upstream] Sending request: model=%s, target=%s", model, targetModel)

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   15 * time.Second,
			ResponseHeaderTimeout: 60 * time.Second,
		},
		// 不设置全局 Timeout，因为 SSE 流式响应需要长时间读取
	}
	resp, err := client.Do(req)
	if err != nil {
		LogError("[Upstream] Request error: %v", err)
		return nil, "", err
	}

	LogInfo("[Upstream] Response status: %d", resp.StatusCode)

	return resp, targetModel, nil
}

type UpstreamData struct {
	Type string `json:"type"`
	Data struct {
		DeltaContent string `json:"delta_content"`
		EditContent  string `json:"edit_content"`
		Phase        string `json:"phase"`
		Done         bool   `json:"done"`
	} `json:"data"`
}

func (u *UpstreamData) GetEditContent() string {
	editContent := u.Data.EditContent
	if editContent == "" {
		return ""
	}

	if len(editContent) > 0 && editContent[0] == '"' {
		var unescaped string
		if err := json.Unmarshal([]byte(editContent), &unescaped); err == nil {
			LogDebug("[GetEditContent] Unescaped edit_content from JSON string")
			return unescaped
		}
	}

	return editContent
}

type ThinkingFilter struct {
	hasSeenFirstThinking bool
	buffer               string
	lastOutputChunk      string
	lastPhase            string
	thinkingRoundCount   int
}

func (f *ThinkingFilter) ProcessThinking(deltaContent string) string {
	if !f.hasSeenFirstThinking {
		f.hasSeenFirstThinking = true
		if idx := strings.Index(deltaContent, "> "); idx != -1 {
			deltaContent = deltaContent[idx+2:]
		} else {
			return ""
		}
	}

	content := f.buffer + deltaContent
	f.buffer = ""

	content = strings.ReplaceAll(content, "\n> ", "\n")

	if strings.HasSuffix(content, "\n>") {
		f.buffer = "\n>"
		return content[:len(content)-2]
	}
	if strings.HasSuffix(content, "\n") {
		f.buffer = "\n"
		return content[:len(content)-1]
	}

	return content
}

func (f *ThinkingFilter) Flush() string {
	result := f.buffer
	f.buffer = ""
	return result
}

func (f *ThinkingFilter) ExtractCompleteThinking(editContent string) string {
	startIdx := strings.Index(editContent, "> ")
	if startIdx == -1 {
		return ""
	}
	startIdx += 2

	endIdx := strings.Index(editContent, "\n</details>")
	if endIdx == -1 {
		return ""
	}

	content := editContent[startIdx:endIdx]
	content = strings.ReplaceAll(content, "\n> ", "\n")
	return content
}

func (f *ThinkingFilter) ExtractIncrementalThinking(editContent string) string {
	completeThinking := f.ExtractCompleteThinking(editContent)
	if completeThinking == "" {
		return ""
	}

	if f.lastOutputChunk == "" {
		return completeThinking
	}

	idx := strings.Index(completeThinking, f.lastOutputChunk)
	if idx == -1 {
		return completeThinking
	}

	incrementalPart := completeThinking[idx+len(f.lastOutputChunk):]
	return incrementalPart
}

func (f *ThinkingFilter) ResetForNewRound() {
	f.lastOutputChunk = ""
	f.hasSeenFirstThinking = false
}

func HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if token == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	if token == "free" {
		anonymousToken, err := GetAnonymousToken()
		if err != nil {
			LogError("Failed to get anonymous token: %v", err)
			http.Error(w, "Failed to get anonymous token", http.StatusInternalServerError)
			return
		}
		token = anonymousToken
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		req.Model = "GLM-4.6"
	}

	isGLM5 := IsGLM5Model(req.Model)

	resp, modelName, err := makeUpstreamRequest(token, req.Messages, req.Model)
	if err != nil {
		LogError("Upstream request failed: %v", err)
		http.Error(w, "Upstream error", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)
		if len(bodyStr) > 500 {
			bodyStr = bodyStr[:500]
		}
		LogError("Upstream error: status=%d, body=%s", resp.StatusCode, bodyStr)
		http.Error(w, "Upstream error", resp.StatusCode)
		return
	}

	completionID := fmt.Sprintf("chatcmpl-%s", uuid.New().String()[:29])

	if req.Stream {
		handleStreamResponse(w, resp.Body, completionID, modelName, isGLM5)
	} else {
		handleNonStreamResponse(w, resp.Body, completionID, modelName, isGLM5)
	}
}

func handleStreamResponse(w http.ResponseWriter, body io.ReadCloser, completionID, modelName string, isGLM5 bool) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	hasContent := false
	searchRefFilter := NewSearchRefFilter()
	thinkingFilter := &ThinkingFilter{}
	pendingSourcesMarkdown := ""
	pendingImageSearchMarkdown := ""
	totalContentOutputLength := 0 // 记录已输出的 content 字符长度

	for scanner.Scan() {
		line := scanner.Text()
		LogDebug("[Upstream] %s", line)

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

		// 跳过 usage 统计数据（GLM-5 的 phase:"other" 带 usage 无内容）
		if upstream.Data.Phase == "other" && upstream.Data.DeltaContent == "" && upstream.GetEditContent() == "" {
			continue
		}

		if upstream.Data.Phase == "thinking" && upstream.Data.DeltaContent != "" {
			if isGLM5 {
				// GLM-5 thinking content is plain text, no blockquote wrapping
				reasoningContent := upstream.Data.DeltaContent
				reasoningContent = searchRefFilter.Process(reasoningContent)
				if reasoningContent != "" {
					hasContent = true
					chunk := ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: time.Now().Unix(),
						Model:   modelName,
						Choices: []Choice{{
							Index:        0,
							Delta:        Delta{ReasoningContent: reasoningContent},
							FinishReason: nil,
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
				continue
			}

			isNewThinkingRound := false
			if thinkingFilter.lastPhase != "" && thinkingFilter.lastPhase != "thinking" {
				thinkingFilter.ResetForNewRound()
				thinkingFilter.thinkingRoundCount++
				isNewThinkingRound = true
			}
			thinkingFilter.lastPhase = "thinking"

			reasoningContent := thinkingFilter.ProcessThinking(upstream.Data.DeltaContent)

			if isNewThinkingRound && thinkingFilter.thinkingRoundCount > 1 && reasoningContent != "" {
				reasoningContent = "\n\n" + reasoningContent
			}

			if reasoningContent != "" {
				thinkingFilter.lastOutputChunk = reasoningContent
				reasoningContent = searchRefFilter.Process(reasoningContent)

				if reasoningContent != "" {
					hasContent = true
					chunk := ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: time.Now().Unix(),
						Model:   modelName,
						Choices: []Choice{{
							Index:        0,
							Delta:        Delta{ReasoningContent: reasoningContent},
							FinishReason: nil,
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
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
				pendingSourcesMarkdown = searchRefFilter.GetSearchResultsMarkdown()
			}
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"search_image"`) {
			textBeforeBlock := ExtractTextBeforeGlmBlock(editContent)
			if textBeforeBlock != "" {
				textBeforeBlock = searchRefFilter.Process(textBeforeBlock)
				if textBeforeBlock != "" {
					hasContent = true
					chunk := ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: time.Now().Unix(),
						Model:   modelName,
						Choices: []Choice{{
							Index:        0,
							Delta:        Delta{Content: textBeforeBlock},
							FinishReason: nil,
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
			}
			if results := ParseImageSearchResults(editContent); len(results) > 0 {
				pendingImageSearchMarkdown = FormatImageSearchResults(results)
			}
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"mcp"`) {
			textBeforeBlock := ExtractTextBeforeGlmBlock(editContent)
			if textBeforeBlock != "" {
				textBeforeBlock = searchRefFilter.Process(textBeforeBlock)
				if textBeforeBlock != "" {
					hasContent = true
					chunk := ChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: time.Now().Unix(),
						Model:   modelName,
						Choices: []Choice{{
							Index:        0,
							Delta:        Delta{Content: textBeforeBlock},
							FinishReason: nil,
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
			}
			continue
		}
		if editContent != "" && IsSearchToolCall(editContent, upstream.Data.Phase) {
			continue
		}

		if pendingSourcesMarkdown != "" {
			hasContent = true
			chunk := ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []Choice{{
					Index:        0,
					Delta:        Delta{Content: pendingSourcesMarkdown},
					FinishReason: nil,
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			pendingSourcesMarkdown = ""
		}
		if pendingImageSearchMarkdown != "" {
			hasContent = true
			chunk := ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []Choice{{
					Index:        0,
					Delta:        Delta{Content: pendingImageSearchMarkdown},
					FinishReason: nil,
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			pendingImageSearchMarkdown = ""
		}

		content := ""
		reasoningContent := ""

		if thinkingRemaining := thinkingFilter.Flush(); thinkingRemaining != "" {
			thinkingFilter.lastOutputChunk = thinkingRemaining
			processedRemaining := searchRefFilter.Process(thinkingRemaining)
			if processedRemaining != "" {
				hasContent = true
				chunk := ChatCompletionChunk{
					ID:      completionID,
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   modelName,
					Choices: []Choice{{
						Index:        0,
						Delta:        Delta{ReasoningContent: processedRemaining},
						FinishReason: nil,
					}},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
		}

		if pendingSourcesMarkdown != "" && thinkingFilter.hasSeenFirstThinking {
			hasContent = true
			chunk := ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []Choice{{
					Index:        0,
					Delta:        Delta{ReasoningContent: pendingSourcesMarkdown},
					FinishReason: nil,
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			pendingSourcesMarkdown = ""
		}

		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			content = upstream.Data.DeltaContent
		} else if upstream.Data.Phase == "answer" && editContent != "" {
			if strings.Contains(editContent, "</details>") {
				reasoningContent = thinkingFilter.ExtractIncrementalThinking(editContent)

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

		if reasoningContent != "" {
			reasoningContent = searchRefFilter.Process(reasoningContent) + searchRefFilter.Flush()
		}
		if reasoningContent != "" {
			hasContent = true
			chunk := ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []Choice{{
					Index:        0,
					Delta:        Delta{ReasoningContent: reasoningContent},
					FinishReason: nil,
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		if content == "" {
			continue
		}

		content = searchRefFilter.Process(content)
		if content == "" {
			continue
		}

		hasContent = true
		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			totalContentOutputLength += len([]rune(content))
		}

		chunk := ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   modelName,
			Choices: []Choice{{
				Index:        0,
				Delta:        Delta{Content: content},
				FinishReason: nil,
			}},
		}

		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	if err := scanner.Err(); err != nil {
		LogError("[Upstream] scanner error: %v", err)
	}

	if remaining := searchRefFilter.Flush(); remaining != "" {
		hasContent = true
		chunk := ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   modelName,
			Choices: []Choice{{
				Index:        0,
				Delta:        Delta{Content: remaining},
				FinishReason: nil,
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	if !hasContent {
		LogError("Stream response 200 but no content received")
	}

	stopReason := "stop"
	finalChunk := ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []Choice{{
			Index:        0,
			Delta:        Delta{},
			FinishReason: &stopReason,
		}},
	}

	data, _ := json.Marshal(finalChunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleNonStreamResponse(w http.ResponseWriter, body io.ReadCloser, completionID, modelName string, isGLM5 bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	var chunks []string
	var reasoningChunks []string
	thinkingFilter := &ThinkingFilter{}
	searchRefFilter := NewSearchRefFilter()
	hasThinking := false
	pendingSourcesMarkdown := ""
	pendingImageSearchMarkdown := ""

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

		if upstream.Data.Phase == "thinking" && upstream.Data.DeltaContent != "" {
			if isGLM5 {
				// GLM-5 thinking content is plain text, no blockquote wrapping
				hasThinking = true
				reasoningChunks = append(reasoningChunks, upstream.Data.DeltaContent)
				continue
			}

			if thinkingFilter.lastPhase != "" && thinkingFilter.lastPhase != "thinking" {
				thinkingFilter.ResetForNewRound()
				thinkingFilter.thinkingRoundCount++
				if thinkingFilter.thinkingRoundCount > 1 {
					reasoningChunks = append(reasoningChunks, "\n\n")
				}
			}
			thinkingFilter.lastPhase = "thinking"

			hasThinking = true
			reasoningContent := thinkingFilter.ProcessThinking(upstream.Data.DeltaContent)
			if reasoningContent != "" {
				thinkingFilter.lastOutputChunk = reasoningContent
				reasoningChunks = append(reasoningChunks, reasoningContent)
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
				pendingSourcesMarkdown = searchRefFilter.GetSearchResultsMarkdown()
			}
			continue
		}
		if editContent != "" && strings.Contains(editContent, `"search_image"`) {
			textBeforeBlock := ExtractTextBeforeGlmBlock(editContent)
			if textBeforeBlock != "" {
				chunks = append(chunks, textBeforeBlock)
			}
			// 解析图片搜索结果
			if results := ParseImageSearchResults(editContent); len(results) > 0 {
				pendingImageSearchMarkdown = FormatImageSearchResults(results)
			}
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

		if pendingSourcesMarkdown != "" {
			if hasThinking {
				reasoningChunks = append(reasoningChunks, pendingSourcesMarkdown)
			} else {
				chunks = append(chunks, pendingSourcesMarkdown)
			}
			pendingSourcesMarkdown = ""
		}
		if pendingImageSearchMarkdown != "" {
			chunks = append(chunks, pendingImageSearchMarkdown)
			pendingImageSearchMarkdown = ""
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

	if fullContent == "" {
		LogError("Non-stream response 200 but no content received")
	}

	stopReason := "stop"
	response := ChatCompletionResponse{
		ID:      completionID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []Choice{{
			Index: 0,
			Message: &MessageResp{
				Role:             "assistant",
				Content:          fullContent,
				ReasoningContent: fullReasoning,
			},
			FinishReason: &stopReason,
		}},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func HandleModels(w http.ResponseWriter, r *http.Request) {
	var models []ModelInfo
	for _, id := range ModelList {
		models = append(models, ModelInfo{
			ID:      id,
			Object:  "model",
			OwnedBy: "z.ai",
		})
	}

	response := ModelsResponse{
		Object: "list",
		Data:   models,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
