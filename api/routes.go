package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/genai"
)

// AIRequest represents the incoming request structure with dynamic fields
type AIRequest struct {
	Text        string      `json:"text"`
	Prompt      string      `json:"prompt,omitempty"`
	Instruction string      `json:"instruction,omitempty"`
	Model       string      `json:"model,omitempty"`
	Temperature interface{} `json:"temperature,omitempty"` // Accept both string and float
	MaxTokens   interface{} `json:"max_tokens,omitempty"`  // Accept both string and int
	Stream      bool        `json:"stream,omitempty"`
	// Support for any additional parameters
	ExtraParams map[string]interface{} `json:"-"`
}

// CloudflareAIRequest represents the request structure specifically formatted for Cloudflare
type CloudflareAIRequest struct {
	Content      string      `json:"content"`                 // Using content instead of text for Cloudflare compatibility
	SystemPrompt string      `json:"system_prompt,omitempty"` // Combines prompt and instruction
	Model        string      `json:"model,omitempty"`
	Temperature  interface{} `json:"temperature,omitempty"`
	MaxTokens    interface{} `json:"max_tokens,omitempty"`
	Stream       bool        `json:"stream,omitempty"`
	// Dynamic fields captured from the original request
	ExtraParams map[string]interface{} `json:"-"`
}

// ErrorResponse represents a standardized error response
type ErrorResponse struct {
	Error string `json:"error"`
}

// StreamEvent represents a single SSE event
type StreamEvent struct {
	ID    string `json:"id,omitempty"`
	Event string `json:"event,omitempty"`
	Data  string `json:"data"`
}

// CloudflareError represents error information from Cloudflare
type CloudflareError struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Error   bool   `json:"error,omitempty"`
}

// registerAIRoutes registers all AI-related routes
func registerAIRoutes(router *gin.RouterGroup) {
	router.POST("/ai/generate", generateAIResponse)
	router.POST("/ai/stream", streamAIResponse)
	router.GET("/ai/debug", debugCloudflareConnection)
}

// UnmarshalJSON implements a custom JSON unmarshaler for AIRequest to capture dynamic fields
func (r *AIRequest) UnmarshalJSON(data []byte) error {
	// First unmarshal into a map to capture all fields
	var rawFields map[string]interface{}
	if err := json.Unmarshal(data, &rawFields); err != nil {
		return err
	}

	// Store extra fields
	r.ExtraParams = make(map[string]interface{})
	for k, v := range rawFields {
		switch k {
		case "text":
			if s, ok := v.(string); ok {
				r.Text = s
			}
		case "prompt":
			if s, ok := v.(string); ok {
				r.Prompt = s
			}
		case "instruction":
			if s, ok := v.(string); ok {
				r.Instruction = s
			}
		case "model":
			if s, ok := v.(string); ok {
				r.Model = s
			}
		case "temperature":
			r.Temperature = v
		case "max_tokens":
			r.MaxTokens = v
		case "stream":
			if b, ok := v.(bool); ok {
				r.Stream = b
			}
		default:
			// Capture all other fields
			r.ExtraParams[k] = v
		}
	}
	return nil
}

// processCloudflareError analyzes response body for Cloudflare-specific errors
func processCloudflareError(rid string, respBody []byte, statusCode int) (*ErrorResponse, bool) {
	// Check for worker exception (error 1101)
	if bytes.Contains(respBody, []byte("Error</span><span class=\"cf-error-code\">1101</span>")) ||
		bytes.Contains(respBody, []byte("Worker threw exception")) {
		log.Printf("[%s] Cloudflare Worker threw exception (Error 1101)", rid)
		return &ErrorResponse{
			Error: "The AI service worker encountered an exception. This often happens with malformed requests or unsupported parameter combinations.",
		}, true
	}

	// Check for other HTML error pages
	if bytes.Contains(respBody, []byte("<html>")) || bytes.Contains(respBody, []byte("<!DOCTYPE html>")) {
		log.Printf("[%s] Received HTML error from Cloudflare", rid)

		// Extract error details from HTML if possible
		htmlError := string(respBody)
		errorMessage := "AI service is currently unavailable. Please try again later."

		// Try to extract specific error code and message
		if strings.Contains(htmlError, "Error</span><span class=\"cf-error-code\">") {
			errorCodeStart := strings.Index(htmlError, "Error</span><span class=\"cf-error-code\">") + 39
			errorCodeEnd := strings.Index(htmlError[errorCodeStart:], "</span>")
			if errorCodeEnd > 0 {
				errorCode := htmlError[errorCodeStart : errorCodeStart+errorCodeEnd]
				log.Printf("[%s] Detected Cloudflare error code: %s", rid, errorCode)
				errorMessage = fmt.Sprintf("Cloudflare error %s: ", errorCode) + errorMessage
			}
		}

		// Try to extract a more specific error message from the HTML
		if strings.Contains(htmlError, "400 Bad Request") {
			errorMessage = "The AI service rejected the request format. Please check your input."
		} else if strings.Contains(htmlError, "403 Forbidden") {
			errorMessage = "Access to the AI service is forbidden. Please check your credentials."
		} else if strings.Contains(htmlError, "429 Too Many Requests") {
			errorMessage = "The AI service is rate limiting requests. Please try again later."
		} else if strings.Contains(htmlError, "500 Internal Server Error") {
			errorMessage = "The AI service encountered an internal error. Please try again later."
		} else if strings.Contains(htmlError, "Worker threw exception") {
			errorMessage = "The AI service worker threw an exception. This may be due to invalid parameters or request structure."
		} else if strings.Contains(htmlError, "502 Bad Gateway") {
			errorMessage = "The AI service is temporarily unavailable. Please try again later."
		} else if strings.Contains(htmlError, "504 Gateway Timeout") {
			errorMessage = "The AI service timed out. Please try with a simpler request."
		}

		return &ErrorResponse{Error: errorMessage}, true
	}

	// Try to parse Cloudflare JSON error
	var cloudflareError CloudflareError
	if err := json.Unmarshal(respBody, &cloudflareError); err == nil {
		if cloudflareError.Error || cloudflareError.Code > 0 || cloudflareError.Message != "" {
			log.Printf("[%s] Cloudflare API error: %+v", rid, cloudflareError)
			errorMessage := "AI service error"
			if cloudflareError.Message != "" {
				errorMessage = cloudflareError.Message
			}
			return &ErrorResponse{Error: errorMessage}, true
		}
	}

	// No recognized Cloudflare error format found
	return nil, false
}

// sanitizeCloudflareRequest removes potentially problematic values from the request
func sanitizeCloudflareRequest(req map[string]interface{}) map[string]interface{} {
	// Create a formatted request compatible with Cloudflare Workers AI API
	sanitized := make(map[string]interface{})

	// Extract user message from the request
	var userContent string
	if content, ok := req["content"]; ok && content != nil {
		userContent = content.(string)
	} else if text, ok := req["text"]; ok && text != nil {
		userContent = text.(string)
	}

	// Extract system prompt (instruction + prompt)
	var systemPrompt string
	if prompt, ok := req["system_prompt"]; ok && prompt != nil {
		systemPrompt = prompt.(string)
	} else if prompt, ok := req["prompt"]; ok && prompt != nil {
		systemPrompt = prompt.(string)
	}

	// Add instruction if present
	if instruction, ok := req["instruction"]; ok && instruction != nil {
		if systemPrompt != "" {
			systemPrompt = instruction.(string) + "\n\n" + systemPrompt
		} else {
			systemPrompt = instruction.(string)
		}
	}

	// Determine which model to use
	var modelName string
	if model, ok := req["model"]; ok && model != nil && model.(string) != "" {
		modelName = model.(string)
	} else if cfg.AI.DefaultModel != "" {
		modelName = cfg.AI.DefaultModel
	} else {
		// Fallback model
		modelName = "@cf/meta/llama-3-8b-instruct"
	}

	// Check if we're using a full model path (e.g., @cf/meta/...) or just a name
	// If it's just a name, we'll use the direct API format instead of the messages format
	if strings.HasPrefix(modelName, "@cf/") {
		// Format the messages in the correct structure for the model (for @cf models)
		messages := []map[string]interface{}{
			{
				"role":    "system",
				"content": systemPrompt,
			},
			{
				"role":    "user",
				"content": userContent,
			},
		}

		// Build the request for Cloudflare Workers AI API (@cf model format)
		sanitized["messages"] = messages
		sanitized["model"] = modelName
	} else {
		// For other models like Claude, use a simpler format
		sanitized["prompt"] = systemPrompt
		sanitized["text"] = userContent
		sanitized["model"] = modelName
	}

	// Include stream parameter
	if stream, ok := req["stream"]; ok {
		sanitized["stream"] = stream
	}

	// Handle temperature parameter - must be between 0 and 1
	if temp, ok := req["temperature"]; ok && temp != nil {
		switch val := temp.(type) {
		case float64:
			if val < 0 {
				sanitized["temperature"] = 0.0
			} else if val > 1 {
				sanitized["temperature"] = 1.0
			} else {
				sanitized["temperature"] = val
			}
		case string:
			if tempFloat, err := strconv.ParseFloat(val, 64); err == nil {
				if tempFloat < 0 {
					sanitized["temperature"] = 0.0
				} else if tempFloat > 1 {
					sanitized["temperature"] = 1.0
				} else {
					sanitized["temperature"] = tempFloat
				}
			}
		}
	}

	// Add max_tokens parameter if valid
	if maxTokens, ok := req["max_tokens"]; ok && maxTokens != nil {
		switch val := maxTokens.(type) {
		case int:
			if val > 0 && val <= 4096 {
				sanitized["max_tokens"] = val
			}
		case float64:
			intVal := int(val)
			if intVal > 0 && intVal <= 4096 {
				sanitized["max_tokens"] = intVal
			}
		case string:
			if intVal, err := strconv.Atoi(val); err == nil && intVal > 0 && intVal <= 4096 {
				sanitized["max_tokens"] = intVal
			}
		}
	}

	// Add any additional parameters that might be supported
	for k, v := range req {
		if k != "content" && k != "text" && k != "system_prompt" && k != "prompt" &&
			k != "instruction" && k != "model" && k != "temperature" &&
			k != "max_tokens" && k != "stream" && k != "messages" {
			sanitized[k] = v
		}
	}

	// Add useful debugging info
	log.Printf("Sanitized request: %+v", sanitized)

	return sanitized
}

// streamAIResponse handles streaming AI responses as server-sent events
func streamAIResponse(c *gin.Context) {
	requestID, _ := c.Get("RequestID")
	rid := requestID.(string)

	// Log the start of streaming request
	log.Printf("[%s] Processing streaming AI request from %s", rid, c.ClientIP())

	// Read request body
	rawBody, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to read request body: %v", rid, err)
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Failed to read request body",
		})
		return
	}

	// Parse the request
	var request AIRequest
	if err := json.Unmarshal(rawBody, &request); err != nil {
		log.Printf("[%s] ERROR: Invalid request format: %v", rid, err)
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Invalid request format",
		})
		return
	}

	// Validate text field
	if request.Text == "" {
		log.Printf("[%s] ERROR: Text field is required", rid)
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Text field is required",
		})
		return
	}

	// Log the received request
	requestJSON, _ := json.Marshal(request)
	log.Printf("[%s] Streaming request body: %s", rid, string(requestJSON))

	// Prepare system prompt by combining instruction and prompt if available
	systemPrompt := request.Prompt
	if request.Instruction != "" {
		if systemPrompt != "" {
			systemPrompt = request.Instruction + "\n\n" + systemPrompt
		} else {
			systemPrompt = request.Instruction
		}
	}

	// Initialize request with dynamic parameters
	cloudflareRequest := map[string]interface{}{
		"text":   request.Text,
		"prompt": systemPrompt,
		"model":  request.Model,
	}

	// Add temperature if provided
	if request.Temperature != nil {
		cloudflareRequest["temperature"] = request.Temperature
	}

	// Add max_tokens if provided
	if request.MaxTokens != nil {
		cloudflareRequest["max_tokens"] = request.MaxTokens
	}

	// Add any additional parameters from the request
	for k, v := range request.ExtraParams {
		// Skip parameters we've already handled
		if k != "text" && k != "prompt" && k != "instruction" &&
			k != "model" && k != "temperature" && k != "max_tokens" && k != "stream" {
			cloudflareRequest[k] = v
		}
	}

	// Sanitize the request to properly format it for Cloudflare's API
	cloudflareRequest = sanitizeCloudflareRequest(cloudflareRequest)

	// Create a client with longer timeout for streaming
	client := &http.Client{
		Timeout: 2 * cfg.AI.Timeout, // Double timeout for streaming
	}

	// Properly encode the request body as JSON
	jsonData, err := json.Marshal(cloudflareRequest)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to marshal JSON request: %v", rid, err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: "Failed to prepare request",
		})
		return
	}

	// Log the formatted Cloudflare request
	log.Printf("[%s] Cloudflare streaming request: %s", rid, string(jsonData))

	requestBody := bytes.NewBuffer(jsonData)

	// Log outgoing request to AI service
	log.Printf("[%s] Sending streaming request to AI service: %s", rid, cfg.AI.Endpoint)

	// Create the request with additional headers for Cloudflare
	req, err := http.NewRequest("POST", cfg.AI.Endpoint, requestBody)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to create request to AI service: %v", rid, err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: "Failed to create request to AI service",
		})
		return
	}

	// Set headers - ensure they're consistent with what Cloudflare Workers expect
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Connection", "keep-alive")

	// Add API key if configured
	if apiKey := os.Getenv("CLOUDFLARE_API_KEY"); apiKey != "" {
		log.Printf("[%s] Using API key from environment for authentication", rid)
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	// Execute the streaming request
	log.Printf("[%s] Executing streaming request to AI service...", rid)
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to communicate with AI service: %v", rid, err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: "Failed to communicate with AI service",
		})
		return
	}

	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()

		// Read error response
		respBody, _ := io.ReadAll(resp.Body)
		log.Printf("[%s] Streaming request failed with status %d: %s", rid, resp.StatusCode, string(respBody))

		// Process Cloudflare error response
		if errorResp, found := processCloudflareError(rid, respBody, resp.StatusCode); found {
			c.JSON(http.StatusInternalServerError, errorResp)
			return
		}

		// Return appropriate error
		if resp.StatusCode >= 500 {
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error: "AI service encountered an error",
			})
		} else {
			c.JSON(resp.StatusCode, ErrorResponse{
				Error: "AI service rejected the request",
			})
		}
		return
	}

	// Set up SSE response headers
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("Transfer-Encoding", "chunked")
	c.Writer.Header().Set("X-Accel-Buffering", "no") // Disable buffering for Nginx

	// Ensure the headers are written to the client
	c.Writer.WriteHeader(http.StatusOK)

	// Flush headers immediately
	c.Writer.Flush()

	// Create a done channel to track connection state
	done := make(chan bool)

	// Handle client disconnect
	go func() {
		<-c.Request.Context().Done()
		log.Printf("[%s] Client disconnected from streaming response", rid)
		done <- true
	}()

	// Process the streaming response
	go func() {
		defer resp.Body.Close()
		defer close(done)

		log.Printf("[%s] Starting to stream response", rid)

		reader := bufio.NewReader(resp.Body)
		eventID := 0

		for {
			select {
			case <-done:
				log.Printf("[%s] Stopping stream due to client disconnect", rid)
				return
			default:
				// Read a line from the response
				line, err := reader.ReadString('\n')
				if err != nil {
					if err == io.EOF {
						log.Printf("[%s] Stream completed normally", rid)
					} else {
						log.Printf("[%s] Error reading from stream: %v", rid, err)
					}
					return
				}

				// Skip empty lines
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}

				// Only log short chunks to avoid overwhelming the logs
				if len(line) <= 100 {
					log.Printf("[%s] Received chunk: %s", rid, line)
				} else {
					log.Printf("[%s] Received chunk of length: %d", rid, len(line))
				}

				// Check for data prefix
				if strings.HasPrefix(line, "data:") {
					data := strings.TrimPrefix(line, "data:")
					data = strings.TrimSpace(data)

					// Handle different streaming formats
					// Some models might return JSON objects in the data field

					// Try to parse as JSON to handle special cases
					var parsedData interface{}
					if err := json.Unmarshal([]byte(data), &parsedData); err == nil {
						// Check if we need to extract from a specific field
						if dataObj, ok := parsedData.(map[string]interface{}); ok {
							// Look for common fields in streaming responses
							if text, ok := dataObj["text"].(string); ok {
								data = text // OpenAI text field
							} else if content, ok := dataObj["content"].(string); ok {
								data = content // Content field
							} else if message, ok := dataObj["message"].(map[string]interface{}); ok {
								if content, ok := message["content"].(string); ok {
									data = content // Nested content in message
								}
							} else if resp, ok := dataObj["response"].(string); ok {
								data = resp // Nested response field
							}
						}
					}

					// Send the event
					eventID++
					// Wrap event data in a response object to match non-streaming API
					responseObj := gin.H{"response": data}
					responseJSON, _ := json.Marshal(responseObj)
					event := fmt.Sprintf("id: %d\ndata: %s\n\n", eventID, string(responseJSON))
					_, err := fmt.Fprint(c.Writer, event)
					if err != nil {
						log.Printf("[%s] Error writing to client: %v", rid, err)
						return
					}
					c.Writer.Flush()
				}
			}
		}
	}()

	// Wait for streaming to complete
	<-done
	log.Printf("[%s] Streaming response completed", rid)
}

// debugCloudflareConnection tests the connection to the Cloudflare service
func debugCloudflareConnection(c *gin.Context) {
	requestID, _ := c.Get("RequestID")
	rid := requestID.(string)

	log.Printf("[%s] Testing connection to Cloudflare endpoint: %s", rid, cfg.AI.Endpoint)

	// Make a simple GET request to check connection
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", cfg.AI.Endpoint, nil)

	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: fmt.Sprintf("Failed to create test request: %v", err),
		})
		return
	}

	// Add basic headers
	req.Header.Set("User-Agent", "GoCustomAIProxy/1.0")

	// Execute the request
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: fmt.Sprintf("Connection test failed: %v", err),
		})
		return
	}
	defer resp.Body.Close()

	// Read response
	respBody, _ := io.ReadAll(resp.Body)

	// Return debug information
	c.JSON(http.StatusOK, gin.H{
		"endpoint":       cfg.AI.Endpoint,
		"status_code":    resp.StatusCode,
		"response_body":  string(respBody),
		"headers":        resp.Header,
		"content_length": resp.ContentLength,
	})
}

// generateAIResponse handles the AI request and forwards it to the actual AI service
func generateAIResponse(c *gin.Context) {
	requestID, _ := c.Get("RequestID")
	rid := requestID.(string)

	// Log the start of request processing
	log.Printf("[%s] Processing AI request from %s", rid, c.ClientIP())

	// Read request body
	rawBody, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.Printf("[%s] ERROR: Failed to read request body: %v", rid, err)
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Failed to read request body",
		})
		return
	}

	// Parse the request
	var request AIRequest
	if err := json.Unmarshal(rawBody, &request); err != nil {
		log.Printf("[%s] ERROR: Invalid request format: %v", rid, err)
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Invalid request format",
		})
		return
	}

	// Validate text field
	if request.Text == "" {
		log.Printf("[%s] ERROR: Text field is required", rid)
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Text field is required",
		})
		return
	}

	// Check if streaming is requested
	if request.Stream {
		// Redirect to streaming handler
		streamAIResponse(c)
		return
	}

	// Log the received request
	requestJSON, _ := json.Marshal(request)
	log.Printf("[%s] Request body: %s", rid, string(requestJSON))

	// Prepare system prompt by combining instruction and prompt if available
	systemPrompt := request.Prompt
	if request.Instruction != "" {
		if systemPrompt != "" {
			systemPrompt = request.Instruction + "\n\n" + systemPrompt
		} else {
			systemPrompt = request.Instruction
		}
	}

	// List of models to try in order if the specified model fails
	fallbackModels := []string{
		"@cf/meta/llama-3.3-70b-instruct-fp8-fast",
	}

	// If a model was specified, use it as the primary choice
	var modelsToTry []string
	if request.Model != "" {
		modelsToTry = append([]string{request.Model}, fallbackModels...)
	} else if cfg.AI.DefaultModel != "" {
		modelsToTry = append([]string{cfg.AI.DefaultModel}, fallbackModels...)
	} else {
		modelsToTry = fallbackModels
	}

	// GeminiModel represents available Gemini AI models
	type GeminiModel string

	const (
		GeminiProVision GeminiModel = "gemini-pro-vision"
		GeminiPro       GeminiModel = "gemini-pro"
		GeminiFlash     GeminiModel = "gemini-2.0-flash"
		GeminiPro15     GeminiModel = "gemini-2.0-pro"
		GeminiUltra     GeminiModel = "gemini-2.0-ultra"
	)

	// Check if requested model is a Gemini model
	isGeminiModel := false
	for _, model := range modelsToTry {
		switch GeminiModel(model) {
		case GeminiProVision, GeminiPro, GeminiFlash, GeminiPro15, GeminiUltra:
			isGeminiModel = true
			break
		}
		if isGeminiModel {
			break
		}
	}

	// Handle Gemini API request if applicable
	// This provides direct integration with Google's Gemini API
	// Set GEMINI_API_KEY environment variable to use this feature
	// Learn more at: https://ai.google.dev/gemini-api/docs/quickstart
	if isGeminiModel && modelsToTry[0] != "" {
		log.Printf("[%s] Using Gemini API for model: %s", rid, modelsToTry[0])

		// Get API key from environment
		apiKey := "AIzaSyDEOpIpJOPOnbTUP61BI9s_kyFPBHnUgow"

		// Create context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), cfg.AI.Timeout)
		defer cancel()

		// Initialize Gemini client
		client, err := genai.NewClient(ctx, &genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		})
		if err != nil {
			log.Printf("[%s] ERROR: Failed to initialize Gemini client: %v", rid, err)
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error: "Failed to initialize Gemini API client",
			})
			return
		}

		// Create system prompt and user content
		var contents []*genai.Content
		if systemPrompt != "" {
			// Combine system prompt with user text for better results
			// Since Gemini doesn't have a dedicated system role, we'll format it explicitly
			combinedText := fmt.Sprintf("System Instructions: %s\n\nUser Query: %s",
				systemPrompt, request.Text)

			contents = []*genai.Content{
				{
					Parts: []*genai.Part{{Text: combinedText}},
					Role:  "user",
				},
			}
		} else {
			// Just user text
			contents = []*genai.Content{
				{
					Parts: []*genai.Part{{Text: request.Text}},
					Role:  "user",
				},
			}
		}

		// Set generation config
		config := &genai.GenerateContentConfig{}

		// Set temperature if provided
		if request.Temperature != nil {
			switch temp := request.Temperature.(type) {
			case float64:
				tempFloat32 := float32(temp)
				config.Temperature = &tempFloat32
			case string:
				if tempFloat, err := strconv.ParseFloat(temp, 32); err == nil {
					tempFloat32 := float32(tempFloat)
					config.Temperature = &tempFloat32
				}
			}
		}

		// Set max tokens if provided
		if request.MaxTokens != nil {
			switch tokens := request.MaxTokens.(type) {
			case int:
				config.MaxOutputTokens = int32(tokens)
			case float64:
				config.MaxOutputTokens = int32(tokens)
			case string:
				if tokensInt, err := strconv.Atoi(tokens); err == nil {
					config.MaxOutputTokens = int32(tokensInt)
				}
			}
		}

		// Generate content
		result, err := client.Models.GenerateContent(
			ctx,
			modelsToTry[0],
			contents,
			config,
		)

		if err != nil {
			log.Printf("[%s] ERROR: Gemini API request failed: %v", rid, err)

			// Extract more detailed error information if available
			errorMessage := fmt.Sprintf("Gemini API request failed: %v", err)

			// Check for specific error types
			if strings.Contains(err.Error(), "INVALID_ARGUMENT") {
				errorMessage = "Invalid request to Gemini API. Please check model name and parameters."
			} else if strings.Contains(err.Error(), "PERMISSION_DENIED") {
				errorMessage = "Permission denied by Gemini API. Please check your API key."
			} else if strings.Contains(err.Error(), "RESOURCE_EXHAUSTED") {
				errorMessage = "Gemini API quota exceeded or rate limit reached. Please try again later."
			} else if strings.Contains(err.Error(), "UNAVAILABLE") {
				errorMessage = "Gemini API service is currently unavailable. Please try again later."
			}

			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error: errorMessage,
			})
			return
		}

		// Extract and return response
		if result != nil {
			response := result.Text()
			log.Printf("[%s] Successfully completed request using Gemini model %s", rid, modelsToTry[0])
			c.JSON(http.StatusOK, gin.H{"response": response})
			return
		} else {
			log.Printf("[%s] ERROR: Empty response from Gemini API", rid)
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error: "Empty response from Gemini API",
			})
			return
		}
	}

	// Try each model until one works
	var finalResponse *http.Response
	var responseBody []byte
	var successModel string

	for _, modelName := range modelsToTry {
		// Initialize request with dynamic parameters
		cloudflareRequest := map[string]interface{}{
			"text":        request.Text,
			"prompt":      systemPrompt,
			"model":       modelName,
			"instruction": request.Instruction,
		}

		// Add temperature if provided
		if request.Temperature != nil {
			cloudflareRequest["temperature"] = request.Temperature
		}

		// Add max_tokens if provided
		if request.MaxTokens != nil {
			cloudflareRequest["max_tokens"] = request.MaxTokens
		}
		//here check if condtion for gemini models
		// Add any additional parameters from the request
		for k, v := range request.ExtraParams {
			// Skip parameters we've already handled
			if k != "text" && k != "prompt" && k != "instruction" &&
				k != "model" && k != "temperature" && k != "max_tokens" && k != "stream" {
				cloudflareRequest[k] = v
			}
		}

		// Sanitize the request to properly format it for Cloudflare's API
		// cloudflareRequest = sanitizeCloudflareRequest(cloudflareRequest)

		// Create a client with timeout
		client := &http.Client{
			Timeout: cfg.AI.Timeout,
		}

		// Properly encode the request body as JSON
		jsonData, err := json.Marshal(cloudflareRequest)
		if err != nil {
			log.Printf("[%s] ERROR: Failed to marshal JSON request: %v", rid, err)
			continue // Try next model
		}

		// Log the formatted Cloudflare request
		log.Printf("[%s] Trying model %s with request: %s", rid, modelName, string(jsonData))

		requestBody := bytes.NewBuffer(jsonData)

		// Create the request with additional headers for Cloudflare
		req, err := http.NewRequest("POST", cfg.AI.Endpoint, requestBody)
		if err != nil {
			log.Printf("[%s] ERROR: Failed to create request to AI service: %v", rid, err)
			continue // Try next model
		}

		// Set headers - ensure they're consistent with what Cloudflare Workers expect
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")

		// Execute the request with retry logic
		var resp *http.Response
		maxRetries := 2
		retryDelay := 1 * time.Second

		for attempt := 1; attempt <= maxRetries; attempt++ {
			log.Printf("[%s] Executing request with model %s (attempt %d/%d)...", rid, modelName, attempt, maxRetries)
			resp, err = client.Do(req)

			if err == nil && resp.StatusCode == http.StatusOK {
				// Success
				respBody, err := io.ReadAll(resp.Body)
				if err == nil {
					// Found a working model and response
					finalResponse = resp
					responseBody = respBody
					successModel = modelName
					log.Printf("[%s] Successfully got response with model %s", rid, modelName)
					break
				}
				resp.Body.Close()
			}

			if err != nil {
				log.Printf("[%s] Attempt %d with model %s failed: %v", rid, attempt, modelName, err)
			} else {
				log.Printf("[%s] Attempt %d with model %s failed with status code: %d", rid, attempt, modelName, resp.StatusCode)
				resp.Body.Close()
			}

			if attempt < maxRetries {
				time.Sleep(retryDelay)
				retryDelay *= 2 // Exponential backoff
			}
		}

		// If we got a successful response, stop trying models
		if finalResponse != nil {
			break
		}
	}

	// Check if all models failed
	if finalResponse == nil {
		log.Printf("[%s] ERROR: Failed to get a valid response from any model", rid)
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: "Failed to communicate with AI service. All models failed.",
		})
		return
	}

	// Log response info
	log.Printf("[%s] AI service response with model %s, status: %d", rid, successModel, finalResponse.StatusCode)
	log.Printf("[%s] AI service response headers: %v", rid, finalResponse.Header)

	// Log the raw response for debugging
	log.Printf("[%s] Raw response: %s", rid, string(responseBody))

	// Check for Cloudflare errors
	if finalResponse.StatusCode >= 400 {
		// Process Cloudflare error response
		if errorResp, found := processCloudflareError(rid, responseBody, finalResponse.StatusCode); found {
			// Use appropriate status code based on the error
			errorStatusCode := http.StatusInternalServerError
			if finalResponse.StatusCode >= 400 && finalResponse.StatusCode < 500 {
				errorStatusCode = finalResponse.StatusCode
			}
			c.JSON(errorStatusCode, errorResp)
			return
		}
	}

	// Try to check for valid JSON response
	if !json.Valid(responseBody) && finalResponse.StatusCode == http.StatusOK {
		log.Printf("[%s] WARNING: Received non-JSON response with OK status", rid)
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: "AI service returned an invalid response format",
		})
		return
	}

	// Try different response formats that Cloudflare might return
	// Since different models have different response structures

	// Parse the response to extract the generated text
	var parsedResponse map[string]interface{}
	if err := json.Unmarshal(responseBody, &parsedResponse); err != nil {
		log.Printf("[%s] ERROR: Failed to parse AI response JSON: %v", rid, err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error: "Invalid response format from AI service",
		})
		return
	}

	// Format the response for our API client
	var result string

	// Extract result based on different response formats various models might return

	// Format 1: {response: {content: "text"}} (Claude format)
	if response, ok := parsedResponse["response"].(map[string]interface{}); ok {
		if content, ok := response["content"].(string); ok {
			result = content
			c.JSON(http.StatusOK, gin.H{"response": result})
			log.Printf("[%s] Successfully completed request using model %s (Claude format)", rid, successModel)
			return
		}
	}

	// Format 2: {result: {response: "text"}} (Some Mistral models)
	if resultObj, ok := parsedResponse["result"].(map[string]interface{}); ok {
		if response, ok := resultObj["response"].(string); ok {
			c.JSON(http.StatusOK, gin.H{"response": response})
			log.Printf("[%s] Successfully completed request using model %s (Mistral format)", rid, successModel)
			return
		}
	}

	// Format 3: Direct response in content field
	if content, ok := parsedResponse["content"].(string); ok {
		c.JSON(http.StatusOK, gin.H{"response": content})
		log.Printf("[%s] Successfully completed request using model %s (direct content format)", rid, successModel)
		return
	}

	// Format 4: Direct response in response field
	if response, ok := parsedResponse["response"].(string); ok {
		c.JSON(http.StatusOK, gin.H{"response": response})
		log.Printf("[%s] Successfully completed request using model %s (direct response format)", rid, successModel)
		return
	}

	// Format 5: Choices array (OpenAI format)
	if choices, ok := parsedResponse["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					c.JSON(http.StatusOK, gin.H{"response": content})
					log.Printf("[%s] Successfully completed request using model %s (OpenAI format)", rid, successModel)
					return
				}
			} else if text, ok := choice["text"].(string); ok {
				// Completion style response
				c.JSON(http.StatusOK, gin.H{"response": text})
				log.Printf("[%s] Successfully completed request using model %s (OpenAI completion format)", rid, successModel)
				return
			}
		}
	}

	// Format 6: @cf models specific format
	if result, ok := parsedResponse["result"].(string); ok {
		c.JSON(http.StatusOK, gin.H{"response": result})
		log.Printf("[%s] Successfully completed request using model %s (Cloudflare result format)", rid, successModel)
		return
	}

	// If we can't extract properly formatted response, return the raw response with a warning
	log.Printf("[%s] WARNING: Could not extract result in a known format, returning raw response", rid)
	c.Data(finalResponse.StatusCode, "application/json", responseBody)
	log.Printf("[%s] Successfully completed request using model %s (raw passthrough)", rid, successModel)
}
