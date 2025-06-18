package main

import (
	"context"
	"crypto/rand"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

var cfg *Config

func init() {
	// Configure the logger
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
}

func main() {
	// Load config
	var err error
	cfg, err = LoadConfig("config.json")
	if err != nil {
		log.Printf("Warning: Failed to load config file: %v", err)
		// Use default config
		cfg = &Config{
			Server: ServerConfig{
				Port:    "8080",
				Host:    "localhost",
				LogFile: "api.log",
			},
			AI: AIConfig{
				Endpoint:     "https://auto-comment.gokulakrishnanr812-492.workers.dev/",
				DefaultModel: "claude-3-haiku-20240307",
				Timeout:      30 * time.Second,
			},
		}
	}

	// Set Gin to release mode in production
	if os.Getenv("GO_ENV") == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create router with default middleware
	router := gin.New()

	//cors
	router.Use(cors.New(cors.Config{
		AllowAllOrigins:  true,
		AllowCredentials: true,
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Content-Type", "Authorization"},
	}))

	// Apply global middleware
	router.Use(gin.Recovery())
	router.Use(func(c *gin.Context) {
		// Generate a request ID
		requestID := generateRequestID()
		c.Set("RequestID", requestID)
		c.Header("X-Request-ID", requestID)

		// Store the start time
		startTime := time.Now()
		c.Set("startTime", startTime)

		// Log the request
		log.Printf("[%s] --> %s %s | IP: %s | UA: %s",
			requestID, c.Request.Method, c.Request.URL.Path,
			c.ClientIP(), c.Request.UserAgent())

		// Process the request
		c.Next()

		// Calculate latency
		latency := time.Since(startTime)

		// Log the response
		log.Printf("[%s] <-- %s %s | Status: %d | Size: %d | Latency: %v",
			requestID, c.Request.Method, c.Request.URL.Path,
			c.Writer.Status(), c.Writer.Size(), latency)

		// Log errors if any
		if c.Writer.Status() >= 400 {
			log.Printf("[%s] ERROR: %s request to %s returned status code %d",
				requestID, c.Request.Method, c.Request.URL.Path, c.Writer.Status())
		}
	})

	// Register routes
	registerRoutes(router)

	// Create HTTP server
	server := &http.Server{
		Addr:    ":" + cfg.Server.Port,
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("Server starting on port %s", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// Shutdown gracefully
	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited")
}

// registerRoutes sets up all the routes for our application
func registerRoutes(router *gin.Engine) {
	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// Register AI routes directly
	registerAIRoutes(router.Group(""))
}

// generateRequestID creates a unique ID for each request
func generateRequestID() string {
	// Generate a random UUID-like string
	return generateUUID()
}

// generateUUID generates a random UUID v4 string
func generateUUID() string {
	// Simple implementation for illustration
	b := make([]byte, 16)
	_, err := rand.Read(b)
	if err != nil {
		// Fallback to time-based ID if random fails
		return time.Now().Format("20060102150405.000000")
	}

	// Format as UUID
	return fmt.Sprintf("%x-%x-%x-%x-%x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}
