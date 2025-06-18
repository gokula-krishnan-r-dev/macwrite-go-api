package middleware

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

// LoggerMiddleware creates a middleware for logging HTTP requests
func LoggerMiddleware() gin.HandlerFunc {
	// Set log output
	log.SetOutput(os.Stdout)

	return func(c *gin.Context) {
		// Start timer
		start := time.Now()

		// Get request ID from context
		requestID, exists := c.Get("RequestID")
		var rid string
		if exists {
			rid = requestID.(string)
		} else {
			rid = "no-request-id"
		}

		// Log request
		path := c.Request.URL.Path
		method := c.Request.Method
		ip := c.ClientIP()
		userAgent := c.Request.UserAgent()

		log.Printf("[%s] --> %s %s | IP: %s | UA: %s", rid, method, path, ip, userAgent)

		// Process request
		c.Next()

		// Calculate latency
		latency := time.Since(start)

		// Get status code
		status := c.Writer.Status()

		// Log status code and latency
		log.Printf("[%s] <-- %s %s | Status: %d | Latency: %s",
			rid, method, path, status, latency.String())

		// Add additional log for errors
		if status >= 400 {
			errorMsg := fmt.Sprintf("%s request to %s returned status code %d", method, path, status)
			log.Printf("[%s] ERROR: %s", rid, errorMsg)
		}
	}
}
