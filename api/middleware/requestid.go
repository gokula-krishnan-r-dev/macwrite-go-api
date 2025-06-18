package middleware

import (
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// RequestIDMiddleware adds a unique request ID to every request
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check if a request ID header is already set
		requestID := c.Request.Header.Get("X-Request-ID")

		// If not, generate a new UUID as the request ID
		if requestID == "" {
			requestID = uuid.New().String()
		}

		// Set the request ID in the headers
		c.Writer.Header().Set("X-Request-ID", requestID)
		c.Set("RequestID", requestID)

		c.Next()
	}
}
