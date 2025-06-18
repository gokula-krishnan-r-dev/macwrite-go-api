package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds all configuration for the application
type Config struct {
	Server   ServerConfig
	AI       AIConfig
	LogLevel string
}

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Port            string
	ShutdownTimeout time.Duration
}

// AIConfig holds AI-specific configuration
type AIConfig struct {
	Endpoint     string
	Timeout      time.Duration
	DefaultModel string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Port:            getEnv("SERVER_PORT", "8080"),
			ShutdownTimeout: time.Duration(getEnvAsInt("SERVER_SHUTDOWN_TIMEOUT", 5)) * time.Second,
		},
		AI: AIConfig{
			Endpoint:     getEnv("AI_ENDPOINT", "https://auto-comment.gokulakrishnanr812-492.workers.dev/"),
			Timeout:      time.Duration(getEnvAsInt("AI_TIMEOUT", 30)) * time.Second,
			DefaultModel: getEnv("AI_DEFAULT_MODEL", "default"),
		},
		LogLevel: getEnv("LOG_LEVEL", "info"),
	}
}

// Helper function to get an environment variable or a default value
func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

// Helper function to get an environment variable as an integer
func getEnvAsInt(key string, defaultValue int) int {
	if value, exists := os.LookupEnv(key); exists {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
