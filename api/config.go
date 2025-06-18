package main

import (
	"encoding/json"
	"os"
	"time"
)

// Config represents the application configuration
type Config struct {
	Server ServerConfig `json:"server"`
	AI     AIConfig     `json:"ai"`
}

// ServerConfig represents the HTTP server configuration
type ServerConfig struct {
	Port    string `json:"port"`
	Host    string `json:"host"`
	LogFile string `json:"log_file"`
}

// AIConfig represents the AI service configuration
type AIConfig struct {
	Endpoint     string        `json:"endpoint"`
	DefaultModel string        `json:"default_model"`
	Timeout      time.Duration `json:"timeout"`
}

// LoadConfig loads the configuration from a file
func LoadConfig(filename string) (*Config, error) {
	// Set default timeout value for unmarshaling
	defaultTimeout := 30 * time.Second

	// Create a temporary struct to unmarshal raw JSON values
	type TempConfig struct {
		Server ServerConfig `json:"server"`
		AI     struct {
			Endpoint     string `json:"endpoint"`
			DefaultModel string `json:"default_model"`
			Timeout      int    `json:"timeout"` // Timeout in seconds
		} `json:"ai"`
	}

	// Open the config file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode the JSON file
	var tempConfig TempConfig
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&tempConfig); err != nil {
		return nil, err
	}

	// Convert the timeout to time.Duration
	timeout := defaultTimeout
	if tempConfig.AI.Timeout > 0 {
		timeout = time.Duration(tempConfig.AI.Timeout) * time.Second
	}

	// Create and return the config
	return &Config{
		Server: tempConfig.Server,
		AI: AIConfig{
			Endpoint:     tempConfig.AI.Endpoint,
			DefaultModel: tempConfig.AI.DefaultModel,
			Timeout:      timeout,
		},
	}, nil
}
