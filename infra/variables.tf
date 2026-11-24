variable "subscription_id" {
  description = "Azure subscription ID (az account show --query id -o tsv)"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "canadacentral"
}

variable "allowed_cidr" {
  description = "Who may reach the API, e.g. your IP as 203.0.113.7/32. Never 0.0.0.0/0: the proxy spends your API credits."
  type        = string
}

variable "image_tag" {
  description = "Image tag for the first deploy only; after that GitHub Actions sets the image"
  type        = string
  default     = "initial"
}

variable "min_replicas" {
  description = "1 keeps it warm (about US$5/day). 0 scales to zero when idle: near-free, but the first request waits ~1 min for models to load."
  type        = number
  default     = 1
}

variable "github_repo" {
  description = "owner/name of the repo allowed to deploy"
  type        = string
  default     = "jaiteshgill/GistCache"
}

variable "config_path" {
  description = "Provider config inside the image"
  type        = string
  default     = "configs/anthropic.yaml"
}

variable "api_key_env" {
  description = "Env var the config reads the key from: GROQ_API_KEY for groq.yaml, ANTHROPIC_API_KEY for anthropic.yaml"
  type        = string
  default     = "ANTHROPIC_API_KEY"
}
