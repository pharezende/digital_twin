variable "grafana_url" {
  type    = string
  default = "http://localhost:3000"
}

variable "grafana_username" {
  type    = string
  default = "admin"
}

variable "grafana_password" {
  type      = string
  sensitive = true
}