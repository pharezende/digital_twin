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

resource "grafana_data_source" "prometheus" {
  type       = "prometheus"
  name       = "Prometheus"
  uid        = "digital-twin-prometheus"
  url        = var.prometheus_url
  is_default = true
}

variable "prometheus_url" {
  description = "Prometheus URL accessible from Grafana"
  type        = string
  default     = "http://prometheus:9090"
}