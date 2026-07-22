provider "grafana" {
  url  = var.grafana_url
  auth = "${var.grafana_username}:${var.grafana_password}"
}