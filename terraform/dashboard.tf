resource "grafana_apps_dashboard_dashboard_v2" "digital_twin" {
  metadata {
    uid = "digital-twin-assistant"
  }

  spec {
    json = jsonencode(
      jsondecode(
        file("${path.module}/dashboards/dashboard.json")
      ).spec
    )
  }

  options {
    allow_ui_updates = true
    overwrite        = true
  }
}