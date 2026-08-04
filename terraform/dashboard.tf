resource "grafana_apps_dashboard_dashboard_v2" "digital_twin" {
  metadata {
    uid = "digital-twin-assistant"
  }

  spec {
    json = jsonencode(
      jsondecode(
        templatefile(
          "${path.module}/dashboards/dashboard.json.tftpl",
          {
            prometheus_uid = grafana_data_source.prometheus.uid
          }
        )
      ).spec
    )
  }

  options {
    allow_ui_updates = true
    overwrite        = true
  }

  depends_on = [
    grafana_data_source.prometheus
  ]
}