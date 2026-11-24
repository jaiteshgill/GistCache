output "api_url" {
  value = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}

output "acr_name" {
  description = "Set as the ACR_NAME repo variable"
  value       = azurerm_container_registry.acr.name
}

output "key_vault_name" {
  value = azurerm_key_vault.kv.name
}

output "deployer_client_id" {
  description = "Set as the AZURE_CLIENT_ID repo variable"
  value       = azurerm_user_assigned_identity.deployer.client_id
}

output "tenant_id" {
  description = "Set as the AZURE_TENANT_ID repo variable"
  value       = data.azurerm_client_config.me.tenant_id
}
