# GistCache on Azure: Azure Container Apps runs the container, images live in Azure
# Container Registry, the API key lives in Key Vault, logs go to Log Analytics, and
# GitHub Actions deploys through OIDC (workload identity federation). No passwords or
# client secrets exist anywhere: every service authenticates with a managed identity.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

data "azurerm_client_config" "me" {}

# ACR and Key Vault names are global across Azure, so add a random suffix.
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-gistcache"
  location = var.location
}

# ---------- Logs ----------
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "log-gistcache"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# ---------- Container registry ----------
resource "azurerm_container_registry" "acr" {
  name                = "acrgistcache${random_string.suffix.result}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Basic"
  admin_enabled       = false # no shared username/password; pulls and pushes use identities
}

# ---------- Secrets ----------
resource "azurerm_key_vault" "kv" {
  name                       = "kv-gistcache-${random_string.suffix.result}"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  tenant_id                  = data.azurerm_client_config.me.tenant_id
  sku_name                   = "standard"
  enable_rbac_authorization  = true # access via Azure RBAC roles, not legacy access policies
  soft_delete_retention_days = 7
}

# You (whoever runs Terraform) can write the secret by hand. The key is never in Terraform.
resource "azurerm_role_assignment" "me_kv_officer" {
  scope                = azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.me.object_id
}

# ---------- App identity (least privilege) ----------
# The running container's identity: may pull images and read secrets, nothing else.
resource "azurerm_user_assigned_identity" "app" {
  name                = "id-gistcache-app"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_role_assignment" "app_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_role_assignment" "app_kv_read" {
  scope                = azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

# ---------- Runtime ----------
resource "azurerm_container_app_environment" "env" {
  name                       = "cae-gistcache"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

resource "azurerm_container_app" "app" {
  name                         = "gistcache"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  revision_mode                = "Single" # new revision replaces the old one only once it is up

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  registry {
    server   = azurerm_container_registry.acr.login_server
    identity = azurerm_user_assigned_identity.app.id
  }

  # A reference to Key Vault, not a copy: rotate the key in Key Vault and the app picks it up.
  secret {
    name                = "llm-api-key"
    key_vault_secret_id = "${azurerm_key_vault.kv.vault_uri}secrets/llm-api-key"
    identity            = azurerm_user_assigned_identity.app.id
  }

  ingress {
    external_enabled = true
    target_port      = 8100
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }

    # Only your IP. The proxy spends real API credits, so it must not be open to the internet.
    ip_security_restriction {
      name             = "allow-me"
      action           = "Allow"
      ip_address_range = var.allowed_cidr
    }
  }

  template {
    # SQLite is a single file on local disk, so exactly one replica: two replicas
    # would each have their own cache. Scaling out means moving the cache to a shared store.
    min_replicas = var.min_replicas
    max_replicas = 1

    container {
      name   = "gistcache"
      image  = "${azurerm_container_registry.acr.login_server}/gistcache:${var.image_tag}"
      cpu    = 2
      memory = "4Gi" # PyTorch plus three verifier models

      env {
        name  = "GISTCACHE_CONFIG"
        value = var.config_path
      }

      # The config names which env var holds the key (GROQ_API_KEY or ANTHROPIC_API_KEY).
      env {
        name        = var.api_key_env
        secret_name = "llm-api-key"
      }

      # Loading the models takes up to a minute; don't send traffic or kill the
      # container until /healthz answers.
      startup_probe {
        transport               = "HTTP"
        port                    = 8100
        path                    = "/healthz"
        interval_seconds        = 10
        failure_count_threshold = 10
      }

      liveness_probe {
        transport               = "HTTP"
        port                    = 8100
        path                    = "/healthz"
        interval_seconds        = 30
        failure_count_threshold = 3
      }

      readiness_probe {
        transport               = "HTTP"
        port                    = 8100
        path                    = "/healthz"
        interval_seconds        = 10
        failure_count_threshold = 3
      }
    }
  }

  # After the first apply, GitHub Actions owns the image tag. Without this, every
  # `terraform apply` would roll the app back to var.image_tag.
  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }

  depends_on = [
    azurerm_role_assignment.app_acr_pull,
    azurerm_role_assignment.app_kv_read,
  ]
}

# ---------- GitHub Actions deploy identity (OIDC, no stored credentials) ----------
resource "azurerm_user_assigned_identity" "deployer" {
  name                = "id-gistcache-deployer"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

# Trust: only a workflow running on main in this repo can sign in as this identity.
resource "azurerm_federated_identity_credential" "github_main" {
  name                = "github-main"
  resource_group_name = azurerm_resource_group.rg.name
  parent_id           = azurerm_user_assigned_identity.deployer.id
  issuer              = "https://token.actions.githubusercontent.com"
  audience            = ["api://AzureADTokenExchange"]
  subject             = "repo:${var.github_repo}:ref:refs/heads/main"
}

resource "azurerm_role_assignment" "deployer_acr_push" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPush"
  principal_id         = azurerm_user_assigned_identity.deployer.principal_id
}

# Contributor on this resource group only, so it can roll out new revisions. In RBAC mode
# Contributor cannot read Key Vault secrets, so the deployer never sees the API key.
resource "azurerm_role_assignment" "deployer_rg" {
  scope                = azurerm_resource_group.rg.id
  role_definition_name = "Contributor"
  principal_id         = azurerm_user_assigned_identity.deployer.principal_id
}
