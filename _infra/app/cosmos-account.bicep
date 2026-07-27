// =========================================
// Azure Cosmos DB (NoSQL) account for the Agent Learning SDK
// =========================================
// Provisions a serverless Cosmos DB account locked down for private access:
// - publicNetworkAccess Disabled  -> all traffic must flow through a private endpoint
// - disableLocalAuth   true       -> shared keys are rejected; only Entra ID (AAD) data-plane auth
// The SDK connects with DefaultAzureCredential (AGENT_LEARNING_COSMOS_AUTH_MODE=aad).

@description('Globally unique Cosmos DB account name (lowercase, <=44 chars).')
param name string

@description('Azure region for the account.')
param location string = resourceGroup().location

@description('Tags applied to the account.')
param tags object = {}

@description('Reject shared-key auth so only Entra ID (AAD) data-plane access is permitted.')
param disableLocalAuth bool = true

@description('Public network access. Disabled forces all data-plane traffic through the private endpoint.')
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Disabled'

@description('Default consistency level for the account.')
@allowed(['Eventual', 'ConsistentPrefix', 'Session', 'BoundedStaleness', 'Strong'])
param defaultConsistencyLevel string = 'Session'

@description('Optional IPv4 addresses/CIDRs allowed through the firewall (only honoured when publicNetworkAccess is Enabled).')
param ipRules array = []

resource account 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: name
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    disableLocalAuth: disableLocalAuth
    publicNetworkAccess: publicNetworkAccess
    enableAutomaticFailover: false
    isVirtualNetworkFilterEnabled: false
    minimalTlsVersion: 'Tls12'
    networkAclBypass: 'AzureServices'
    consistencyPolicy: {
      defaultConsistencyLevel: defaultConsistencyLevel
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    capabilities: [
      {
        name: 'EnableServerless'
      }
    ]
    ipRules: [for ip in ipRules: {
      ipAddressOrRange: ip
    }]
  }
}

output id string = account.id
output name string = account.name
output endpoint string = account.properties.documentEndpoint
