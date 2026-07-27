targetScope = 'subscription'

// =====================================================================
// Agent Learning SDK — focused infrastructure
// =====================================================================
// Provisions everything the SDK needs to run privately:
//   1. Virtual network (vm + private-endpoints subnets)
//   2. Cosmos DB (serverless, AAD-only, public access disabled) + the
//      learning database/containers + a private endpoint & DNS
//   3. A private Storage account (blob) + a private endpoint & DNS
//   4. A Linux VM in the vm subnet with a system-assigned managed
//      identity, granted Cosmos DB + Storage data-plane roles
//
// The VM reaches Cosmos DB and Storage over their private endpoints and
// authenticates with its managed identity — no keys or secrets on the box.
// =====================================================================

@minLength(1)
@maxLength(64)
@description('Name of the environment; used to derive a short unique hash for all resources.')
param environmentName string

@minLength(1)
@description('Primary Azure region for all resources.')
@metadata({
  azd: {
    type: 'location'
  }
})
param location string

@description('Optional explicit resource group name. Defaults to rg-<environmentName>.')
param resourceGroupName string = ''

// ----- Resource name overrides (optional) -----
param cosmosAccountName string = ''
param storageAccountName string = ''
param vNetName string = ''
param vmName string = ''

@description('Name of the Agent Learning Cosmos DB database.')
param cosmosDatabaseName string = 'dq_rl'

// ----- Virtual machine -----
@description('Size of the SDK virtual machine.')
param vmSize string = 'Standard_D2s_v3'

@description('Administrator username for the VM.')
param adminUsername string = 'azureuser'

@description('SSH public key (OpenSSH format) for the VM administrator. Required; password auth is disabled.')
param sshPublicKey string

@description('Attach a public IP to the VM for direct SSH. Leave false to keep the VM fully private.')
param enableVmPublicIp bool = false

@description('Optional source IP/CIDR permitted to SSH to the VM subnet. Empty = no inbound SSH rule.')
param allowedSshSourceAddressPrefix string = ''

// ----- Optional developer access -----
@description('Optional principal (object) ID of a developer to grant Cosmos DB data-plane access for local testing (reachable only from within the VNet).')
param developerPrincipalId string = ''

// ---------------------------------------------------------------------
// Derived names & tags
// ---------------------------------------------------------------------
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var tags = { 'azd-env-name': environmentName }

var cosmosName = !empty(cosmosAccountName) ? cosmosAccountName : '${abbrs.documentDBDatabaseAccounts}${resourceToken}'
var storageName = !empty(storageAccountName) ? storageAccountName : '${abbrs.storageStorageAccounts}${resourceToken}'
var vnetResourceName = !empty(vNetName) ? vNetName : '${abbrs.networkVirtualNetworks}${resourceToken}'
var virtualMachineName = !empty(vmName) ? vmName : '${abbrs.computeVirtualMachines}${resourceToken}'

// Built-in role definition IDs
var cosmosDataContributorRoleId = '00000000-0000-0000-0000-000000000002' // Cosmos DB Built-in Data Contributor (data plane)
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor

// ---------------------------------------------------------------------
// Resource group
// ---------------------------------------------------------------------
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

// ---------------------------------------------------------------------
// Virtual network
// ---------------------------------------------------------------------
module vnet 'app/vnet.bicep' = {
  name: 'vnet'
  scope: rg
  params: {
    location: location
    tags: tags
    vNetName: vnetResourceName
    allowedSshSourceAddressPrefix: allowedSshSourceAddressPrefix
  }
}

// ---------------------------------------------------------------------
// Cosmos DB account (private, AAD-only) + learning database/containers
// ---------------------------------------------------------------------
module cosmosAccount 'app/cosmos-account.bicep' = {
  name: 'cosmosAccount'
  scope: rg
  params: {
    name: cosmosName
    location: location
    tags: tags
    disableLocalAuth: true
    publicNetworkAccess: 'Disabled'
  }
}

module learningCosmos 'app/learning-cosmos.bicep' = {
  name: 'learningCosmos'
  scope: rg
  params: {
    parentAccountName: cosmosAccount.outputs.name
    databaseName: cosmosDatabaseName
    tags: tags
  }
}

module cosmosPrivateEndpoint 'app/cosmos-PrivateEndpoint.bicep' = {
  name: 'cosmosPrivateEndpoint'
  scope: rg
  params: {
    location: location
    tags: tags
    virtualNetworkName: vnet.outputs.vnetName
    subnetName: vnet.outputs.peSubnetName
    resourceName: cosmosAccount.outputs.name
  }
}

// ---------------------------------------------------------------------
// Private storage account (blob) + private endpoint
// ---------------------------------------------------------------------
module storage 'core/storage/storage-account.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    name: storageName
    location: location
    tags: tags
    publicNetworkAccess: 'Disabled'
    allowSharedKeyAccess: false
    allowBlobPublicAccess: false
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
    containers: [
      {
        name: 'agent-learning'
      }
    ]
  }
}

module storagePrivateEndpoint 'app/storage-PrivateEndpoint.bicep' = {
  name: 'storagePrivateEndpoint'
  scope: rg
  params: {
    location: location
    tags: tags
    virtualNetworkName: vnet.outputs.vnetName
    subnetName: vnet.outputs.peSubnetName
    resourceName: storage.outputs.name
  }
}

// ---------------------------------------------------------------------
// SDK virtual machine (system-assigned managed identity)
// ---------------------------------------------------------------------
module vm 'app/vm.bicep' = {
  name: 'vm'
  scope: rg
  params: {
    name: virtualMachineName
    location: location
    tags: tags
    subnetId: vnet.outputs.vmSubnetId
    vmSize: vmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
    enablePublicIp: enableVmPublicIp
    cosmosEndpoint: cosmosAccount.outputs.endpoint
    cosmosDatabaseName: cosmosDatabaseName
  }
}

// ---------------------------------------------------------------------
// Data-plane role assignments for the VM managed identity
// ---------------------------------------------------------------------
module vmCosmosRole 'app/cosmos-RoleAssignment.bicep' = {
  name: 'vmCosmosRole'
  scope: rg
  params: {
    cosmosAccountName: cosmosAccount.outputs.name
    roleDefinitionID: cosmosDataContributorRoleId
    principalID: vm.outputs.principalId
  }
}

module vmStorageRole 'app/storage-Access.bicep' = {
  name: 'vmStorageRole'
  scope: rg
  params: {
    storageAccountName: storage.outputs.name
    roleDefinitionID: storageBlobDataContributorRoleId
    principalID: vm.outputs.principalId
  }
}

// Optional: grant a developer Cosmos DB data-plane access (VNet-reachable only).
module developerCosmosRole 'app/cosmos-RoleAssignment.bicep' = if (!empty(developerPrincipalId)) {
  name: 'developerCosmosRole'
  scope: rg
  params: {
    cosmosAccountName: cosmosAccount.outputs.name
    roleDefinitionID: cosmosDataContributorRoleId
    principalID: developerPrincipalId
  }
}

// ---------------------------------------------------------------------
// Outputs (captured by azd as environment variables)
// ---------------------------------------------------------------------
output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = rg.name
output AZURE_TENANT_ID string = tenant().tenantId

// SDK configuration (agent_learning.config.CosmosConfig)
output AGENT_LEARNING_COSMOS_ENDPOINT string = cosmosAccount.outputs.endpoint
output AGENT_LEARNING_COSMOS_DATABASE string = learningCosmos.outputs.databaseName
output AGENT_LEARNING_COSMOS_AUTH_MODE string = 'aad'

output AZURE_COSMOS_ACCOUNT_NAME string = cosmosAccount.outputs.name
output AZURE_STORAGE_ACCOUNT_NAME string = storage.outputs.name
output AZURE_STORAGE_BLOB_ENDPOINT string = storage.outputs.primaryEndpoints.blob
output AZURE_VNET_NAME string = vnet.outputs.vnetName
output AZURE_VM_NAME string = vm.outputs.vmName
output AZURE_VM_PRINCIPAL_ID string = vm.outputs.principalId
output AZURE_VM_PRIVATE_IP string = vm.outputs.privateIpAddress
output AZURE_VM_PUBLIC_IP string = vm.outputs.publicIpAddress
