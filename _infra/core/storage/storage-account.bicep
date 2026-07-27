param name string
param location string = resourceGroup().location
param tags object = {}

param allowBlobPublicAccess bool = false
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Enabled'
param containers array = []
param kind string = 'StorageV2'
param minimumTlsVersion string = 'TLS1_2'
param sku object = { name: 'Standard_LRS' }
param networkAcls object = {
  bypass: 'AzureServices'
  defaultAction: 'Allow'
}
@description('Controls whether Shared Key authorization is permitted. Leave disabled to enforce managed identity only access.')
param allowSharedKeyAccess bool = false

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: name
  location: location
  tags: tags
  kind: kind
  sku: sku
  properties: {
    minimumTlsVersion: minimumTlsVersion
    allowBlobPublicAccess: allowBlobPublicAccess
    publicNetworkAccess: publicNetworkAccess
    allowSharedKeyAccess: allowSharedKeyAccess
    networkAcls: networkAcls
  }

  resource blobServices 'blobServices' = if (!empty(containers)) {
    name: 'default'
    resource container 'containers' = [for container in containers: {
      name: container.name
      properties: {
        publicAccess: container.?publicAccess ?? 'None'
      }
    }]
  }
}

output name string = storage.name
output primaryEndpoints object = storage.properties.primaryEndpoints
