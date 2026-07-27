// =========================================
// Private endpoint + private DNS for a Storage account (blob)
// =========================================
// Publishes the storage account's blob service into the VNet so the SDK VM
// resolves the account host to a private IP. Only the blob sub-resource is
// wired up because the Agent Learning SDK reads/writes blobs (judge snapshots,
// exported data) and does not use queues.

@description('Specifies the name of the virtual network.')
param virtualNetworkName string

@description('Specifies the name of the subnet that hosts the private endpoint.')
param subnetName string

@description('Specifies the resource name of the Storage account with an endpoint.')
param resourceName string

@description('Specifies the location.')
param location string = resourceGroup().location

param tags object = {}

// Virtual Network
resource vnet 'Microsoft.Network/virtualNetworks@2021-08-01' existing = {
  name: virtualNetworkName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-09-01' existing = {
  name: resourceName
}

var blobPrivateDNSZoneName = format('privatelink.blob.{0}', environment().suffixes.storage)
var blobPrivateDnsZoneVirtualNetworkLinkName = format('{0}-blob-link-{1}', resourceName, take(toLower(uniqueString(resourceName, virtualNetworkName)), 4))

// Private DNS Zone
resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: blobPrivateDNSZoneName
  location: 'global'
  tags: tags
  properties: {}
  dependsOn: [
    vnet
  ]
}

// Virtual Network Link
resource blobPrivateDnsZoneVirtualNetworkLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: blobPrivateDnsZone
  name: blobPrivateDnsZoneVirtualNetworkLinkName
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// Private Endpoint (blob)
resource blobPrivateEndpoint 'Microsoft.Network/privateEndpoints@2021-08-01' = {
  name: '${resourceName}-blob-private-endpoint'
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: 'blobPrivateLinkConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'blob'
          ]
        }
      }
    ]
    subnet: {
      id: '${vnet.id}/subnets/${subnetName}'
    }
  }
}

resource blobPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2022-01-01' = {
  parent: blobPrivateEndpoint
  name: 'blobPrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'storageBlobARecord'
        properties: {
          privateDnsZoneId: blobPrivateDnsZone.id
        }
      }
    ]
  }
}

output privateEndpointId string = blobPrivateEndpoint.id
output privateDnsZoneId string = blobPrivateDnsZone.id
