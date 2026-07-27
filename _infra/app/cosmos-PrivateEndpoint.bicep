// Parameters
@description('Specifies the name of the virtual network.')
param virtualNetworkName string

@description('Specifies the name of the subnet which contains the private endpoints.')
param subnetName string

@description('Specifies the resource name of the Cosmos DB account.')
param resourceName string

@description('Specifies the location.')
param location string = resourceGroup().location

param tags object = {}

// Virtual Network
resource vnet 'Microsoft.Network/virtualNetworks@2021-08-01' existing = {
  name: virtualNetworkName
}

resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: resourceName
}

var cosmosPrivateDNSZoneName = 'privatelink.documents.azure.com'
var cosmosPrivateDnsZoneVirtualNetworkLinkName = format('{0}-cosmos-link-{1}', resourceName, take(toLower(uniqueString(resourceName, virtualNetworkName)), 4))

// Private DNS Zone for Cosmos DB
resource cosmosPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: cosmosPrivateDNSZoneName
  location: 'global'
  tags: tags
  properties: {}
  dependsOn: [
    vnet
  ]
}

// Virtual Network Link
resource cosmosPrivateDnsZoneVirtualNetworkLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: cosmosPrivateDnsZone
  name: cosmosPrivateDnsZoneVirtualNetworkLinkName
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// Private Endpoint for Cosmos DB SQL API
resource cosmosPrivateEndpoint 'Microsoft.Network/privateEndpoints@2021-08-01' = {
  name: '${resourceName}-cosmos-private-endpoint'
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: 'cosmosPrivateLinkConnection'
        properties: {
          privateLinkServiceId: cosmosDbAccount.id
          groupIds: [
            'Sql'
          ]
        }
      }
    ]
    subnet: {
      id: '${vnet.id}/subnets/${subnetName}'
    }
  }
}

// Private DNS Zone Group
resource cosmosPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2021-08-01' = {
  parent: cosmosPrivateEndpoint
  name: 'cosmosPrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'config1'
        properties: {
          privateDnsZoneId: cosmosPrivateDnsZone.id
        }
      }
    ]
  }
}

output privateEndpointId string = cosmosPrivateEndpoint.id
output privateDnsZoneId string = cosmosPrivateDnsZone.id
