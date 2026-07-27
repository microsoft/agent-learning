// =========================================
// Virtual network for the Agent Learning SDK
// =========================================
// Two subnets:
//   - vm                : hosts the SDK virtual machine
//   - private-endpoints : hosts the Cosmos DB and Storage private endpoints
// The VM reaches Cosmos/Storage privately because both subnets share the VNet.

@description('Specifies the name of the virtual network.')
param vNetName string

@description('Specifies the location.')
param location string = resourceGroup().location

@description('Name of the subnet that hosts the SDK virtual machine.')
param vmSubnetName string = 'vm'

@description('Name of the subnet that hosts the private endpoints (Cosmos DB, Storage).')
param peSubnetName string = 'private-endpoints'

@description('Optional source IP/CIDR allowed to SSH to the VM subnet. Empty = no inbound SSH rule (private-only).')
param allowedSshSourceAddressPrefix string = ''

param tags object = {}

resource vmSubnetNsg 'Microsoft.Network/networkSecurityGroups@2023-05-01' = {
  name: 'nsg-${vmSubnetName}'
  location: location
  tags: tags
  properties: {
    securityRules: empty(allowedSshSourceAddressPrefix) ? [] : [
      {
        name: 'Allow-SSH-Inbound'
        properties: {
          priority: 1000
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '22'
          sourceAddressPrefix: allowedSshSourceAddressPrefix
          destinationAddressPrefix: 'VirtualNetwork'
        }
      }
    ]
  }
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: vNetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: vmSubnetName
        properties: {
          addressPrefix: '10.0.1.0/24'
          networkSecurityGroup: {
            id: vmSubnetNsg.id
          }
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
        }
      }
      {
        name: peSubnetName
        properties: {
          addressPrefix: '10.0.2.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
        }
      }
    ]
    enableDdosProtection: false
  }
}

output vnetId string = virtualNetwork.id
output vnetName string = virtualNetwork.name
output vmSubnetName string = virtualNetwork.properties.subnets[0].name
output vmSubnetId string = virtualNetwork.properties.subnets[0].id
output peSubnetName string = virtualNetwork.properties.subnets[1].name
output peSubnetId string = virtualNetwork.properties.subnets[1].id
