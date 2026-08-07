// =========================================
// Linux virtual machine that runs the Agent Learning SDK inside the VNet
// =========================================
// The VM lives in the `vm` subnet so it can reach Cosmos DB and Storage over
// their private endpoints. It carries a system-assigned managed identity that
// is granted Cosmos DB and Storage data-plane roles in main.bicep, so the SDK
// authenticates with DefaultAzureCredential (no secrets on the box).
//
// cloud-init installs Python + the SDK's runtime dependencies and writes the
// AGENT_LEARNING_* environment variables consumed by agent_learning.config.

@description('Name of the virtual machine.')
param name string

@description('Azure region for the VM and its network interface.')
param location string = resourceGroup().location

@description('Tags applied to all resources in this module.')
param tags object = {}

@description('Resource ID of the subnet that hosts the VM.')
param subnetId string

@description('VM size. Standard_D2s_v3 is broadly available; override for larger training workloads.')
param vmSize string = 'Standard_D2s_v3'

@description('Administrator username for the VM.')
param adminUsername string = 'azureuser'

@description('SSH public key (OpenSSH format) used for administrator login. Password auth is disabled.')
param sshPublicKey string

@description('Attach a public IP for direct SSH access. Leave false to keep the VM private (reach it via Bastion/jumpbox).')
param enablePublicIp bool = false

@description('Cosmos DB account endpoint injected into the SDK environment (AGENT_LEARNING_COSMOS_ENDPOINT).')
param cosmosEndpoint string

@description('Cosmos DB database name injected into the SDK environment (AGENT_LEARNING_COSMOS_DATABASE).')
param cosmosDatabaseName string

// cloud-init: multi-line strings don't interpolate in Bicep, so build via format().
// {0}=cosmosEndpoint  {1}=cosmosDatabaseName  {2}=adminUsername
var cloudInit = format('''#cloud-config
package_update: true
packages:
  - python3-pip
  - python3-venv
  - git
write_files:
  - path: /etc/profile.d/agent-learning.sh
    permissions: '0644'
    content: |
      export AGENT_LEARNING_STORE_BACKEND="cosmos"
      export AGENT_LEARNING_COSMOS_ENDPOINT="{0}"
      export AGENT_LEARNING_COSMOS_DATABASE="{1}"
      export AGENT_LEARNING_COSMOS_AUTH_MODE="aad"
runcmd:
  - pip3 install --upgrade pip
  - pip3 install azure-cosmos azure-identity numpy
  - su - {2} -c "git clone https://github.com/microsoft/agents-learning-sdk.git ~/agents-learning-sdk || true"
''', cosmosEndpoint, cosmosDatabaseName, adminUsername)

resource publicIp 'Microsoft.Network/publicIPAddresses@2023-05-01' = if (enablePublicIp) {
  name: '${name}-pip'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2023-05-01' = {
  name: '${name}-nic'
  location: location
  tags: tags
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: subnetId
          }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: enablePublicIp ? {
            id: publicIp!.id
          } : null
        }
      }
    ]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2023-09-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    osProfile: {
      computerName: take(name, 15)
      adminUsername: adminUsername
      customData: base64(cloudInit)
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
        }
      ]
    }
    diagnosticsProfile: {
      bootDiagnostics: {
        enabled: true
      }
    }
  }
}

output vmName string = vm.name
output principalId string = vm.identity.principalId
output privateIpAddress string = nic.properties.ipConfigurations[0].properties.privateIPAddress
output publicIpAddress string = enablePublicIp ? publicIp!.properties.ipAddress : ''
