param principalID string
param roleDefinitionID string
param storageAccountName string

@description('Optional suffix to make role assignment name unique across deployments')
param nameSuffix string = ''

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-09-01' existing = {
  name: storageAccountName
}

// Allow access from API to storage account using a managed identity and least priv Storage roles
// Role assignment name is deterministic based on scope, principal, and role to ensure idempotency
resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: empty(nameSuffix) ? guid(storageAccount.id, principalID, roleDefinitionID) : guid(storageAccount.id, principalID, roleDefinitionID, nameSuffix)
  scope: storageAccount
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionID)
    principalId: principalID
    principalType: 'ServicePrincipal'
  }
}

output ROLE_ASSIGNMENT_NAME string = storageRoleAssignment.name
