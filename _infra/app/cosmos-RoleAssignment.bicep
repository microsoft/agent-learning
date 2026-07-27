@description('Name of the Cosmos DB account.')
param cosmosAccountName string

@description('Role definition ID to assign.')
param roleDefinitionID string

@description('Principal ID to assign the role to.')
param principalID string

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: cosmosAccountName
}

// Cosmos DB SQL Role Assignment for data plane access
// Uses built-in role or custom role definition
resource cosmosRoleAssignment 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = {
  name: guid(cosmosAccount.id, principalID, roleDefinitionID)
  parent: cosmosAccount
  properties: {
    principalId: principalID
    roleDefinitionId: '${cosmosAccount.id}/sqlRoleDefinitions/${roleDefinitionID}'
    scope: cosmosAccount.id
  }
}

output roleAssignmentId string = cosmosRoleAssignment.id
