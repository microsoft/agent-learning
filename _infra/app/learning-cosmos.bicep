// =========================================
// Agent Learning Cosmos DB Resources
// =========================================
// This module provisions the Cosmos DB database and containers
// required for the Agent Learning SDK's policy-bandit RL loop:
// - learning_episodes: Agent interactions (input, actions, output)
// - learning_metrics:  Per-step / per-episode metric results
// - learning_policies: Persisted SoftmaxPolicy snapshots
// - learning_rewards:  Per-metric + aggregate rewards per episode
// - learning_runs:     Training run records
//
// All containers use `/agent_id` as the partition key.

@description('Name of the parent Azure Cosmos DB account.')
param parentAccountName string

@description('Name of the Agent Learning database (default: dq_rl).')
param databaseName string = 'dq_rl'

@description('Tags for all resources.')
param tags object = {}

// Reference to existing Cosmos account
resource account 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: parentAccountName
}

// =========================================
// Agent Learning Database
// =========================================
resource learningDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  name: databaseName
  parent: account
  tags: tags
  properties: {
    resource: {
      id: databaseName
    }
  }
}

// =========================================
// learning_episodes Container
// Stores agent interactions (prompt → actions → response)
// Partition key: /agent_id
// =========================================
resource episodesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'learning_episodes'
  parent: learningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'learning_episodes'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
      // TTL disabled by default - episodes are long-lived training data
      defaultTtl: -1
    }
  }
}

// =========================================
// learning_metrics Container
// Stores per-step / per-episode metric results
// Partition key: /agent_id
// =========================================
resource metricsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'learning_metrics'
  parent: learningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'learning_metrics'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// learning_policies Container
// Stores persisted SoftmaxPolicy snapshots
// Partition key: /agent_id
// =========================================
resource policiesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'learning_policies'
  parent: learningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'learning_policies'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// learning_rewards Container
// Stores per-metric + aggregate rewards attached to episodes
// Partition key: /agent_id
// =========================================
resource rewardsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'learning_rewards'
  parent: learningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'learning_rewards'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// learning_runs Container
// Stores training run records
// Partition key: /agent_id
// =========================================
resource runsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'learning_runs'
  parent: learningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'learning_runs'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// Outputs
// =========================================
output databaseName string = learningDatabase.name
output episodesContainerName string = episodesContainer.name
output metricsContainerName string = metricsContainer.name
output policiesContainerName string = policiesContainer.name
output rewardsContainerName string = rewardsContainer.name
output runsContainerName string = runsContainer.name
