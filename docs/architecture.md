# Infrastructure architecture

Private topology so the Agent Learning SDK runs on a virtual machine inside the
VNet, reaching Cosmos DB and Storage only over private endpoints and
authenticating with the VM's managed identity (no keys or secrets on the box).

The diagram below is embedded from [architecture.mermaid](architecture.mermaid).

```mermaid
flowchart LR
  subgraph RG[Resource group]
    subgraph VNET[VNet 10.0.0.0/16]
      subgraph VMSUB[vm subnet 10.0.1.0/24]
        VM[Linux VM<br/>system-assigned identity<br/>runs the SDK]
      end
      subgraph PESUB[private-endpoints subnet 10.0.2.0/24]
        CPE[Cosmos private endpoint]
        SPE[Blob private endpoint]
      end
    end
    COSMOS[Cosmos DB<br/>serverless, AAD-only<br/>public access Disabled]
    STG[Storage account<br/>public access Disabled]
  end
  VM -->|managed identity| CPE --> COSMOS
  VM -->|managed identity| SPE --> STG
```
