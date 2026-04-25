# Archived Background: Multi-Region Cluster Discovery

This note is retained for older workset-monitor and cluster-discovery context. Current Ursa runtime work should start from the root [README](../README.md) and the checked-in config example [../config/ursa-config.example.yaml](../config/ursa-config.example.yaml).

## Current Shape

Ursa still reads configured AWS regions through `daylib_ursa.ursa_config.UrsaConfig`. Cluster API routes expose create options, cluster listing, cluster inspection, delete planning, and delete execution through the authenticated Ursa API.

Example config shape:

```yaml
aws_profile: lsmc

regions:
  - us-west-2:
      ssh_pem: ~/.ssh/cluster-us-west-2.pem
  - us-east-1:
      ssh_pem: ~/.ssh/cluster-us-east-1.pem
  - eu-central-1:
      ssh_pem: ~/.ssh/cluster-eu-central-1.pem
```

## Delegated Boundaries

- Use `ursa ...` for Ursa-owned runtime operations.
- Use `tapdb ...` only where Ursa delegates TapDB DB lifecycle.
- Use `daylily-ec ...` for execution-plane staging, cluster, and workflow operations that Ursa delegates.

## TapDB Runtime

TapDB connectivity and namespace are selected separately from region scanning:

- `tapdb_client_id`
- `tapdb_database_name`
- `tapdb_config_path`
- `tapdb_env`
- `tapdb_domain_registry_path`
- `tapdb_prefix_ownership_registry_path`

See [../config/ursa-config.example.yaml](../config/ursa-config.example.yaml) and `daylib_ursa.integrations.tapdb_runtime` for the current runtime shape.
