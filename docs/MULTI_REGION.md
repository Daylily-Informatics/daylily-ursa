# Multi-Region Cluster Discovery

Ursa can scan multiple AWS regions for ParallelCluster instances and run worksets on clusters in any configured region.

TapDB persistence is Postgres-backed and configured via `TAPDB_*` strict-namespace environment variables. TapDB is not configured as per-region “tables”.

## Configure Regions

Create `~/.config/ursa/ursa-config.yaml`:

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

The `regions` list controls which AWS regions Ursa will scan for clusters, and lets you provide region-specific SSH keys for headnode access.

## Where It’s Used

- Cluster discovery and region-specific SSH configuration is loaded via `daylily_ursa.ursa_config.UrsaConfig` ([daylily_ursa/ursa_config.py](/Users/jmajor/projects/daylily/daylily-ursa/daylily_ursa/ursa_config.py)).
- Workset monitor/worker AWS settings are primarily driven by the workset monitor YAML config (see `config/workset-monitor-config.yaml` and `config/workset-monitor-config.template.yaml`).

## TapDB Note

TapDB connectivity and namespace are configured separately via:

- `TAPDB_STRICT_NAMESPACE=1`
- `TAPDB_CLIENT_ID`
- `TAPDB_DATABASE_NAME`
- `TAPDB_ENV`

See `config/ursa-config.example.yaml` for an example and `daylily_ursa.tapdb_graph.backend.TapDBBackend` for enforcement ([daylily_ursa/tapdb_graph/backend.py](/Users/jmajor/projects/daylily/daylily-ursa/daylily_ursa/tapdb_graph/backend.py)).

