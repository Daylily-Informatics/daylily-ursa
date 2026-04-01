# Multi-Region Cluster Discovery

This is a legacy workset-monitor note retained for background. The current repo
surface is the analysis API described in [../README.md](../README.md).

Ursa can scan multiple AWS regions for ParallelCluster instances and run worksets on clusters in any configured region.

TapDB persistence is Postgres-backed and configured via the deployment TapDB
config file plus explicit `--config` and `--env` selection. TapDB is not
configured as per-region “tables”.

## Configure Regions

Create `~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml`:

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

- Cluster discovery and region-specific SSH configuration is loaded via `daylib_ursa.ursa_config.UrsaConfig` ([../daylib_ursa/ursa_config.py](../daylib_ursa/ursa_config.py)).
- Workset monitor/worker AWS settings are primarily driven by the workset monitor YAML config (see `config/workset-monitor-config.yaml` and `config/workset-monitor-config.template.yaml`).

## TapDB Note

TapDB connectivity and namespace are configured separately via:

- `tapdb_strict_namespace: true`
- `tapdb_client_id`
- `tapdb_database_name`
- `tapdb_env`

See `config/ursa-config.example.yaml` for an example and `daylib_ursa.tapdb_graph.backend.TapDBBackend` for enforcement ([../daylib_ursa/tapdb_graph/backend.py](../daylib_ursa/tapdb_graph/backend.py)).
