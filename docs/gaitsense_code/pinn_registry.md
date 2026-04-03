# PINN Checkpoint Registry

Immutable record of all PINN training checkpoints. Each entry is appended after archival. SHA-256 hashes are computed over the raw `.pt` binary. Validation status is updated by pinn-validator only — never by the archivist.

---

## v1 — 2026-04-03

| Field | Value |
|---|---|
| SHA-256 | `062bfa2c810e40aa936de444e45ac35443f4e4b7ec3faf70b3a6a7b4a37bc376` |
| Best epoch | 20 |
| Best val loss | 0.442229 |
| Early stop epoch | 190 |
| Physics weight at best | 0.20 |
| Validated | NO — pending pinn-validator |
| Bills | bill_loss_weights_v1, bill_train_config_v1, bill_data_config_v1 |
