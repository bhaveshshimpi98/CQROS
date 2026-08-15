# Post-attempt hashes (incomplete regeneration)

**Captured:** see `after.json` `timestamp_utc`  
**Verdict:** FAIL — not all five partitions regenerated.

## Walk Forward

| TF | Status | Bytes | SHA-256 | vs baseline |
|----|--------|------:|---------|-------------|
| 1d | regenerated | 185,255 | `B2391EE4ECBCD89E145015A488789415C410F6A5C050A818192EAEB7DFE58469` | CHANGED |
| 4h | not regenerated | 141,704 | `888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F` | UNCHANGED |
| 1h | not started | 470,948 | `735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373` | UNCHANGED |
| 15m | not started | 226,656 | `97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806` | UNCHANGED |
| 5m | not started | 390,938 | `937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D` | UNCHANGED |

## Factor Selection / Factor Validation / Purged CV

All fifteen panels: **UNCHANGED** (byte-for-byte vs `baseline.json`).

Details: `hashes_after.csv`, `after.json`.
