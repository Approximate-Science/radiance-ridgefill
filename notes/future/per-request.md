# Parked: per-request ON/OFF (Stage F)

Parked by Dylan on 2026-10-05 for a future update ("There's a lot of moving parts there. Don't get rid of
the code, just save it for later"). This release selects the mode server-wide (`RADIANCE_RIDGEFILL=off|quality|speed`).

What exists, kept as is:
- Design: `~/AI-Work/radiance-kva-plugin-20261004/fix-246/` (`PLAN-FIX.md` §11, `REFUTATION-2-marker.md`,
  `HANDOVER-FIX.md` §10, requirements R101-R134) and the marker spec `worker-context/ridgefill-marker-spec.json`.
- Verified prerequisites: `notes/stagef-verify.md` (14/14 engine claims confirmed at radiance 140987f;
  marker pieces are single tokens; eos 248044). Re-verify against the radiance release current at resumption.
- Implementation breakdown: `notes/stagef-plan.md` (work items, ops, tests, gate order, critical path, sizes).
- Tools: `tools/ridgefill_template.py` (marker snippet / merge into a model's template / check / strip),
  `tools/template_identity.py` (engine-side check that a merged template renders marker-free requests
  byte-identically), `notes/template.md`, `docs/release/TEMPLATE-README.md` (draft).
- The release E2E script's per-request case (`scripts/e2e_fresh.sh`, behind `RK_E2E_STAGE_F=1`).

Not shipped in this release: the chat-template package (its only job is the per-request marker).
Resumption order: rebuild on the core/adapter split, then follow `notes/stagef-plan.md` from F.2.
