# API

- `GET /health`: baseline model/device availability.
- `POST /predict`: legacy local-data route; it only works when its configured directories form a matched patch.
- `POST /predict/upload`: multipart fields `s1_zip` and `s2_zip`; accepts one validated BigEarthNet-compatible patch pair.

Responses distinguish model predictions, spectral/radar measurements, heuristic evidence, and rule-based explanation.
