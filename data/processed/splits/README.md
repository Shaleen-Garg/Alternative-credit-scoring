# Frozen application-level split IDs

These files contain only `SK_ID_CURR` identifiers. They freeze the Phase 7A/7B/8/9/10 borrower assignment and order for the Home Credit `application_train` modeling population. All future experiments load these files; they must not regenerate the split.

The split was generated once using the original two-stage `train_test_split` procedure, stratified by `TARGET`, seed 42: first reserve 15% for test; then reserve 15/85 of the remainder for validation. Counts are train 215,257, validation 46,127, and test 46,127. IDs are disjoint and their union contains all 307,511 modeling records.

Files: `train_ids.csv`, `validation_ids.csv`, `test_ids.csv`.
