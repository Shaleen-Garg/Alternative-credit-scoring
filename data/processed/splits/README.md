# Frozen borrower splits

These files contain only SK_ID_CURR identifiers. They freeze the borrower assignment and row order for the Home Credit application_train modeling population. All analysis modules load these files and must not regenerate the split.

The split is stratified and contains 215,257 training, 46,127 validation, and 46,127 test borrowers. The three files are mutually disjoint and cover all 307,511 applicants.
