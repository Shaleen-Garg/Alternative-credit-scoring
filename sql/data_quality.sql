SELECT 'invalid_application_target' AS check_name, COUNT(*) AS violation_count
FROM application_train
WHERE TARGET IS NULL OR TARGET NOT IN (0, 1)

UNION ALL
SELECT 'future_bureau_record', COUNT(*)
FROM bureau
WHERE DAYS_CREDIT > 0

UNION ALL
SELECT 'future_previous_application', COUNT(*)
FROM previous_application
WHERE DAYS_DECISION > 0

UNION ALL
SELECT 'future_installment_due_date', COUNT(*)
FROM installments_payments
WHERE DAYS_INSTALMENT > 0

UNION ALL
SELECT 'future_installment_payment_date', COUNT(*)
FROM installments_payments
WHERE DAYS_ENTRY_PAYMENT > 0

UNION ALL
SELECT 'feature_row_count_mismatch', ABS(
    (SELECT COUNT(*) FROM feature_master) -
    (SELECT COUNT(*) FROM application_train)
)

UNION ALL
SELECT 'duplicate_feature_applicant', COUNT(*) - COUNT(DISTINCT SK_ID_CURR)
FROM feature_master;
