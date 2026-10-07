-- Data Quality Checks

-- 1. Check uniqueness of application IDs
SELECT 
    COUNT(SK_ID_CURR) AS total_rows,
    COUNT(DISTINCT SK_ID_CURR) AS unique_ids
FROM application_train;

-- 2. Verify temporal leakage (no future dates in bureau)
SELECT COUNT(*) AS future_bureau_records
FROM bureau 
WHERE DAYS_CREDIT > 0;

-- 3. Check for TARGET column leakage in features (ensure TARGET only exists once)
-- (Handled logically in the pipeline, but conceptual query below)
-- SELECT TARGET, COUNT(*) FROM feature_master GROUP BY TARGET;

-- 4. Verify thin-file counts match expectation
SELECT 
    COUNT(*) as thin_file_count
FROM feature_master 
WHERE BUREAU_CREDIT_COUNT = 0;
