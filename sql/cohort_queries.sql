-- Cohort description queries for the cleaned encounter table.
-- Run with DuckDB via readmission.sql_analysis. These describe the full
-- analysis cohort. Model metrics are computed later on held-out patients.

-- name: cohort_overview
SELECT
    COUNT(*) AS encounters,
    COUNT(DISTINCT patient_nbr) AS patients,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate,
    ROUND(AVG(time_in_hospital), 2) AS avg_length_of_stay_days,
    ROUND(AVG(number_inpatient), 2) AS avg_prior_inpatient_stays
FROM encounters;

-- name: readmission_by_age
SELECT
    age,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM encounters
GROUP BY age
ORDER BY age;

-- name: readmission_by_primary_diagnosis
SELECT
    diag_1_group AS primary_diagnosis,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM encounters
GROUP BY diag_1_group
ORDER BY readmission_rate DESC, encounters DESC;

-- name: readmission_by_prior_inpatient
SELECT
    CASE
        WHEN number_inpatient >= 4 THEN '4+'
        ELSE CAST(number_inpatient AS VARCHAR)
    END AS prior_inpatient_stays,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM encounters
GROUP BY 1
ORDER BY 1;

-- name: high_utilizer_gap
WITH bands AS (
    SELECT
        readmitted_30d,
        CASE
            WHEN number_inpatient = 0 THEN 'No prior inpatient stay'
            ELSE 'One or more prior inpatient stays'
        END AS utilizer
    FROM encounters
)
SELECT
    utilizer,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM bands
GROUP BY utilizer
ORDER BY utilizer;

-- name: readmission_by_a1c
SELECT
    a1c_result,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM encounters
GROUP BY a1c_result
ORDER BY encounters DESC;

-- name: readmission_by_discharge
SELECT
    discharge_group,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM encounters
GROUP BY discharge_group
ORDER BY encounters DESC;

-- name: top_specialties
SELECT
    medical_specialty,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d), 4) AS readmission_rate
FROM encounters
GROUP BY medical_specialty
ORDER BY encounters DESC
LIMIT 12;
