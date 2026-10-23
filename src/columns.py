"""Column decisions for the credit risk engine.

Rule: a column may only become a feature if its value existed at the
moment the lending decision was made.

Every column of loans_raw appears in exactly one list below.
"""

# ---- the answer -------------------------------------------------------
TARGET = "target"
TARGET_SOURCE = ["loan_status"]

# ---- no information (empty, constant, or an identifier) ---------------
DROP_NO_INFO = [
    "id",            # unique per row
    "url",           # unique per row
    "member_id",     # 100% empty
    "policy_code",   # always 1.0
]

# ---- free text, unusable as-is ----------------------------------------
DROP_FREE_TEXT = [
    "emp_title",     # tens of thousands of spellings
    "title",         # borrower-written, overlaps 'purpose'
    "desc",          # 94% empty
]

# ---- decided after approval, not at application -----------------------
DROP_POST_APPROVAL = [
    "funded_amnt",
    "funded_amnt_inv",
]

# ---- leakage: money moving after the loan was funded ------------------
DROP_LEAKAGE_PAYMENTS = [
    "pymnt_plan",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
]

# ---- leakage: the borrower's credit record checked later --------------
DROP_LEAKAGE_LATER_CREDIT = [
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
]

# ---- leakage: hardship plans and settlements --------------------------
DROP_LEAKAGE_TROUBLE = [
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]

# ---- second applicant: legitimate, but 98.7% missing ------------------
# 'application_type' already records whether an application was joint.
DROP_JOINT = [
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]

# ---- dates: not features as-is, turned into numbers later -------------
DATE_COLUMNS = [
    "issue_d",           # when the loan was issued; used to split train/test by time
    "earliest_cr_line",  # becomes "length of credit history"
]

# ---- known in time, but a fairness risk: tested in slice 7 ------------
FLAG_FAIRNESS = [
    "zip_code",
    "addr_state",
]

# ---- Lending Club's own risk verdict ----------------------------------
# Known before funding, so not leakage. But using them means the model
# partly copies LC's grading model instead of judging the applicant.
# Kept separate so both versions can be compared in slice 6.
LENDER_ASSESSMENT = [
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
]


DROP_ALL = (
    TARGET_SOURCE
    + DROP_NO_INFO
    + DROP_FREE_TEXT
    + DROP_POST_APPROVAL
    + DROP_LEAKAGE_PAYMENTS
    + DROP_LEAKAGE_LATER_CREDIT
    + DROP_LEAKAGE_TROUBLE
    + DROP_JOINT
)

def feature_columns(all_columns, use_lender_assessment=False):
    """Return the usable feature columns, in their original order."""
    excluded = set(DROP_ALL) | set(DATE_COLUMNS) | {TARGET}
    if not use_lender_assessment:
        excluded |= set(LENDER_ASSESSMENT)
    return [c for c in all_columns if c not in excluded]


# text that is really an ordered number
NUMERIC_FROM_TEXT = ["term", "emp_length"]

# genuine categories, encoded in slice 3
CATEGORICAL = [
    "home_ownership",
    "verification_status",
    "purpose",
    "zip_code",
    "addr_state",
    "initial_list_status",
    "application_type",
    "disbursement_method",
]
