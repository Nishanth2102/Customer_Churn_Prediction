import gradio as gr
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "catboost_churn_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# MODEL FEATURE LIST
# EXACTLY MATCHES THE NOTEBOOK
# ============================================================

NUMERICAL_FEATURES = [
    "Age",
    "Tenure_Months",
    "Auto_Renewal_Flag",
    "Usage_Frequency",
    "Days_Since_Last_Activity",
    "Feature_Usage_Count",
    "Avg_Session_Duration_Min",
    "Monthly_Charges",
    "Total_Charges",
    "Discount_Applied_Pct",
    "Late_Payment_Count",
    "Support_Tickets_Raised",
    "Complaint_Count",
    "Avg_Resolution_Time_Hrs",
    "Satisfaction_Score",
    "Referral_Count",
    "Loyalty_Program_Member",
    "Email_Open_Rate_Pct",
    "Signup_Year",
    "Signup_Month",
    "Signup_Quarter",
    "Signup_DayOfWeek",
    "Avg_Monthly_Charge_From_Total",
    "Charge_Difference",
    "Charge_Per_Tenure",
    "Total_Customer_Issues"
]


CATEGORICAL_FEATURES = [
    "Gender",
    "City",
    "Income_Level",
    "Contract_Type",
    "Plan_Type",
    "Payment_Method",
    "Upsell_Downgrade_History",
    "Tenure_Category",
    "Payment_Risk",
    "Inactivity_Level",
    "Satisfaction_Level",
    "Age_Group",
    "Contract_Renewal_Status"
]


ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# FEATURE ENGINEERING
# EXACTLY MATCHES THE NOTEBOOK
# ============================================================

def create_features(
    age,
    gender,
    city,
    income_level,
    signup_date,
    contract_type,
    plan_type,
    payment_method,
    auto_renewal_flag,
    loyalty_program_member,
    tenure_months,
    usage_frequency,
    days_since_last_activity,
    feature_usage_count,
    avg_session_duration_min,
    monthly_charges,
    total_charges,
    discount_applied_pct,
    late_payment_count,
    support_tickets_raised,
    complaint_count,
    avg_resolution_time_hrs,
    satisfaction_score,
    referral_count,
    email_open_rate_pct,
    upsell_downgrade_history
):

    # --------------------------------------------------------
    # Create raw customer dataframe
    # --------------------------------------------------------

    signup_date = pd.to_datetime(
        signup_date,
        errors="coerce"
    )

    if pd.isna(signup_date):
        raise ValueError(
            "Invalid Signup Date. Please use YYYY-MM-DD format."
        )

    customer = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "City": city,
        "Income_Level": income_level,
        "Signup_Date": signup_date,
        "Contract_Type": contract_type,
        "Plan_Type": plan_type,
        "Payment_Method": payment_method,
        "Auto_Renewal_Flag": int(auto_renewal_flag),
        "Loyalty_Program_Member": int(loyalty_program_member),
        "Tenure_Months": tenure_months,
        "Usage_Frequency": usage_frequency,
        "Days_Since_Last_Activity": days_since_last_activity,
        "Feature_Usage_Count": feature_usage_count,
        "Avg_Session_Duration_Min": avg_session_duration_min,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Discount_Applied_Pct": discount_applied_pct,
        "Late_Payment_Count": late_payment_count,
        "Support_Tickets_Raised": support_tickets_raised,
        "Complaint_Count": complaint_count,
        "Avg_Resolution_Time_Hrs": avg_resolution_time_hrs,
        "Satisfaction_Score": satisfaction_score,
        "Referral_Count": referral_count,
        "Email_Open_Rate_Pct": email_open_rate_pct,
        "Upsell_Downgrade_History": upsell_downgrade_history
    }])


    # ========================================================
    # NOTEBOOK FEATURE ENGINEERING
    # ========================================================

    # --------------------------------------------------------
    # Signup date features
    # --------------------------------------------------------

    customer["Signup_Year"] = (
        customer["Signup_Date"].dt.year
    )

    customer["Signup_Month"] = (
        customer["Signup_Date"].dt.month
    )

    customer["Signup_Quarter"] = (
        customer["Signup_Date"].dt.quarter
    )

    customer["Signup_DayOfWeek"] = (
        customer["Signup_Date"].dt.dayofweek
    )


    # --------------------------------------------------------
    # Tenure Category
    # --------------------------------------------------------

    customer["Tenure_Category"] = pd.cut(
        customer["Tenure_Months"],
        bins=[
            -1,
            6,
            12,
            24,
            48,
            np.inf
        ],
        labels=[
            "New",
            "Short_Term",
            "Medium_Term",
            "Long_Term",
            "Very_Long_Term"
        ]
    )


    # --------------------------------------------------------
    # Financial features
    # --------------------------------------------------------

    customer["Avg_Monthly_Charge_From_Total"] = (
        customer["Total_Charges"]
        /
        customer["Tenure_Months"].replace(
            0,
            np.nan
        )
    ).fillna(
        customer["Monthly_Charges"]
    )


    customer["Charge_Difference"] = (
        customer["Total_Charges"]
        -
        (
            customer["Monthly_Charges"]
            *
            customer["Tenure_Months"]
        )
    )


    customer["Charge_Per_Tenure"] = (
        customer["Total_Charges"]
        /
        (
            customer["Tenure_Months"]
            + 1
        )
    )


    # --------------------------------------------------------
    # Payment Risk
    # --------------------------------------------------------

    customer["Payment_Risk"] = pd.cut(
        customer["Late_Payment_Count"],
        bins=[
            -1,
            0,
            2,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )


    # --------------------------------------------------------
    # Total Customer Issues
    # --------------------------------------------------------

    customer["Total_Customer_Issues"] = (
        customer["Support_Tickets_Raised"]
        +
        customer["Complaint_Count"]
    )


    # --------------------------------------------------------
    # Inactivity Level
    # --------------------------------------------------------

    customer["Inactivity_Level"] = pd.cut(
        customer["Days_Since_Last_Activity"],
        bins=[
            -1,
            7,
            30,
            90,
            np.inf
        ],
        labels=[
            "Active",
            "Recently_Inactive",
            "Inactive",
            "Highly_Inactive"
        ]
    )


    # --------------------------------------------------------
    # Satisfaction Level
    # --------------------------------------------------------

    customer["Satisfaction_Level"] = pd.cut(
        customer["Satisfaction_Score"],
        bins=[
            0,
            3,
            7,
            10
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )


    # --------------------------------------------------------
    # Age Group
    # --------------------------------------------------------

    customer["Age_Group"] = pd.cut(
        customer["Age"],
        bins=[
            0,
            25,
            35,
            50,
            65,
            np.inf
        ],
        labels=[
            "Young",
            "Adult",
            "Middle_Aged",
            "Senior",
            "Older"
        ]
    )


    # --------------------------------------------------------
    # Contract Renewal Status
    # --------------------------------------------------------

    customer["Contract_Renewal_Status"] = (
        customer["Contract_Type"].astype(str)
        + "_"
        +
        np.where(
            customer["Auto_Renewal_Flag"] == 1,
            "AutoRenew",
            "Manual"
        )
    )


    # ========================================================
    # REMOVE COLUMNS EXACTLY LIKE NOTEBOOK
    # ========================================================

    drop_columns = [
        "Signup_Date"
    ]

    customer = customer.drop(
        columns=drop_columns,
        errors="ignore"
    )


    # ========================================================
    # ENSURE EXACT MODEL FEATURE SET
    # ========================================================

    customer = customer[ALL_FEATURES]


    return customer


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_churn(
    age,
    gender,
    city,
    income_level,
    signup_date,
    contract_type,
    plan_type,
    payment_method,
    auto_renewal,
    loyalty_member,
    tenure_months,
    usage_frequency,
    days_since_last_activity,
    feature_usage_count,
    avg_session_duration,
    monthly_charges,
    total_charges,
    discount_applied,
    late_payment_count,
    support_tickets,
    complaint_count,
    avg_resolution_time,
    satisfaction_score,
    referral_count,
    email_open_rate,
    upsell_downgrade_history
):

    try:

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if age < 18 or age > 100:
            return (
                "❌ Invalid Age",
                "Age must be between 18 and 100.",
                0
            )


        if discount_applied < 0 or discount_applied > 100:
            return (
                "❌ Invalid Discount",
                "Discount must be between 0 and 100%.",
                0
            )


        if email_open_rate < 0 or email_open_rate > 100:
            return (
                "❌ Invalid Email Open Rate",
                "Email open rate must be between 0 and 100%.",
                0
            )


        if satisfaction_score < 1 or satisfaction_score > 10:
            return (
                "❌ Invalid Satisfaction Score",
                "Satisfaction score must be between 1 and 10.",
                0
            )


        # ----------------------------------------------------
        # Create exact model features
        # ----------------------------------------------------

        X_new = create_features(
            age=age,
            gender=gender,
            city=city,
            income_level=income_level,
            signup_date=signup_date,
            contract_type=contract_type,
            plan_type=plan_type,
            payment_method=payment_method,
            auto_renewal_flag=auto_renewal,
            loyalty_program_member=loyalty_member,
            tenure_months=tenure_months,
            usage_frequency=usage_frequency,
            days_since_last_activity=days_since_last_activity,
            feature_usage_count=feature_usage_count,
            avg_session_duration_min=avg_session_duration,
            monthly_charges=monthly_charges,
            total_charges=total_charges,
            discount_applied_pct=discount_applied,
            late_payment_count=late_payment_count,
            support_tickets_raised=support_tickets,
            complaint_count=complaint_count,
            avg_resolution_time_hrs=avg_resolution_time,
            satisfaction_score=satisfaction_score,
            referral_count=referral_count,
            email_open_rate_pct=email_open_rate,
            upsell_downgrade_history=upsell_downgrade_history
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(X_new)[0]

        probability = model.predict_proba(
            X_new
        )[0][1]


        probability_percent = (
            probability * 100
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 1:

            result = "🔴 CUSTOMER LIKELY TO CHURN"

            explanation = (
                f"The model predicts that this customer "
                f"is likely to churn.\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )

        else:

            result = "🟢 CUSTOMER LIKELY TO STAY"

            explanation = (
                f"The model predicts that this customer "
                f"is likely to remain.\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )


        return (
            result,
            explanation,
            probability_percent
        )


    except Exception as e:

        return (
            "❌ Prediction Error",
            str(e),
            0
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Customer Churn Prediction"
) as demo:

    gr.Markdown(
        """
        # 👤 Customer Churn Prediction

        Enter the customer's details below to predict
        churn probability using the trained CatBoost model.
        """
    )


    # ========================================================
    # CUSTOMER DETAILS
    # ========================================================

    with gr.Accordion(
        "👤 Customer Details",
        open=True
    ):

        with gr.Row():

            age = gr.Number(
                label="Age",
                value=30,
                minimum=18,
                maximum=100,
                precision=0
            )

            gender = gr.Dropdown(
                label="Gender",
                choices=[
                    "Female",
                    "Male",
                    "Other"
                ],
                value="Male"
            )

            city = gr.Dropdown(
                label="City",
                choices=[
                    "Bengaluru",
                    "Chennai",
                    "Delhi",
                    "Hyderabad",
                    "Kolkata",
                    "Mumbai",
                    "Pune"
                ],
                value="Chennai"
            )

            income_level = gr.Dropdown(
                label="Income Level",
                choices=[
                    "Low",
                    "Medium",
                    "High"
                ],
                value="Medium"
            )


    # ========================================================
    # SIGNUP & SUBSCRIPTION
    # ========================================================

    with gr.Accordion(
        "📅 Subscription Details",
        open=True
    ):

        with gr.Row():

            signup_date = gr.Textbox(
                label="Signup Date",
                value="2024-01-01",
                placeholder="YYYY-MM-DD"
            )

            contract_type = gr.Dropdown(
                label="Contract Type",
                choices=[
                    "Annual",
                    "Monthly",
                    "Prepaid"
                ],
                value="Monthly"
            )

            plan_type = gr.Dropdown(
                label="Plan Type",
                choices=[
                    "Basic",
                    "Premium",
                    "Standard"
                ],
                value="Standard"
            )

            payment_method = gr.Dropdown(
                label="Payment Method",
                choices=[
                    "Credit Card",
                    "Debit Card",
                    "Net Banking",
                    "UPI",
                    "Wallet"
                ],
                value="UPI"
            )


        with gr.Row():

            auto_renewal = gr.Checkbox(
                label="Auto Renewal",
                value=True
            )

            loyalty_member = gr.Checkbox(
                label="Loyalty Program Member",
                value=False
            )

            upsell_downgrade_history = gr.Dropdown(
                label="Upsell / Downgrade History",
                choices=[
                    "Downgraded",
                    "No History",
                    "Upgraded"
                ],
                value="No History"
            )


    # ========================================================
    # USAGE & ENGAGEMENT
    # ========================================================

    with gr.Accordion(
        "📊 Usage & Engagement",
        open=True
    ):

        with gr.Row():

            tenure_months = gr.Number(
                label="Tenure (Months)",
                value=12,
                minimum=0,
                precision=0
            )

            usage_frequency = gr.Number(
                label="Usage Frequency",
                value=15,
                minimum=0
            )

            days_since_last_activity = gr.Number(
                label="Days Since Last Activity",
                value=5,
                minimum=0,
                precision=0
            )

            feature_usage_count = gr.Number(
                label="Feature Usage Count",
                value=5,
                minimum=0,
                precision=0
            )


        with gr.Row():

            avg_session_duration = gr.Number(
                label="Avg Session Duration (Min)",
                value=30,
                minimum=0
            )

            referral_count = gr.Number(
                label="Referral Count",
                value=0,
                minimum=0,
                precision=0
            )

            email_open_rate = gr.Number(
                label="Email Open Rate (%)",
                value=50,
                minimum=0,
                maximum=100
            )


    # ========================================================
    # BILLING
    # ========================================================

    with gr.Accordion(
        "💳 Billing Details",
        open=True
    ):

        with gr.Row():

            monthly_charges = gr.Number(
                label="Monthly Charges",
                value=999,
                minimum=0
            )

            total_charges = gr.Number(
                label="Total Charges",
                value=11988,
                minimum=0
            )

            discount_applied = gr.Number(
                label="Discount Applied (%)",
                value=0,
                minimum=0,
                maximum=100
            )


        with gr.Row():

            late_payment_count = gr.Number(
                label="Late Payment Count",
                value=0,
                minimum=0,
                precision=0
            )


    # ========================================================
    # SUPPORT & SATISFACTION
    # ========================================================

    with gr.Accordion(
        "🎧 Support & Satisfaction",
        open=True
    ):

        with gr.Row():

            support_tickets = gr.Number(
                label="Support Tickets Raised",
                value=0,
                minimum=0,
                precision=0
            )

            complaint_count = gr.Number(
                label="Complaint Count",
                value=0,
                minimum=0,
                precision=0
            )

            avg_resolution_time = gr.Number(
                label="Avg Resolution Time (Hrs)",
                value=5,
                minimum=0
            )


        with gr.Row():

            satisfaction_score = gr.Slider(
                label="Satisfaction Score",
                minimum=1,
                maximum=10,
                value=7,
                step=1
            )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = gr.Button(
        "🔮 Predict Customer Churn",
        variant="primary",
        size="lg"
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    gr.Markdown(
        "## Prediction Result"
    )


    with gr.Row():

        prediction_result = gr.Textbox(
            label="Prediction",
            interactive=False
        )

        churn_probability = gr.Number(
            label="Churn Probability (%)",
            interactive=False
        )


    prediction_explanation = gr.Textbox(
        label="Details",
        lines=4,
        interactive=False
    )


    # ========================================================
    # BUTTON EVENT
    # ========================================================

    predict_button.click(
        fn=predict_churn,

        inputs=[
            age,
            gender,
            city,
            income_level,
            signup_date,
            contract_type,
            plan_type,
            payment_method,
            auto_renewal,
            loyalty_member,
            tenure_months,
            usage_frequency,
            days_since_last_activity,
            feature_usage_count,
            avg_session_duration,
            monthly_charges,
            total_charges,
            discount_applied,
            late_payment_count,
            support_tickets,
            complaint_count,
            avg_resolution_time,
            satisfaction_score,
            referral_count,
            email_open_rate,
            upsell_downgrade_history
        ],

        outputs=[
            prediction_result,
            prediction_explanation,
            churn_probability
        ]
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.Markdown(
        """
        ---
        **Model:** Tuned CatBoost Classifier  
        **Task:** Customer Churn Prediction  
        **Output:** Churn prediction + churn probability
        """
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )