import streamlit as st
import pandas as pd
import numpy as np
import io

st.title("📊 CSV Column Analyzer with Editable Suggested Types + Unique Value Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔹 Preview of Data")
    st.dataframe(df.head())

    # Build metadata summary
    summary = []
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=True)
        total_vals = df.shape[0]
        null_vals = df[col].isnull().sum()
        dtype = df[col].dtype

        # Detect type
        if unique_vals == total_vals:
            detected_type = "Unique Character"
        elif np.issubdtype(dtype, np.number):
            detected_type = "Numerical"
        else:
            detected_type = "String"
            
        convert_type = detected_type

        summary.append([
            col, unique_vals, total_vals, null_vals,
            detected_type, convert_type
        ])

    summary_df = pd.DataFrame(summary, columns=[
        "Column Name", "Unique Records", "Total Records",
        "Null Records", "Detected Type", "Suggested Type"
    ])

    st.subheader("📌 Column Summary (Choose Suggested Type)")
    edited_df = st.data_editor(
        summary_df,
        use_container_width=True,
        num_rows="fixed",
        key="editable_summary",
        column_config={
            "Suggested Type": st.column_config.SelectboxColumn(
                "Suggested Type",
                help="Select the type to which you want to convert this column",
                options=["Drop", "String", "Numerical"],
                required=True
            )
        }
    )

    # Track changes
    changes = {}
    for i, row in summary_df.iterrows():
        old_val = row["Suggested Type"]
        new_val = edited_df.loc[i, "Suggested Type"]
        if old_val != new_val:
            changes[row["Column Name"]] = {"old": old_val, "new": new_val}

    if changes:
        st.subheader("🔄 Changes Made by User")
        changes_df = pd.DataFrame([
            {"Column Name": col, "Old Type": vals["old"], "New Type": vals["new"]}
            for col, vals in changes.items()
        ])
        st.dataframe(changes_df, use_container_width=True)

    # Confirmation button
    if st.button("✅ Confirm Changes"):
        st.success("User has confirmed the changes!")

        # Apply transformations
        processed_df = df.copy()
        for i, row in edited_df.iterrows():
            col = row["Column Name"]
            action = row["Suggested Type"]

            if action == "Drop":
                processed_df.drop(columns=[col], inplace=True)
            elif action == "String":
                processed_df[col] = processed_df[col].astype(str)
            elif action == "Numerical":
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        # Export updated table as CSV
        csv_buffer = io.StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Processed File (CSV)",
            data=csv_data,
            file_name="processed_file.csv",
            mime="text/csv"
        )

        # -----------------------------
        # 🔍 MULTI-COLUMN UNIQUE VALUE ANALYSIS
        # -----------------------------
        st.header("🔍 Multi-Column Unique Value Analysis")

        cols = processed_df.columns.tolist()
        group_col = st.selectbox("Select Column to Group By", cols)

        numeric_cols = processed_df.select_dtypes(include="number").columns.tolist()
        agg_cols = st.multiselect("Select Numerical Columns for Analysis", numeric_cols)

        if group_col and agg_cols:
            st.subheader(f"📊 Aggregation by '{group_col}'")

            # Aggregate sums
            grouped = processed_df.groupby(group_col)[agg_cols].sum().reset_index()

            # Calculate % contribution
            for col in agg_cols:
                grouped[f"{col}_%"] = (grouped[col] / grouped[col].sum() * 100).round(2)

            st.dataframe(grouped, use_container_width=True)

            # Download option
            csv_buffer2 = io.StringIO()
            grouped.to_csv(csv_buffer2, index=False)
            st.download_button(
                label="📥 Download Aggregated Analysis (CSV)",
                data=csv_buffer2.getvalue(),
                file_name="unique_value_analysis.csv",
                mime="text/csv"
            )

