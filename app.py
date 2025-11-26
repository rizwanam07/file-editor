import streamlit as st
import pandas as pd
import io
import os

# 1. PAGE CONFIG
st.set_page_config(page_title="Logic File Editor", page_icon="⚡", layout="wide")

# CSS to hide sidebar and footer
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarCollapsedControl"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stFileUploader {padding-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Auto-Pipeline: Append & Replace (No AI)")
st.markdown("This tool uses strict logic to merge files. No API Key required.")

# 2. FILE UPLOADER SECTION
col1, col2, col3 = st.columns(3)

with col1:
    st.info("1. Base File (Main)")
    base_file = st.file_uploader("Upload Main Dataset", type=["csv", "xlsx", "xls"], key="base")

with col2:
    st.warning("2. Append File (Add Rows)")
    append_file = st.file_uploader("Upload Rows to Add", type=["csv", "xlsx", "xls"], key="append")

with col3:
    st.error("3. Replace File (Update Values)")
    replace_file = st.file_uploader("Upload Updates", type=["csv", "xlsx", "xls"], key="replace")

# 3. PROCESSING LOGIC
if base_file:
    # --- LOAD BASE FILE ---
    try:
        if base_file.name.endswith('.csv'):
            df_base = pd.read_csv(base_file)
            file_type = "csv"
        else:
            df_base = pd.read_excel(base_file)
            file_type = "excel"
            
        initial_count = len(df_base)
        st.write(f"### 1. Base Data Loaded ({initial_count} rows)")
        with st.expander("Preview Base Data"):
            st.dataframe(df_base.head())

        # --- STEP A: APPEND LOGIC ---
        if append_file:
            st.divider()
            st.subheader("2. Append Operation")
            try:
                if append_file.name.endswith('.csv'):
                    df_append = pd.read_csv(append_file)
                else:
                    df_append = pd.read_excel(append_file)
                
                st.write(f"Found **{len(df_append)}** new rows to add.")
                
                # Perform the append
                # ignore_index=True ensures we don't get duplicate index numbers
                df_base = pd.concat([df_base, df_append], ignore_index=True)
                
                st.success(f"✅ Appended successfully! Total rows: {len(df_base)}")
                
            except Exception as e:
                st.error(f"Error appending file: {e}")

        # --- STEP B: REPLACE LOGIC ---
        if replace_file:
            st.divider()
            st.subheader("3. Replace/Update Operation")
            try:
                if replace_file.name.endswith('.csv'):
                    df_replace = pd.read_csv(replace_file)
                else:
                    df_replace = pd.read_excel(replace_file)

                # UI: Ask user how to match rows
                st.info("Select how to match rows between the Main file and the Update file.")
                
                # Find common columns
                common_cols = list(set(df_base.columns) & set(df_replace.columns))
                
                c1, c2 = st.columns(2)
                with c1:
                    # User selects ID columns (e.g. Program, Brand)
                    match_cols = st.multiselect(
                        "Select ID Column(s) to match on (e.g. Program Name, Brand):", 
                        options=common_cols,
                        default=common_cols[0] if common_cols else None
                    )
                
                with c2:
                    # User selects Value columns (e.g. Power, kW)
                    # We default to all columns in replace file EXCEPT the match cols
                    potential_value_cols = [c for c in df_replace.columns if c not in match_cols]
                    value_cols = st.multiselect(
                        "Select Column(s) to Update (e.g. Power, kW):", 
                        options=potential_value_cols,
                        default=potential_value_cols
                    )

                if match_cols and value_cols:
                    if st.button("Apply Updates"):
                        # STRICT UPDATE LOGIC
                        # 1. Set index to the match columns (temporarily)
                        df_base_indexed = df_base.set_index(match_cols)
                        df_replace_indexed = df_replace.set_index(match_cols)
                        
                        # 2. Filter replace file to only relevant columns
                        df_replace_indexed = df_replace_indexed[value_cols]
                        
                        # 3. Perform update
                        # This updates values in Base where indices match Replace
                        df_base_indexed.update(df_replace_indexed)
                        
                        # 4. Reset index back to normal
                        df_base = df_base_indexed.reset_index()
                        
                        st.success(f"✅ Updated values for matches on {match_cols}")
                        with st.expander("View Updated Data"):
                            st.dataframe(df_base.tail(10))
                else:
                    st.warning("Please select at least one ID column and one Value column.")

            except Exception as e:
                st.error(f"Error updating file: {e}")

        # --- STEP C: DOWNLOAD ---
        st.divider()
        st.subheader("4. Download Result")
        
        final_count = len(df_base)
        st.caption(f"Final Dataset: {final_count} rows")
        
        if file_type == "csv":
            data = df_base.to_csv(index=False).encode('utf-8')
            fname = "processed_data.csv"
            mime = "text/csv"
        else:
            buffer = io.BytesIO()
            # Ensure xlsxwriter is installed: pip install xlsxwriter
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_base.to_excel(writer, index=False)
            data = buffer.getvalue()
            fname = "processed_data.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(
            label="📥 Download Final File",
            data=data,
            file_name=fname,
            mime=mime,
            type="primary"
        )

    except Exception as e:
        st.error(f"Error loading base file: {e}")