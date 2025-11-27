import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
import os
import io
import re

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AI File Editor", page_icon="✏️", layout="wide")

# --- HIDE SIDEBAR CSS ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarCollapsedControl"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("📂 Auto-Detect File Editor (Cascading Filters)")

# 2. HELPER: RESCUE FUNCTION
def rescue_dataframe(text_data):
    try:
        text_data = re.sub(r'\[\d+ rows x \d+ columns\]', '', text_data).strip()
        return pd.read_csv(io.StringIO(text_data), sep=r'\s{2,}', engine='python')
    except:
        try:
             return pd.read_csv(io.StringIO(text_data), sep=r'\s+', engine='python')
        except:
            return None

# 3. SESSION STATE SETUP
if "result_df" not in st.session_state:
    st.session_state.result_df = None

# 4. API KEY INPUT
api_key = "AIzaSyDJdQYb7fIGo973cJjwsXbcxVMDB6C-jAE"  # <--- PASTE YOUR REAL KEY HERE

# 5. FILE UPLOADER
uploaded_file = st.file_uploader("Upload your data", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # --- A. AUTO-DETECT FILE TYPE ---
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
            file_type = "csv"
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
            file_type = "excel"
        else:
            st.error("Unsupported file type!")
            st.stop()
            
        st.success(f"✅ Loaded {file_type.upper()} file successfully!")

        #unique countries
        col = df.columns[df.columns.str.contains("country", case=False)][0]
        n_unique_countries = df[col].nunique()

        # --- B. FULL DATA PREVIEW ---
        st.markdown(f"### 1️⃣ Original Data (Shape: {df.shape[0]} rows × {df.shape[1]} cols)")
        with st.expander("View Full Data", expanded=True):
            st.dataframe(df, use_container_width=True)

        # --- C. CASCADING FILTER BUILDER (THE FIX) ---
        st.divider()
        col_filter, col_action = st.columns([1, 1])
        
        with col_filter:
            st.markdown("### 2️⃣ Cascading Filters")
            st.info("Select columns in order. Each dropdown filters the next one.")
            
            all_columns = df.columns.tolist()
            # User selects which columns they want to filter by (Order matters!)
            selected_filter_cols = st.multiselect("Select Filter Path (e.g. Brand -> Nameplate -> Program):", all_columns)
            
            # --- CASCADING LOGIC ---
            df_filtered = df.copy() # Start with full data
            filter_conditions = {}
            
            if selected_filter_cols:
                for col in selected_filter_cols:
                    # Get unique values ONLY from the currently filtered data
                    available_values = sorted(df_filtered[col].astype(str).unique())
                    
                    # Create dropdown
                    selected_val = st.selectbox(f"Select {col}:", available_values, key=f"casc_{col}")
                    
                    # Filter the dataframe immediately for the next loop iteration
                    df_filtered = df_filtered[df_filtered[col].astype(str) == selected_val]
                    
                    # Store condition
                    filter_conditions[col] = selected_val
                
                st.caption(f"ℹ️ Current Selection matches **{len(df_filtered)} rows**.")
                if len(df_filtered) > n_unique_countries:
                    st.warning(f"⚠️ Warning: You have selected more than {n_unique_countries} rows. Creating a variant might create duplicates if you don't filter further.")
            
            # Construct WHERE clause
            if filter_conditions:
                conditions_str = " and ".join([f"{k} is '{v}'" for k, v in filter_conditions.items()])
            else:
                conditions_str = "(No filter selected - Apply to ALL rows)"

        # --- D. CHOOSE ACTION ---
        with col_action:
            st.markdown("### 3️⃣ Choose Action")
            action_type = st.radio("Action Type:", ["➕ Add New Variant", "✏️ Replace Value"], horizontal=True)
            
            # Column Selector
            default_ix = 0
            for i, c in enumerate(all_columns):
                if "power" in c.lower() or "kw" in c.lower():
                    default_ix = i
                    break
            target_col = st.selectbox("Column to Change:", all_columns, index=default_ix)
            
            # Value Selector
            existing_values = sorted([str(x) for x in df[target_col].dropna().unique().tolist()])
            value_options = ["(Type Custom Value...)"] + existing_values
            
            selected_val_opt = st.selectbox(f"Select New Value for '{target_col}':", value_options)
            
            if selected_val_opt == "(Type Custom Value...)":
                final_value = st.text_input(f"Type custom value for '{target_col}':")
            else:
                final_value = selected_val_opt
                
            # DEDUPLICATION OPTION
            dedup_instruction = ""
            if action_type == "➕ Add New Variant":
                dedup = st.checkbox("Ensure unique row per Country? (Prevents duplicates)", value=True)
                if dedup:
                    dedup_instruction = "IMPORTANT: Before copying, group the source rows by 'Sales Country' (or 'Country') and pick only ONE row per country to copy. This ensures we do not create duplicate variants."

        # --- PROMPT GENERATION ---
        if action_type == "➕ Add New Variant":
            base_prompt = f"Find rows where {conditions_str}. {dedup_instruction} COPY these rows, change '{target_col}' to '{final_value}', and APPEND them as new rows."
        else:
            base_prompt = f"Find rows where {conditions_str}. UPDATE these rows by setting '{target_col}' to '{final_value}'."

        # --- E. SETUP AI ---
        llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=api_key)
        sdf = SmartDataframe(df, config={"llm": llm, "conversational": False})

        # --- F. RUN INTERFACE ---
        st.divider()
        st.markdown("### 4️⃣ Review & Run")
        query = st.text_area("AI Instructions:", value=base_prompt, height=100)
        
        if st.button("🚀 Run AI Edits", type="primary"):
            with st.spinner("AI is processing..."):
                try:
                    temp_file = "temp_ai_output.csv"
                    strict_query = f"""
                    {query}
                    STRICT INSTRUCTIONS:
                    1. Perform the operation on the dataframe 'df'.
                    2. Save the result to a CSV file named '{temp_file}' using: df.to_csv('{temp_file}', index=False)
                    3. Do NOT return the dataframe text.
                    4. The last line of your code must be: print("SAVED")
                    """
                    
                    sdf.chat(strict_query)
                    
                    if os.path.exists(temp_file):
                        result_df = pd.read_csv(temp_file)
                        st.session_state.result_df = result_df
                        st.success("Processing Complete!")
                        os.remove(temp_file)
                        st.rerun()
                    else:
                        st.error("The AI claimed it finished, but no output file was created.")

                except Exception as e:
                    st.error(f"AI Error: {e}")

        # --- G. RESULT & DOWNLOAD ---
        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df
            
            st.divider()
            st.markdown(f"### 5️⃣ Result (Effective Changes)")
            
            try:
                original_len = len(df)
                result_len = len(result_df)
                
                added_rows = pd.DataFrame()
                if result_len > original_len:
                    added_rows = result_df.iloc[original_len:].copy()
                    added_rows.insert(0, "Status", "🆕 Added")

                min_len = min(original_len, result_len)
                df_orig_slice = df.iloc[:min_len].reset_index(drop=True)
                df_res_slice = result_df.iloc[:min_len].reset_index(drop=True)
                
                common_cols = df_orig_slice.columns.intersection(df_res_slice.columns)
                mask_diff = (df_orig_slice[common_cols].fillna("##") != df_res_slice[common_cols].fillna("##")).any(axis=1)
                
                modified_rows = result_df.iloc[:min_len][mask_diff].copy()
                if not modified_rows.empty:
                    modified_rows.insert(0, "Status", "✏️ Modified")

                effective_changes = pd.concat([modified_rows, added_rows])

                if not effective_changes.empty:
                    st.success(f"✅ Found {len(effective_changes)} effective changes.")
                    st.dataframe(effective_changes, use_container_width=True)
                else:
                    st.warning("No effective changes detected.")

            except Exception as e:
                st.error(f"Diff Error: {e}")
                st.dataframe(result_df.tail())

            if file_type == "csv":
                data = result_df.to_csv(index=False).encode('utf-8')
                mime = "text/csv"
                fname = "updated_data.csv"
            else:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False)
                data = buffer.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                fname = "updated_data.xlsx"

            st.download_button(
                label="📥 Download Final File",
                data=data,
                file_name=fname,
                mime=mime,
                type="primary"
            )

    except Exception as e:
        st.error(f"File Error: {e}")
