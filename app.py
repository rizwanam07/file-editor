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

st.title("📂 AI File Editor (Multi-Select Filters)")

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

# 3. SESSION STATE MANAGEMENT
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "history" not in st.session_state:
    st.session_state.history = []

# 4. API KEY SETUP (SECURE)
# The app will look for the key in Streamlit's encrypted secrets.
# If it's not found (e.g., on GitHub), it will stop and ask for it.
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 API Key missing! Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# 5. BUTTON CALLBACKS
def clear_all():
    for key in list(st.session_state.keys()):
        if key.startswith("casc_") or key in ["result_df", "history"]:
            del st.session_state[key]
    st.session_state.result_df = None
    st.session_state.history = []

def undo_last():
    if st.session_state.history:
        st.session_state.result_df = st.session_state.history.pop()
        st.toast("Undo Successful! Went back 1 step.", icon="↩️")
    else:
        st.warning("Nothing to undo!")

# 6. FILE UPLOADER
uploaded_file = st.file_uploader("Upload your data", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # --- A. LOAD FILE (CHAINING LOGIC) ---
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if file_extension == '.csv':
            df_original = pd.read_csv(uploaded_file)
            file_type = "csv"
        elif file_extension in ['.xlsx', '.xls']:
            df_original = pd.read_excel(uploaded_file)
            file_type = "excel"
        else:
            st.error("Unsupported file type!")
            st.stop()
        
        # Determine Current DataFrame (Editing the latest result)
        if st.session_state.result_df is not None:
            current_df = st.session_state.result_df
            st.info(f"📍 Editing: **Step {len(st.session_state.history) + 1}** (Result of previous changes)")
        else:
            current_df = df_original
            st.info("📍 Editing: **Original File**")

        # --- B. CASCADING FILTER BUILDER (MULTI-SELECT) ---
        st.divider()
        col_filter, col_action = st.columns([1, 1])
        
        df_filtered = current_df.copy()
        
        with col_filter:
            st.markdown("### 1️⃣ Filter Data")
            st.caption("Select multiple values to target specific rows.")
            
            all_columns = current_df.columns.tolist()
            selected_filter_cols = st.multiselect("Select Filter Path:", all_columns)
            
            filter_conditions = {}
            if selected_filter_cols:
                for col in selected_filter_cols:
                    # Get values available in current filtered subset
                    available_values = sorted(df_filtered[col].astype(str).unique())
                    
                    # MULTI-SELECT WIDGET
                    selected_vals = st.multiselect(f"Select {col}:", available_values, key=f"casc_{col}")
                    
                    # Update Filter Logic
                    if selected_vals:
                        # Filter dataframe to keep rows where column value matches ANY of the selected values
                        df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected_vals)]
                        filter_conditions[col] = selected_vals
                    else:
                        # If nothing selected, we assume "All" for this step, but don't filter 'df_filtered' further
                        pass
            
            # Construct Prompt String (handling lists)
            if filter_conditions:
                # Creates string like: "Brand in ['Audi', 'BMW'] and Country in ['France']"
                conditions_str = " and ".join([f"{k} in {v}" for k, v in filter_conditions.items()])
            else:
                conditions_str = "(No filter selected - Apply to ALL rows)"

        # --- C. LIVE PREVIEW (TARGETED ROWS) ---
        # Show exactly what rows are currently targeted
        with st.expander(f"🔎 Targeted Data Preview: {len(df_filtered)} rows selected", expanded=True):
            st.dataframe(df_filtered, use_container_width=True)

        # --- D. CHOOSE ACTION ---
        with col_action:
            st.markdown("### 2️⃣ Choose Action")
            action_type = st.radio("Action Type:", ["➕ Add New Variant", "✏️ Replace Value"], horizontal=True)
            
            country_col = next((c for c in current_df.columns if "country" in c.lower() or "region" in c.lower()), None)
            
            # Target Column
            default_ix = 0
            for i, c in enumerate(all_columns):
                if "power" in c.lower() or "kw" in c.lower():
                    default_ix = i
                    break
            target_col = st.selectbox("Column to Change:", all_columns, index=default_ix)
            
            # Value Selector
            if not df_filtered.empty:
                relevant_values = sorted([str(x) for x in df_filtered[target_col].dropna().unique().tolist()])
            else:
                relevant_values = []
                
            value_options = ["(Type Custom Value...)"] + relevant_values
            selected_val_opt = st.selectbox(f"Select New Value for '{target_col}':", value_options)
            
            if selected_val_opt == "(Type Custom Value...)":
                final_value = st.text_input(f"Type custom value for '{target_col}':")
            else:
                final_value = selected_val_opt

            # DEDUPLICATION / COUNTRY LOGIC
            dedup_instruction = ""
            if action_type == "➕ Add New Variant":
                if country_col and not df_filtered.empty:
                    unique_countries = df_filtered[country_col].unique().tolist()
                    count_countries = len(unique_countries)
                    
                    if count_countries > 0:
                        st.success(f"🌍 Detected **{count_countries} Unique Countries** in selection.")
                        # Strict prompt to handle the multi-selected countries
                        dedup_instruction = (
                            f"IMPORTANT: The source rows cover {count_countries} unique countries: {unique_countries}. "
                            f"You must create exactly {count_countries} new rows (ONE per country). "
                            f"Group the source rows by '{country_col}' and pick one representative row per country to copy."
                        )
                    else:
                        st.warning("Selection is empty.")

        # --- PROMPT GENERATION ---
        if action_type == "➕ Add New Variant":
            base_prompt = f"Find rows where {conditions_str}. {dedup_instruction} COPY these rows, change '{target_col}' to '{final_value}', and APPEND them as new rows."
        else:
            base_prompt = f"Find rows where {conditions_str}. UPDATE these rows by setting '{target_col}' to '{final_value}'."

        # --- E. RUN INTERFACE ---
        st.divider()
        st.markdown("### 3️⃣ Review & Run")
        query = st.text_area("AI Instructions:", value=base_prompt, height=100)
        
        b_col1, b_col2, b_col3 = st.columns([1, 0.5, 3])
        with b_col1:
            run_clicked = st.button("🚀 Run AI Edits", type="primary", use_container_width=True)
        with b_col2:
            undo_disabled = len(st.session_state.history) == 0
            undo_clicked = st.button("↩️ Undo", on_click=undo_last, disabled=undo_disabled, use_container_width=True)
        with b_col3:
            reset_clicked = st.button("🔄 Reset All", on_click=clear_all)

        # --- F. AI EXECUTION ---
        if run_clicked:
            st.session_state.history.append(current_df.copy()) # Save History
            
            llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=api_key)
            sdf = SmartDataframe(current_df, config={"llm": llm, "conversational": False})
            
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
                    5. Changes should be merge with the full file
                    """
                    sdf.chat(strict_query)
                    
                    if os.path.exists(temp_file):
                        result_df = pd.read_csv(temp_file)
                        st.session_state.result_df = result_df
                        st.success("Processing Complete!")
                        os.remove(temp_file)
                        st.rerun()
                    else:
                        st.error("The AI finished, but no output file was created.")
                except Exception as e:
                    st.error(f"AI Error: {e}")

        # --- G. RESULT & DOWNLOAD ---
        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df
            st.divider()
            st.markdown(f"### 4️⃣ Result (Total Rows: {len(result_df)})")
            
            try:
                # Compare vs Original to find effective changes
                original_len = len(df_original)
                result_len = len(result_df)
                
                added_rows = pd.DataFrame()
                if result_len > original_len:
                    added_rows = result_df.iloc[original_len:].copy()
                    added_rows.insert(0, "Status", "🆕 Added")

                min_len = min(original_len, result_len)
                df_orig_slice = df_original.iloc[:min_len].reset_index(drop=True)
                df_res_slice = result_df.iloc[:min_len].reset_index(drop=True)
                
                common_cols = df_orig_slice.columns.intersection(df_res_slice.columns)
                # Robust comparison handling NaNs
                mask_diff = (df_orig_slice[common_cols].fillna("##") != df_res_slice[common_cols].fillna("##")).any(axis=1)
                
                modified_rows = result_df.iloc[:min_len][mask_diff].copy()
                if not modified_rows.empty:
                    modified_rows.insert(0, "Status", "✏️ Modified")

                effective_changes = pd.concat([modified_rows, added_rows])

                if not effective_changes.empty:
                    st.success(f"✅ Found {len(effective_changes)} effective changes (vs Original).")
                    st.dataframe(effective_changes, use_container_width=True)
                else:
                    st.warning("No effective changes detected vs Original.")

            except Exception as e:
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
