import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from phi.agent.duckdb import DuckDbAgent
from phi.model.openai.chat import OpenAIChat
import re
import os

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns with better error handling
        for col in df.columns:
            col_lower = col.lower()
            if any(date_keyword in col_lower for date_keyword in ['date', 'time', 'created', 'updated', 'release']):
                try:
                    # Try to parse as datetime with more flexible parsing
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    # Convert to string format that's more compatible with Streamlit
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    st.warning(f"Could not parse column '{col}' as datetime: {e}")
                    # Keep as string if datetime parsing fails
                    df[col] = df[col].astype(str)
            elif df[col].dtype == 'object':
                try:
                    # Try to convert to numeric, but be more careful
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    # Only convert if more than 80% of values are successfully converted
                    if numeric_values.notna().sum() / len(df) > 0.8:
                        df[col] = numeric_values
                    else:
                        df[col] = df[col].astype(str)
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    df[col] = df[col].astype(str)
        
        # Clean up any problematic characters that might cause serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('\x00', '', regex=False)
                df[col] = df[col].str.replace('\n', ' ', regex=False)
                df[col] = df[col].str.replace('\r', ' ', regex=False)
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

st.title("📊 Data Analyst Agent")

with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        if openai_key.startswith("sk-") and len(openai_key) >= 50:
            st.session_state.openai_key = openai_key
            st.success("API key saved!")
        else:
            st.error("Invalid API key format. OpenAI keys should start with 'sk-' and be at least 50 characters long.")
            st.session_state.openai_key = None
    else:
        st.warning("Please enter your OpenAI API key to proceed.")


uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        
        # Safe dataframe display with error handling
        try:
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display interactive dataframe: {e}")
            st.write("Displaying data as static table instead...")
            try:
                # Try to display as a simple table
                st.table(df.head(100))  # Limit to first 100 rows for performance
                if len(df) > 100:
                    st.info(f"Showing first 100 rows out of {len(df)} total rows")
            except Exception as e2:
                st.error(f"Could not display data table: {e2}")
                st.write("Data loaded successfully but display failed. You can still use the AI agent to query your data.")
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Data validation and type information
        st.subheader("🔍 Data Validation")
        if len(columns) > 0:
            # Show data types and potential issues
            validation_info = []
            for col in columns:
                col_info = {
                    "Column": col,
                    "Data Type": str(df[col].dtype),
                    "Sample Values": str(df[col].head(3).tolist()),
                    "Missing Values": df[col].isnull().sum(),
                    "Unique Values": df[col].nunique()
                }
                validation_info.append(col_info)
            
            validation_df = pd.DataFrame(validation_info)
            st.dataframe(validation_df, use_container_width=True)
            
            # Check for potential issues
            issues = []
            for col in columns:
                if df[col].dtype == 'object':
                    # Check for very long strings that might cause issues
                    max_length = df[col].astype(str).str.len().max()
                    if max_length > 1000:
                        issues.append(f"Column '{col}' has very long strings (max: {max_length} chars)")
                    
                    # Check for special characters
                    if df[col].astype(str).str.contains(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', regex=True).any():
                        issues.append(f"Column '{col}' contains control characters")
            
            if issues:
                st.warning("⚠️ Potential data issues detected:")
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ Data validation passed - no obvious issues detected")
        
        # Data Visualization Section
        st.header("📊 Data Visualization")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap", "Pie Chart", "Area Chart"]
        )
        
        # Column selection for visualization
        if len(columns) > 0:
            # For single column charts
            if chart_type in ["Histogram", "Box Plot"]:
                selected_column = st.selectbox("Select Column:", columns)
                if selected_column:
                    if chart_type == "Histogram":
                        fig = px.histogram(df, x=selected_column, title=f"Histogram of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Box Plot":
                        fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # For two column charts
            elif chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("Select X-axis Column:", columns, key="x_col")
                with col2:
                    y_column = st.selectbox("Select Y-axis Column:", columns, key="y_col")
                
                if x_column and y_column:
                    if chart_type == "Bar Chart":
                        fig = px.bar(df, x=x_column, y=y_column, title=f"Bar Chart: {y_column} vs {x_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Line Chart":
                        fig = px.line(df, x=x_column, y=y_column, title=f"Line Chart: {y_column} vs {x_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter Plot: {y_column} vs {x_column}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # For correlation heatmap
            elif chart_type == "Heatmap":
                # Select numeric columns for correlation
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_columns) > 1:
                    correlation_matrix = df[numeric_columns].corr()
                    fig = px.imshow(
                        correlation_matrix,
                        title="Correlation Heatmap",
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation heatmap.")
            
            # For pie chart
            elif chart_type == "Pie Chart":
                selected_column = st.selectbox("Select Column for Pie Chart:", columns)
                if selected_column:
                    value_counts = df[selected_column].value_counts()
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart of {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # For area chart
            elif chart_type == "Area Chart":
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("Select X-axis Column:", columns, key="area_x")
                with col2:
                    y_column = st.selectbox("Select Y-axis Column:", columns, key="area_y")
                
                if x_column and y_column:
                    fig = px.area(df, x=x_column, y=y_column, title=f"Area Chart: {y_column} vs {x_column}")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.header("📈 Summary Statistics")
        if len(columns) > 0:
            st.subheader("Basic Statistics")
            st.write(df.describe())
            
            st.subheader("Data Information")
            info_data = {
                "Column": columns,
                "Data Type": [str(df[col].dtype) for col in columns],
                "Non-Null Count": [df[col].count() for col in columns],
                "Missing Values": [df[col].isnull().sum() for col in columns]
            }
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df)
        
        st.header("🎨 Advanced Visualizations")
        
        if len(columns) > 0:
            st.subheader("Multi-Chart Dashboard")
            
            # Create subplots
            if len(columns) >= 2:
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if len(numeric_columns) >= 2:
                    # Create a 2x2 subplot layout
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Distribution", "Correlation", "Box Plot", "Scatter"),
                        specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                               [{"type": "box"}, {"type": "scatter"}]]
                    )
                    
                    # Histogram
                    fig.add_trace(
                        go.Histogram(x=df[numeric_columns[0]], name="Distribution"),
                        row=1, col=1
                    )
                    
                    # Correlation heatmap
                    corr_matrix = df[numeric_columns[:min(5, len(numeric_columns))]].corr()
                    fig.add_trace(
                        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index),
                        row=1, col=2
                    )
                    
                    # Box plot
                    fig.add_trace(
                        go.Box(y=df[numeric_columns[0]], name="Box Plot"),
                        row=2, col=1
                    )
                    
                    # Scatter plot
                    fig.add_trace(
                        go.Scatter(x=df[numeric_columns[0]], y=df[numeric_columns[1]], mode='markers', name="Scatter"),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, title_text="Data Analysis Dashboard")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for advanced visualizations.")
            
            st.subheader("Custom Chart Options")
            
            # Color scheme selection
            color_scheme = st.selectbox(
                "Select Color Scheme:",
                ["plotly", "plotly_dark", "ggplot2", "seaborn", "viridis", "plasma", "inferno"]
            )
            
            # Chart customization
            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"] and 'x_column' in locals() and 'y_column' in locals():
                if st.checkbox("Customize Chart"):
                    # Chart size
                    chart_height = st.slider("Chart Height", 400, 800, 500)
                    
                    # Customize based on chart type
                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            df, x=x_column, y=y_column, 
                            title=f"Bar Chart: {y_column} vs {x_column}",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            height=chart_height
                        )
                    elif chart_type == "Line Chart":
                        fig = px.line(
                            df, x=x_column, y=y_column, 
                            title=f"Line Chart: {y_column} vs {x_column}",
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            height=chart_height
                        )
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(
                            df, x=x_column, y=y_column, 
                            title=f"Scatter Plot: {y_column} vs {x_column}",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            height=chart_height
                        )
                    
                    # Update layout
                    fig.update_layout(
                        template=color_scheme,
                        showlegend=True,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }
        
        st.write("Debug - Semantic Model:", json.dumps(semantic_model, indent=2))
        st.write("Debug - Temp file path:", temp_path)
        st.write("Debug - File exists:", os.path.exists(temp_path))
        
        # Initialize the DuckDbAgent for SQL query generation
        try:
            duckdb_agent = DuckDbAgent(
                model=OpenAIChat(
                    model="gpt-4",
                    api_key=st.session_state.openai_key
                ),
                semantic_model=json.dumps(semantic_model),
                markdown=True,
                add_history_to_messages=False,  # Disable chat history
                followups=False,  # Disable follow-up queries
                read_tool_call_history=False,  # Disable reading tool call history
                system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
            )
            st.success("DuckDbAgent initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing DuckDbAgent: {e}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            st.stop()
        
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None
        
        # Main query input 
        user_query = st.text_area("Ask a query about the data:")
        
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    with st.spinner('Processing your query...'):
                        response = duckdb_agent.run(user_query)

                        if hasattr(response, 'content'):
                            response_content = response.content
                        else:
                            response_content = str(response)

                    st.markdown(response_content)
                
                    
                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error(f"Error type: {type(e).__name__}")
                    st.error(f"Full error details: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")
