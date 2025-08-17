# AI Data Analysis Agent

This is an AI data analysis Agent built using the Agno Agent framework and Openai's gpt-4o model. This agent helps users analyze their data - csv, excel files through natural language queries, powered by OpenAI's language models and DuckDB for efficient data processing - making data analysis accessible to users regardless of their SQL expertise.

## Features
1. Upload CSV/ Excel files
2. Automatic data type detection and schema inference
3. Perform complex data aggregations
4. Filter & sort data
5. Generate summaries & data visualizations


## How to Run

1. **Setup Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/yamanalrifaii/AI-Data-Analyst.git
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   - Get OpenAI API key from [OpenAI Platform](https://platform.openai.com)

3. **Run the Application**
   ```bash
   streamlit run ai_data_analyst.py
   ```

## Usage

1. Launch the application using the command above
2. Provide your OpenAI API key in the sidebar of Streamlit
3. Upload your CSV or Excel file through the Streamlit interface
4. Ask questions about your data in natural language
5. View the results and generated visualizations
