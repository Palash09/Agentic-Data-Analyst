import streamlit as st
import tempfile
import os
import shutil
import json
import pandas as pd
from agents.agent_runner import get_agent_choices, run_agent
from agents.cache_manager import cache_manager, performance_monitor
from agents.data_preprocessor import preprocess_for_analysis

def get_chart_explanation(chart_name: str, chart_index: int, is_domain_insights: bool = False) -> str:
    """Generate explanations for charts based on their type and content"""
    
    # Common chart explanations based on filename patterns
    chart_explanations = {
        'correlation_heatmap': """
        **Correlation Heatmap Analysis:**
        
        This heatmap shows the strength and direction of relationships between different variables in your dataset.
        
        **How to read it:**
        - **Red colors** indicate positive correlations (variables increase together)
        - **Blue colors** indicate negative correlations (variables move in opposite directions)
        - **Darker colors** indicate stronger correlations
        - **Lighter colors** indicate weaker correlations
        
        **Key insights to look for:**
        - Strong correlations (>0.7) may indicate redundant variables
        - Moderate correlations (0.3-0.7) show meaningful relationships
        - Weak correlations (<0.3) suggest independent variables
        
        **Business implications:**
        - Identify which factors influence each other most strongly
        - Avoid multicollinearity in predictive models
        - Focus on variables with strong relationships to your target
        """,
        
        'cluster_analysis': """
        **Cluster Analysis Visualization:**
        
        This chart shows how your data points group together based on their similarities across multiple variables.
        
        **What it reveals:**
        - **Data Segmentation**: Natural groupings in your data
        - **Pattern Recognition**: Hidden structures and relationships
        - **Outlier Detection**: Points that don't fit typical patterns
        
        **Business implications:**
        - Customer segmentation for targeted marketing
        - Product grouping for inventory management
        - Risk assessment and anomaly detection
        - Resource allocation based on group characteristics
        """,
        
        'anomaly_detection': """
        **Anomaly Detection Results:**
        
        This visualization identifies unusual data points that deviate significantly from the normal patterns.
        
        **Key findings:**
        - **Red points**: Potential anomalies or outliers
        - **Blue points**: Normal data points
        - **Patterns**: Clusters of normal behavior
        
        **Why this matters:**
        - Fraud detection in financial data
        - Quality control in manufacturing
        - Customer behavior analysis
        - System performance monitoring
        - Risk identification and mitigation
        """,
        
        'age_distribution': """
        **Age Distribution Analysis:**
        
        This chart shows the spread and concentration of age values across your dataset.
        
        **Insights to look for:**
        - **Peak ages**: Most common age groups
        - **Distribution shape**: Normal, skewed, or bimodal
        - **Age ranges**: Young, middle-aged, or senior populations
        - **Outliers**: Unusually young or old individuals
        
        **Business applications:**
        - Target audience identification
        - Product development for specific age groups
        - Marketing strategy optimization
        - Service customization by age segment
        """,
        
        'gender_distribution': """
        **Gender Distribution Analysis:**
        
        This visualization shows the balance and representation of different genders in your dataset.
        
        **Key observations:**
        - **Gender balance**: Equal or skewed representation
        - **Sample size**: Adequate representation for analysis
        - **Demographic insights**: Population characteristics
        
        **Strategic implications:**
        - Gender-specific marketing campaigns
        - Product design considerations
        - Diversity and inclusion analysis
        - Market penetration strategies
        """,
        
        'investment_preferences': """
        **Investment Preferences Analysis:**
        
        This chart reveals patterns in investment choices and financial behavior.
        
        **What to analyze:**
        - **Popular investments**: Most preferred options
        - **Risk tolerance**: Conservative vs. aggressive choices
        - **Demographic patterns**: Age/gender-based preferences
        - **Market trends**: Current investment behaviors
        
        **Business opportunities:**
        - Product development for underserved segments
        - Marketing strategies for different risk profiles
        - Financial planning recommendations
        - Portfolio optimization strategies
        """,
        
        'savings_objectives': """
        **Savings Objectives Analysis:**
        
        This chart shows what people are saving for and their financial goals.
        
        **Key insights:**
        - **Primary goals**: Most common savings objectives
        - **Goal distribution**: How savings are prioritized
        - **Demographic patterns**: Age/gender-based goals
        - **Market opportunities**: Underserved objectives
        
        **Business value:**
        - Product development for specific goals
        - Marketing messaging alignment
        - Financial planning services
        - Customer segmentation strategies
        """,
        
        'monitoring_frequency': """
        **Investment Monitoring Frequency:**
        
        This chart shows how often people check and manage their investments.
        
        **Behavioral insights:**
        - **Engagement levels**: Active vs. passive investors
        - **Monitoring patterns**: Daily, weekly, monthly habits
        - **Risk tolerance indicators**: Frequent monitoring may indicate higher risk tolerance
        - **Service opportunities**: Automated vs. manual management preferences
        
        **Strategic applications:**
        - App feature development
        - Notification frequency optimization
        - Customer service planning
        - Product recommendation timing
        """
    }
    
    # Domain-specific chart explanations for ML-based charts
    domain_chart_explanations = {
        'feature_importance': """
        **Feature Importance Analysis:**
        
        This chart shows which variables have the strongest predictive power in your dataset.
        
        **What it reveals:**
        - **Key predictors**: Variables that most influence outcomes
        - **Feature ranking**: Relative importance of each variable
        - **Model insights**: Understanding of data relationships
        
        **Business value:**
        - Focus resources on most important factors
        - Simplify models by removing less important features
        - Understand what drives your key metrics
        - Optimize data collection efforts
        """,
        
        'trend_analysis': """
        **Predictive Trend Analysis:**
        
        This visualization shows patterns and trends in your data over time or across values.
        
        **Key insights:**
        - **Trend direction**: Increasing, decreasing, or stable patterns
        - **Trend strength**: How consistent the pattern is
        - **Predictive signals**: Future behavior indicators
        - **Seasonal patterns**: Cyclical variations
        
        **Strategic applications:**
        - Forecasting future trends
        - Identifying seasonal patterns
        - Planning based on historical patterns
        - Resource allocation timing
        """,
        
        'clustering_analysis': """
        **Predictive Clustering Analysis:**
        
        This chart uses machine learning to group similar data points together.
        
        **What it shows:**
        - **Natural segments**: Data-driven groupings
        - **Pattern recognition**: Hidden structures in data
        - **Predictive insights**: Future behavior predictions
        - **Segment characteristics**: Unique traits of each group
        
        **Business implications:**
        - Customer segmentation strategies
        - Product recommendation systems
        - Risk assessment and targeting
        - Personalized marketing approaches
        """,
        
        'outlier_detection': """
        **Predictive Outlier Detection:**
        
        This analysis identifies unusual data points using machine learning algorithms.
        
        **Key findings:**
        - **Anomalies**: Data points that don't follow normal patterns
        - **Risk indicators**: Potential issues or opportunities
        - **Data quality**: Validation of data integrity
        - **Pattern boundaries**: Limits of normal behavior
        
        **Why this matters:**
        - Fraud detection and prevention
        - Quality control and monitoring
        - Opportunity identification
        - Risk management
        - Data validation and cleaning
        """,
        
        'pattern_analysis': """
        **Predictive Pattern Analysis:**
        
        This chart reveals relationships between categorical and numerical variables.
        
        **What it reveals:**
        - **Category performance**: How different groups compare
        - **Pattern strength**: Consistency of relationships
        - **Predictive factors**: Variables that influence outcomes
        - **Group differences**: Statistical significance of variations
        
        **Business applications:**
        - Targeted marketing strategies
        - Performance optimization
        - Resource allocation decisions
        - Product development priorities
        """
    }
    
    # Try to match chart name to explanations
    chart_name_lower = chart_name.lower()
    
    # Use domain-specific explanations for domain insights
    if is_domain_insights:
        for key, explanation in domain_chart_explanations.items():
            if key in chart_name_lower:
                return explanation
    
    # Use regular explanations for statistical analysis
    for key, explanation in chart_explanations.items():
        if key in chart_name_lower:
            return explanation
    
    # Default explanation for unmatched charts
    return f"""
    **Chart Analysis: {chart_name}**
    
    This visualization provides insights into your dataset through {chart_name.lower()}.
    
    **What to look for:**
    - Patterns and trends in the data
    - Relationships between variables
    - Outliers or unusual data points
    - Distribution characteristics
    
    **Business implications:**
    - Identify key insights for decision making
    - Understand data patterns and relationships
    - Spot opportunities or risks in the data
    - Guide strategic planning and optimization
    """

def format_insights_as_bullets(text: str) -> str:
    """Formats a string of insights into a bullet point list with AI-powered sentence detection."""
    if not text:
        return ""
    
    import re
    
    # Clean the text first
    text = text.strip()
    
    # Split by multiple sentence endings and handle various formats
    # This is more sophisticated than simple regex splitting
    sentences = []
    
    # First, split by common sentence endings with proper handling
    # Look for patterns like: . ! ? followed by space and capital letter
    sentence_patterns = [
        r'[.!?]\s+[A-Z]',  # Standard sentence endings
        r'[.!?]\s*\n\s*[A-Z]',  # Sentence endings with line breaks
        r'[.!?]\s*[A-Z]',  # Sentence endings without space
    ]
    
    # Start with the full text
    remaining_text = text
    
    # Find all sentence boundaries
    boundaries = []
    for pattern in sentence_patterns:
        matches = list(re.finditer(pattern, remaining_text))
        for match in matches:
            boundaries.append(match.start() + 1)  # +1 to include the punctuation
    
    # Sort boundaries and add start/end
    boundaries = sorted(list(set(boundaries)))
    boundaries = [0] + boundaries + [len(remaining_text)]
    
    # Extract sentences
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        sentence = remaining_text[start:end].strip()
        
        # Clean up the sentence
        sentence = re.sub(r'^\s*[-•*]\s*', '', sentence)  # Remove existing bullets
        sentence = sentence.strip()
        
        if sentence and len(sentence) > 10:  # Only include meaningful sentences
            sentences.append(sentence)
    
    # If no sentences were found with the pattern, fall back to simple splitting
    if not sentences:
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    # Format as bullet points
    bullet_lines = []
    for sentence in sentences:
        if sentence:
            # Remove any existing bullet points and add our own
            sentence = re.sub(r'^[-•*]\s*', '', sentence)
            sentence = sentence.strip()
            if sentence:
                bullet_lines.append(f"• {sentence}")
    
    return "\n".join(bullet_lines)

st.set_page_config(
    page_title="Agentic Data Assistant", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .main-description {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Agentic Data Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-description">Upload your CSV file, select an analysis agent, and get deep statistical insights with automated visualizations!</p>', unsafe_allow_html=True)

# Sidebar for additional features
with st.sidebar:
    st.header("🔧 Quick Actions")
    
    # Cache management
    if st.button("🗑️ Clear Cache"):
        cache_manager.clear_cache()
        st.success("Cache cleared!")
    
    st.divider()
    
    # Help section
    st.header("❓ Help")
    st.markdown("""
    **How to use:**
    1. Upload your CSV file
    2. Click "Dataset Information" to learn about your data
    3. Choose an analysis agent
    4. Click "Analyze Data" to get insights
    
    **Tips:**
    - Use "Statistical Analysis" for advanced insights
    - Check data quality before analysis
    - Results are cached for faster repeat analysis
    """)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'preprocessing_report' not in st.session_state:
    st.session_state.preprocessing_report = None
if 'show_data_quality' not in st.session_state:
    st.session_state.show_data_quality = False
if 'show_dataset_info' not in st.session_state:
    st.session_state.show_dataset_info = False

# --- File upload section ---
st.markdown("### 📁 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type=['csv'],
    help="Upload a CSV file to analyze. The file should contain structured data with headers."
)

# Store uploaded file in session state
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# --- Dataset Information Buttons (Fixed Position) ---
if uploaded_file:
    st.markdown("### 📊 Dataset Information")
    
    # Create a container for buttons to prevent shifting
    button_container = st.container()
    with button_container:
        # Use equal width columns for buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("📊 Dataset Details", use_container_width=True, key="dataset_details_btn"):
                st.session_state.show_dataset_info = True
                st.session_state.show_data_quality = False
                st.rerun()
        
        with col2:
            if st.button("🔍 Data Quality", use_container_width=True, key="data_quality_btn"):
                st.session_state.show_data_quality = True
                st.session_state.show_dataset_info = False
                st.rerun()
    
    # Show data quality report or dataset details if requested
    if st.session_state.get('show_data_quality', False) or st.session_state.get('show_dataset_info', False):
        # Add close button at the rightmost border
        col1, col2 = st.columns([20, 1])
        with col2:
            if st.button("✕", help="Close this section", key="close_section_btn"):
                st.session_state.show_data_quality = False
                st.session_state.show_dataset_info = False
                st.rerun()
        
        # Show content in the same expandable section
        section_title = "📊 Dataset Details" if st.session_state.get('show_dataset_info', False) else "🔍 Data Quality Report"
        with st.expander(section_title, expanded=True):
            # Show dataset info if requested
            if st.session_state.get('show_dataset_info', False):
                try:
                    df_preview = pd.read_csv(uploaded_file)
                    
                    # Display basic info with smaller text
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"<small><strong>Rows:</strong> {len(df_preview):,}</small>", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"<small><strong>Columns:</strong> {len(df_preview.columns):,}</small>", unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"<small><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</small>", unsafe_allow_html=True)
                    
                    # Column types
                    col_types = df_preview.dtypes.value_counts()
                    st.write("**Column Types:**")
                    for dtype, count in col_types.items():
                        st.write(f"• {dtype}: {count}")
                    
                    # Missing data
                    missing_cols = df_preview.isnull().sum()
                    missing_cols = missing_cols[missing_cols > 0]
                    if len(missing_cols) > 0:
                        st.write(f"**Missing Data:** {len(missing_cols)} columns")
                        for col, missing_count in missing_cols.items():
                            st.write(f"  - {col}: {missing_count} missing values")
                    
                    # Sample data
                    st.write("**Sample Data (First 5 rows):**")
                    st.dataframe(df_preview.head(), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            
            # Show data quality report if requested
            elif st.session_state.get('show_data_quality', False):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file.flush()
                        
                        df_cleaned, preprocessing_report = preprocess_for_analysis(tmp_file.name)
                        st.session_state.preprocessing_report = preprocessing_report
                        
                        # Display preprocessing summary with smaller text
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"<small><strong>Original Rows:</strong> {preprocessing_report['original_shape'][0]}</small>", unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f"<small><strong>Final Rows:</strong> {preprocessing_report['final_shape'][0]}</small>", unsafe_allow_html=True)
                        with col_c:
                            reduction = preprocessing_report.get('data_reduction_percentage', 0)
                            st.markdown(f"<small><strong>Data Reduction:</strong> {reduction:.1f}%</small>", unsafe_allow_html=True)
                        
                        # Data quality issues
                        if preprocessing_report.get('data_quality_issues'):
                            st.warning("⚠️ Data Quality Issues Found:")
                            for issue in preprocessing_report['data_quality_issues']:
                                st.write(f"• {issue}")
                        
                        # Recommendations
                        if preprocessing_report.get('recommendations'):
                            st.info("💡 Feature Engineering Suggestions:")
                            for rec in preprocessing_report['recommendations']:
                                st.write(f"• {rec}")
                        
                        os.unlink(tmp_file.name)
                except Exception as e:
                    st.error(f"Error analyzing data quality: {str(e)}")

# --- Agent selection ---
st.markdown("### 🤖 Select Analysis Agent")

# Create agent selection with descriptions
agent_descriptions = {
    '📊 Statistical Analysis (Recommended)': 'Advanced statistical analysis with clustering, anomaly detection, and correlation analysis. Includes data preprocessing and quality assessment.',
    'Primary Insights': 'Structured analysis using LangGraph with domain detection and comprehensive insights.',
    'Domain Insights': 'Domain-specific analysis with specialized prompting for industry insights.',
    'Deep-Dive': 'Advanced reasoning with multiple expert perspectives and visualization planning.'
}

# Agent selection dropdown
selected_agent_name = st.selectbox(
    "Choose an analysis agent:",
    options=list(agent_descriptions.keys()),
    key="agent_selection"
)

# Show description under dropdown with smaller text
selected_description = agent_descriptions.get(selected_agent_name, "")
if selected_description:
    st.markdown(f"<small><em>{selected_description}</em></small>", unsafe_allow_html=True)

# Get the agent key from the name
agent_choices = get_agent_choices()
agent_keys, agent_names = zip(*agent_choices)
selected_agent_key = agent_keys[agent_names.index(selected_agent_name)]

# --- Analyze button ---
analyze_col1, analyze_col2 = st.columns([1, 3])
with analyze_col1:
    run_btn = st.button("🚀 Analyze Data", type="primary", use_container_width=True)
with analyze_col2:
    if st.session_state.results is not None:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.results = None
            st.session_state.current_agent = None
            st.session_state.show_data_quality = False
            st.session_state.show_dataset_info = False
            st.rerun()

# Check if we need to run analysis
should_run = (uploaded_file and run_btn and 
              (st.session_state.results is None or 
               st.session_state.current_agent != selected_agent_key))

if should_run:
    # Create a placeholder for results
    results_placeholder = st.empty()
    
    with st.spinner(f"Running {selected_agent_key} analysis..."):
        # Save uploaded file to a temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, uploaded_file.name)
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                result = run_agent(selected_agent_key, csv_path)
                
                # Store results in session state
                st.session_state.results = result
                st.session_state.current_agent = selected_agent_key
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Tip: Make sure your OpenAI API key is set in the .env file and has sufficient credits.")

# Display results if available
if st.session_state.results is not None:
    result = st.session_state.results
    
    # Display success message with performance info
    success_msg = f"✅ Analysis completed successfully using {result['agent']}!"
    
    # Add cache and performance info if available
    cache_info = result.get('_cache_info', {})
    if cache_info.get('cached'):
        success_msg += " (⚡ From cache)"
    elif cache_info.get('execution_time'):
        success_msg += f" (⏱️ {cache_info['execution_time']}s)"
    
    st.success(success_msg)
    
    # Show preprocessing report if available
    if 'preprocessing_report' in result:
        with st.expander("🔧 Data Preprocessing Summary"):
            prep_report = result['preprocessing_report']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Original Shape", f"{prep_report['original_shape'][0]} × {prep_report['original_shape'][1]}")
            with col_b:
                st.metric("Final Shape", f"{prep_report['final_shape'][0]} × {prep_report['final_shape'][1]}")
            
            if prep_report.get('preprocessing_steps'):
                st.write("**Preprocessing Steps:**")
                for step in prep_report['preprocessing_steps']:
                    st.write(f"• {step['step']}")
            
            if prep_report.get('recommendations'):
                st.write("**Recommendations:**")
                for rec in prep_report['recommendations']:
                    st.write(f"• {rec}")
    
    # --- Display insights in beautified format ---
    # Skip insights section for Tree-of-Thought Dashboard and GPT-4o Domain Insights
    if not result.get('agent', '').startswith('Deep-Dive') and not result.get('agent', '').startswith('Domain Insights'):
        st.subheader("📊 Insights")
        insights = result.get('insights', {})
        charts = result.get('charts', [])
        
        if isinstance(insights, dict):
            # Check if we have any meaningful insights
            has_insights = any([
                insights.get('descriptive') and insights.get('descriptive') != 'Analysis failed',
                insights.get('predictive') and insights.get('predictive') != 'Analysis failed',
                insights.get('domain_related') and insights.get('domain_related') != 'Analysis failed',
                insights.get('raw')
            ])
            
            if has_insights:
                # Create tabs for different insight types
                insight_tabs = st.tabs(["Descriptive", "Predictive", "Domain Related"])
                
                with insight_tabs[0]:
                    if insights.get('descriptive') and insights.get('descriptive') != 'Analysis failed':
                        st.markdown("### Descriptive Analysis")
                        descriptive = insights['descriptive']
                        # Format as bullet points for System Agent
                        if result.get('agent', '').startswith('Primary Insights'):
                            st.markdown(format_insights_as_bullets(descriptive))
                        else:
                            # If it's a JSON string, try to parse and format it nicely
                            if isinstance(descriptive, str) and descriptive.strip().startswith('{'):
                                try:
                                    import json
                                    parsed = json.loads(descriptive)
                                    # Try to extract readable text from common JSON structures
                                    if isinstance(parsed, dict):
                                        text_content = (parsed.get('text', '') or 
                                                      parsed.get('content', '') or 
                                                      parsed.get('summary', '') or 
                                                      parsed.get('analysis', '') or
                                                      str(parsed))
                                        st.write(text_content)
                                    else:
                                        st.write(parsed)
                                except:
                                    st.write(descriptive)
                            else:
                                st.write(descriptive)
                    else:
                        st.info("No descriptive analysis available")
                
                with insight_tabs[1]:
                    if insights.get('predictive') and insights.get('predictive') != 'Analysis failed':
                        st.markdown("### Predictive Analysis")
                        predictive = insights['predictive']
                        # Format as bullet points for System Agent
                        if result.get('agent', '').startswith('Primary Insights'):
                            st.markdown(format_insights_as_bullets(predictive))
                        else:
                            # If it's a JSON string, try to parse and format it nicely
                            if isinstance(predictive, str) and predictive.strip().startswith('{'):
                                try:
                                    import json
                                    parsed = json.loads(predictive)
                                    # Try to extract readable text from common JSON structures
                                    if isinstance(parsed, dict):
                                        text_content = (parsed.get('text', '') or 
                                                     parsed.get('content', '') or 
                                                     parsed.get('summary', '') or 
                                                     parsed.get('analysis', '') or
                                                     str(parsed))
                                        st.write(text_content)
                                    else:
                                        st.write(predictive)
                                except:
                                    st.write(predictive)
                            else:
                                st.write(predictive)
                    else:
                        st.info("No predictive analysis available")
                
                with insight_tabs[2]:
                    if insights.get('domain_related') and insights.get('domain_related') != 'Analysis failed':
                        st.markdown("### Domain-Related Analysis")
                        domain_related = insights['domain_related']
                        # Format as bullet points for System Agent
                        if result.get('agent', '').startswith('Primary Insights'):
                            st.markdown(format_insights_as_bullets(domain_related))
                        else:
                            # If it's a JSON string, try to parse and format it nicely
                            if isinstance(domain_related, str) and domain_related.strip().startswith('{'):
                                try:
                                    import json
                                    parsed = json.loads(domain_related)
                                    # Try to extract readable text from common JSON structures
                                    if isinstance(parsed, dict):
                                        text_content = (parsed.get('text', '') or 
                                                     parsed.get('content', '') or 
                                                     parsed.get('summary', '') or 
                                                     parsed.get('analysis', '') or
                                                     str(parsed))
                                        st.write(text_content)
                                    else:
                                        st.write(domain_related)
                                except:
                                    st.write(domain_related)
                            else:
                                st.write(domain_related)
                    else:
                        st.info("No domain-related analysis available")
            else:
                # If no insights but we have charts, just show a simple message
                charts = result.get('charts', [])
                if charts:
                    pass  # Don't show any message, just display charts
                else:
                    st.warning("⚠️ Analysis completed but no insights were generated. This might be due to data format or analysis limitations.")
        
        elif isinstance(insights, str):
            st.markdown(insights)
        else:
            st.write(insights)
    
    # --- Display charts ---
    charts = result.get('charts', [])
    if charts:
        st.subheader("📈 Generated Charts")
        
        # Check if this is from statistical analysis for enhanced explanations
        is_statistical_analysis = (result.get('agent', '').startswith('📊 Statistical Analysis') or 
                                 'Statistical Analysis' in result.get('agent', '') or
                                 result.get('agent', '').startswith('Enhanced Analysis Agent'))
        is_domain_insights = result.get('agent', '').startswith('Domain Insights')
        
        # Create columns for charts
        cols = st.columns(min(2, len(charts)))
        
        for i, chart_item in enumerate(charts):
            # Handle both string paths and dictionary objects
            if isinstance(chart_item, dict):
                chart_path = chart_item.get('path', '')
                chart_name = chart_item.get('name', f'Chart {i+1}')
            else:
                chart_path = chart_item
                chart_name = os.path.basename(chart_path) if chart_path else f'Chart {i+1}'
            
            if chart_path and os.path.exists(chart_path):
                col_idx = i % 2
                with cols[col_idx]:
                    st.markdown(f"**Chart {i+1}:** {chart_name}")
                    st.image(chart_path, use_column_width=True)
                    
                    # Add chart explanation for statistical analysis or domain insights
                    if is_statistical_analysis or is_domain_insights:
                        explanation = get_chart_explanation(chart_name, i, is_domain_insights)
                        if explanation:
                            with st.expander("📝 What this chart tells us"):
                                st.markdown(explanation)
            else:
                col_idx = i % 2
                with cols[col_idx]:
                    st.warning(f"Chart {i+1} file not found: {chart_path}")
    else:
        # If no charts but we have insights, just show a simple message
        insights = result.get('insights', {})
        if insights:
            pass  # Don't show any message, just display insights
        else:
            st.warning("⚠️ Analysis completed but no charts were generated. This might be due to data format or analysis limitations.")
    
    # --- Downloadable files ---
    files = result.get('files', [])
    if files:
        st.subheader("📁 Generated Files")
        
        # Create tabs for different file types
        file_tabs = st.tabs(["Markdown", "Python Code", "All Files"])
        
        with file_tabs[0]:
            md_files = [f for f in files if (isinstance(f, str) and f.endswith('.md')) or 
                       (isinstance(f, dict) and f.get('name', '').endswith('.md'))]
            if md_files:
                for file_item in md_files:
                    # Handle both string paths and dictionary objects
                    if isinstance(file_item, dict):
                        file_path = file_item.get('path', '')
                        file_name = file_item.get('name', '')
                    else:
                        file_path = file_item
                        file_name = os.path.basename(file_path)
                    
                    if file_path and os.path.exists(file_path):
                        st.markdown(f"**📄 {file_name}:**")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.markdown(content)
                        
                        # Download button
                        with open(file_path, "rb") as file_data:
                            st.download_button(
                                label=f"📥 Download {file_name}",
                                data=file_data,
                                file_name=file_name,
                                key=f"md_download_{file_name}"
                            )
            else:
                st.info("No markdown files available")
        
        with file_tabs[1]:
            py_files = [f for f in files if (isinstance(f, str) and f.endswith('.py')) or 
                       (isinstance(f, dict) and f.get('name', '').endswith('.py'))]
            if py_files:
                for file_item in py_files:
                    # Handle both string paths and dictionary objects
                    if isinstance(file_item, dict):
                        file_path = file_item.get('path', '')
                        file_name = file_item.get('name', '')
                    else:
                        file_path = file_item
                        file_name = os.path.basename(file_path)
                    
                    if file_path and os.path.exists(file_path):
                        st.markdown(f"**🐍 {file_name}:**")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.code(content, language='python')
                        
                        # Download button
                        with open(file_path, "rb") as file_data:
                            st.download_button(
                                label=f"📥 Download {file_name}",
                                data=file_data,
                                file_name=file_name,
                                key=f"py_download_{file_name}"
                            )
            else:
                st.info("No Python files available")
        
        with file_tabs[2]:
            for file_item in files:
                # Handle both string paths and dictionary objects
                if isinstance(file_item, dict):
                    file_path = file_item.get('path', '')
                    file_name = file_item.get('name', '')
                else:
                    file_path = file_item
                    file_name = os.path.basename(file_path)
                
                if file_path and os.path.exists(file_path):
                    # Download button for all files
                    with open(file_path, "rb") as file_data:
                        st.download_button(
                            label=f"📥 Download {file_name}",
                            data=file_data,
                            file_name=file_name,
                            key=f"file_download_{file_name}"
                        )
    
    # --- Raw results (for debugging) ---
    if st.checkbox("Show raw results (for debugging)"):
        st.subheader("🔧 Raw Results")
        st.json(result.get('raw', {}))

# Add a clear results button
if st.session_state.results is not None:
    if st.button("🔄 Clear Results & Start New Analysis"):
        st.session_state.results = None
        st.session_state.current_agent = None
        st.rerun() 