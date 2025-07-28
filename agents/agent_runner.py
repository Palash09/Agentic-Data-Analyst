import os
import tempfile
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Import agent functions
from agents.d2insight_agent_sys import run_domain_detector
from agents.d2insight_gpt4o import analyze_csv_with_insights
from agents.d2insight_gpt4o_domain import analyze_csv_with_insights_domain
from agents.insight2dashboard_tot import generate_analysis
from agents.enhanced_analysis_agent import run_enhanced_analysis
from agents.data_preprocessor import preprocess_for_analysis

# User-friendly agent mapping
AGENT_MAP = {
    'enhanced_analysis': {
        'name': '📊 Statistical Analysis (Recommended)',
        'function': 'run_enhanced_analysis_agent',
    },
    'system_agent': {
        'name': 'Primary Insights',
        'function': 'run_system_agent',
    },
    'gpt4o_domain_insights': {
        'name': 'Domain Insights',
        'function': 'run_gpt4o_domain_insights',
    },
    'tree_of_thought_dashboard': {
        'name': 'Deep-Dive',
        'function': 'run_tree_of_thought_dashboard',
    },
}

DEFAULT_PROMPT = "Please analyze this dataset and provide key insights."

def is_gpt4o_error(error_str):
    """Check if the error is related to GPT-4o access"""
    return ("gpt-4o" in error_str.lower() and 
            ("does not have access" in error_str.lower() or 
             "model_not_found" in error_str.lower() or
             "403" in error_str))

def generate_basic_charts(csv_path, temp_dir):
    """Generate basic charts for the dataset"""
    try:
        df = pd.read_csv(csv_path)
        charts = []
        
        # Create charts directory
        charts_dir = os.path.join(temp_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Chart 1: Age distribution (if age column exists)
        if 'age' in df.columns:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(df['age'], bins=10, edgecolor='black', alpha=0.7)
                plt.title('Age Distribution')
                plt.xlabel('Age')
                plt.ylabel('Frequency')
                plt.tight_layout()
                age_chart_path = os.path.join(charts_dir, "age_distribution.png")
                plt.savefig(age_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append(age_chart_path)
            except Exception as e:
                print(f"Error creating age chart: {e}")
        
        # Chart 2: Gender distribution (if gender column exists)
        if 'gender' in df.columns:
            try:
                plt.figure(figsize=(8, 6))
                gender_counts = df['gender'].value_counts()
                plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
                plt.title('Gender Distribution')
                plt.tight_layout()
                gender_chart_path = os.path.join(charts_dir, "gender_distribution.png")
                plt.savefig(gender_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append(gender_chart_path)
            except Exception as e:
                print(f"Error creating gender chart: {e}")
        
        # Chart 3: Investment preferences (if numeric columns exist)
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            investment_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['fund', 'equity', 'bond', 'deposit', 'gold', 'ppf'])]
            
            if investment_cols:
                plt.figure(figsize=(12, 6))
                investment_means = df[investment_cols].mean().sort_values(ascending=True)
                plt.barh(investment_means.index, investment_means.values)
                plt.title('Average Investment Preferences (Lower = More Preferred)')
                plt.xlabel('Average Rating')
                plt.tight_layout()
                investment_chart_path = os.path.join(charts_dir, "investment_preferences.png")
                plt.savefig(investment_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append(investment_chart_path)
        except Exception as e:
            print(f"Error creating investment chart: {e}")
        
        # Chart 4: General numeric columns overview (if no specific charts were created)
        if not charts:
            try:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    # Take first few numeric columns for overview
                    cols_to_plot = numeric_cols[:min(5, len(numeric_cols))]
                    plt.figure(figsize=(12, 6))
                    df[cols_to_plot].boxplot()
                    plt.title('Numeric Columns Overview')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    overview_chart_path = os.path.join(charts_dir, "numeric_overview.png")
                    plt.savefig(overview_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    charts.append(overview_chart_path)
            except Exception as e:
                print(f"Error creating overview chart: {e}")
        
        # Chart 5: Categorical columns distribution (if no charts yet)
        if not charts:
            try:
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    # Take first categorical column
                    col = categorical_cols[0]
                    plt.figure(figsize=(10, 6))
                    value_counts = df[col].value_counts().head(10)  # Top 10 values
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel('Categories')
                    plt.ylabel('Count')
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                    plt.tight_layout()
                    cat_chart_path = os.path.join(charts_dir, "categorical_distribution.png")
                    plt.savefig(cat_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    charts.append(cat_chart_path)
            except Exception as e:
                print(f"Error creating categorical chart: {e}")
        
        print(f"Generated {len(charts)} charts successfully")
        return charts
    except Exception as e:
        print(f"Error generating charts: {e}")
        return []

def run_system_agent(csv_path):
    try:
        result = run_domain_detector(csv_path)
        # Extract insights for standardization
        analysis = result.get('analysis', {})
        insights = {
            'descriptive': analysis.get('analysis', {}).get('descriptive', ''),
            'predictive': analysis.get('analysis', {}).get('predictive', ''),
            'domain_related': analysis.get('analysis', {}).get('domain_related', ''),
        }
        return {
            'agent': AGENT_MAP['system_agent']['name'],
            'insights': insights,
            'raw': result,
            'files': [],
            'charts': [],
        }
    except Exception as e:
        error_str = str(e)
        if is_gpt4o_error(error_str):
            print(f"GPT-4o not available, trying fallback: {error_str}")
            return run_system_agent_fallback(csv_path)
        else:
            raise e

def run_system_agent_fallback(csv_path):
    """Fallback version using gpt-3.5-turbo"""
    try:
        # Create a temporary directory for charts
        temp_dir = tempfile.mkdtemp()
        
        # Temporarily modify the model in the imported module
        import agents.d2insight_agent_sys as sys_module
        original_model = sys_module.llm.model_name
        sys_module.llm.model_name = "gpt-3.5-turbo"
        print(f"Using fallback model: gpt-3.5-turbo (was: {original_model})")
        
        result = run_domain_detector(csv_path)
        
        # Restore original model
        sys_module.llm.model_name = original_model
        
        # Generate basic charts
        charts = generate_basic_charts(csv_path, temp_dir)
        
        # Extract insights for standardization
        analysis = result.get('analysis', {})
        insights = {
            'descriptive': analysis.get('analysis', {}).get('descriptive', ''),
            'predictive': analysis.get('analysis', {}).get('predictive', ''),
            'domain_related': analysis.get('analysis', {}).get('domain_related', ''),
        }
        return {
            'agent': AGENT_MAP['system_agent']['name'] + ' (GPT-3.5-turbo)',
            'insights': insights,
            'raw': result,
            'files': [],
            'charts': charts,
        }
    except Exception as e:
        print(f"Fallback also failed: {str(e)}")
        # Try to generate charts even if analysis fails
        try:
            temp_dir = tempfile.mkdtemp()
            charts = generate_basic_charts(csv_path, temp_dir)
        except:
            charts = []
        
        return {
            'agent': AGENT_MAP['system_agent']['name'],
            'insights': {
                'descriptive': f'Analysis failed: {str(e)}',
                'predictive': 'Analysis failed',
                'domain_related': 'Analysis failed'
            },
            'raw': {'error': str(e)},
            'files': [],
            'charts': charts,
        }

def run_gpt4o_insights(csv_path):
    try:
        output = analyze_csv_with_insights(csv_path, DEFAULT_PROMPT)
        try:
            insights = json.loads(output)
        except Exception:
            insights = {'raw': output}
        
        # Generate charts
        temp_dir = tempfile.mkdtemp()
        charts = generate_basic_charts(csv_path, temp_dir)
        
        return {
            'agent': AGENT_MAP['gpt4o_insights']['name'],
            'insights': insights,
            'raw': output,
            'files': [],
            'charts': charts,
        }
    except Exception as e:
        error_str = str(e)
        if is_gpt4o_error(error_str):
            print(f"GPT-4o not available, trying fallback: {error_str}")
            return run_gpt4o_insights_fallback(csv_path)
        else:
            raise e

def run_gpt4o_insights_fallback(csv_path):
    """Fallback version using gpt-3.5-turbo"""
    try:
        import agents.d2insight_gpt4o as gpt4o_module
        original_model = gpt4o_module.llm.model_name
        gpt4o_module.llm.model_name = "gpt-3.5-turbo"
        print(f"Using fallback model: gpt-3.5-turbo (was: {original_model})")
        
        output = analyze_csv_with_insights(csv_path, DEFAULT_PROMPT)
        
        # Restore original model
        gpt4o_module.llm.model_name = original_model
        
        # Generate charts
        temp_dir = tempfile.mkdtemp()
        charts = generate_basic_charts(csv_path, temp_dir)
        
        try:
            insights = json.loads(output)
        except Exception:
            insights = {'raw': output}
        return {
            'agent': AGENT_MAP['gpt4o_insights']['name'] + ' (GPT-3.5-turbo)',
            'insights': insights,
            'raw': output,
            'files': [],
            'charts': charts,
        }
    except Exception as e:
        print(f"Fallback also failed: {str(e)}")
        # Try to generate charts even if analysis fails
        try:
            temp_dir = tempfile.mkdtemp()
            charts = generate_basic_charts(csv_path, temp_dir)
        except:
            charts = []
        
        return {
            'agent': AGENT_MAP['gpt4o_insights']['name'],
            'insights': {'error': str(e)},
            'raw': str(e),
            'files': [],
            'charts': charts,
        }

def run_gpt4o_domain_insights(csv_path):
    try:
        output = analyze_csv_with_insights_domain(csv_path, DEFAULT_PROMPT)
        try:
            insights = json.loads(output)
            # Ensure we have the proper structure for domain insights
            if isinstance(insights, dict):
                # Extract and format domain-related insights
                descriptive = insights.get('descriptive', insights.get('analysis', ''))
                predictive = insights.get('predictive', '')
                domain_related = insights.get('domain_related', insights.get('domain_insights', ''))
                
                # If descriptive is a dict, try to extract the text content
                if isinstance(descriptive, dict):
                    # Look for common keys that might contain the actual analysis text
                    descriptive_text = (descriptive.get('text', '') or 
                                      descriptive.get('content', '') or 
                                      descriptive.get('summary', '') or
                                      str(descriptive))
                else:
                    descriptive_text = str(descriptive) if descriptive else ''
                
                # If predictive is a dict, extract text content
                if isinstance(predictive, dict):
                    predictive_text = (predictive.get('text', '') or 
                                     predictive.get('content', '') or 
                                     predictive.get('summary', '') or
                                     str(predictive))
                else:
                    predictive_text = str(predictive) if predictive else ''
                
                # If domain_related is a dict, extract text content
                if isinstance(domain_related, dict):
                    domain_text = (domain_related.get('text', '') or 
                                 domain_related.get('content', '') or 
                                 domain_related.get('summary', '') or
                                 str(domain_related))
                else:
                    domain_text = str(domain_related) if domain_related else ''
                
                domain_insights = {
                    'descriptive': descriptive_text,
                    'predictive': predictive_text,
                    'domain_related': domain_text,
                    'raw': insights
                }
            else:
                domain_insights = {'raw': insights}
        except Exception as e:
            print(f"Error parsing insights JSON: {e}")
            domain_insights = {'raw': output}
        
        # Generate charts using the enhanced chart generation
        temp_dir = tempfile.mkdtemp()
        charts = generate_enhanced_charts_for_domain(csv_path, temp_dir)
        
        return {
            'agent': AGENT_MAP['gpt4o_domain_insights']['name'],
            'insights': domain_insights,
            'raw': output,
            'files': [],
            'charts': charts,
        }
    except Exception as e:
        error_str = str(e)
        if is_gpt4o_error(error_str):
            print(f"GPT-4o not available, trying fallback: {error_str}")
            return run_gpt4o_domain_insights_fallback(csv_path)
        else:
            raise e

def run_gpt4o_domain_insights_fallback(csv_path):
    """Fallback version using gpt-3.5-turbo"""
    try:
        import agents.d2insight_gpt4o_domain as domain_module
        original_model = domain_module.llm.model_name
        domain_module.llm.model_name = "gpt-3.5-turbo"
        print(f"Using fallback model: gpt-3.5-turbo (was: {original_model})")
        
        output = analyze_csv_with_insights_domain(csv_path, DEFAULT_PROMPT)
        
        # Restore original model
        domain_module.llm.model_name = original_model
        
        # Generate charts
        temp_dir = tempfile.mkdtemp()
        charts = generate_basic_charts(csv_path, temp_dir) # Fallback to basic charts for domain insights
        
        try:
            insights = json.loads(output)
            # Ensure we have the proper structure for domain insights
            if isinstance(insights, dict):
                # Extract and format domain-related insights
                descriptive = insights.get('descriptive', insights.get('analysis', ''))
                predictive = insights.get('predictive', '')
                domain_related = insights.get('domain_related', insights.get('domain_insights', ''))
                
                # If descriptive is a dict, try to extract the text content
                if isinstance(descriptive, dict):
                    # Look for common keys that might contain the actual analysis text
                    descriptive_text = (descriptive.get('text', '') or 
                                      descriptive.get('content', '') or 
                                      descriptive.get('summary', '') or
                                      str(descriptive))
                else:
                    descriptive_text = str(descriptive) if descriptive else ''
                
                # If predictive is a dict, extract text content
                if isinstance(predictive, dict):
                    predictive_text = (predictive.get('text', '') or 
                                     predictive.get('content', '') or 
                                     predictive.get('summary', '') or
                                     str(predictive))
                else:
                    predictive_text = str(predictive) if predictive else ''
                
                # If domain_related is a dict, extract text content
                if isinstance(domain_related, dict):
                    domain_text = (domain_related.get('text', '') or 
                                 domain_related.get('content', '') or 
                                 domain_related.get('summary', '') or
                                 str(domain_related))
                else:
                    domain_text = str(domain_related) if domain_related else ''
                
                domain_insights = {
                    'descriptive': descriptive_text,
                    'predictive': predictive_text,
                    'domain_related': domain_text,
                    'raw': insights
                }
            else:
                domain_insights = {'raw': insights}
        except Exception as e:
            print(f"Error parsing insights JSON: {e}")
            domain_insights = {'raw': output}
        
        return {
            'agent': AGENT_MAP['gpt4o_domain_insights']['name'] + ' (GPT-3.5-turbo)',
            'insights': domain_insights,
            'raw': output,
            'files': [],
            'charts': charts,
        }
    except Exception as e:
        print(f"Fallback also failed: {str(e)}")
        # Try to generate charts even if analysis fails
        try:
            temp_dir = tempfile.mkdtemp()
            charts = generate_basic_charts(csv_path, temp_dir)
        except:
            charts = []
        
        return {
            'agent': AGENT_MAP['gpt4o_domain_insights']['name'],
            'insights': {'error': str(e)},
            'raw': str(e),
            'files': [],
            'charts': charts,
        }

def run_tree_of_thought_dashboard(csv_path):
    temp_dir = None
    try:
        # Create a temporary directory for the output
        temp_dir = tempfile.mkdtemp()
        
        # Copy the CSV file to the temp directory with a standard name
        import shutil
        csv_filename = os.path.basename(csv_path)
        temp_csv_path = os.path.join(temp_dir, csv_filename)
        shutil.copy2(csv_path, temp_csv_path)
        
        # Create a basic insight JSON file for the Tree-of-Thought analysis
        insight_data = {
            "insights": [
                {
                    "title": "Data Overview",
                    "description": "Comprehensive analysis of the uploaded dataset",
                    "domain": "General",
                    "key_findings": ["Data structure analysis", "Pattern identification", "Trend analysis"]
                }
            ]
        }
        insight_json_path = os.path.join(temp_dir, "insights.json")
        with open(insight_json_path, 'w') as f:
            json.dump(insight_data, f, indent=2)
        
        # Run the Tree-of-Thought dashboard
        output = generate_analysis(
            csv_path=temp_csv_path,
            insight_json_path=insight_json_path,
            save_dir=temp_dir,
            run_code=True
        )
        
        # Collect generated files
        files = []
        if os.path.exists(temp_dir):
            for root, dirs, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if filename.endswith(('.md', '.py', '.txt', '.csv', '.json')):
                        files.append({
                            'name': filename,
                            'path': file_path,
                            'type': 'file'
                        })
        
        # Look for generated chart files
        charts = []
        for root, dirs, filenames in os.walk(temp_dir):
            for filename in filenames:
                if filename.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                    chart_path = os.path.join(root, filename)
                    charts.append({
                        'name': filename,
                        'path': chart_path,
                        'type': 'chart'
                    })
        
        return {
            'agent': AGENT_MAP['tree_of_thought_dashboard']['name'],
            'insights': {'raw': output},
            'raw': output,
            'files': files,
            'charts': charts,
        }
    except Exception as e:
        print(f"Tree-of-Thought Dashboard error: {str(e)}")
        # Try to generate basic charts as fallback
        charts = []
        try:
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            charts = generate_basic_charts(csv_path, temp_dir)
        except Exception as chart_error:
            print(f"Error generating fallback charts: {chart_error}")
            charts = []
        
        return {
            'agent': AGENT_MAP['tree_of_thought_dashboard']['name'],
            'insights': {'error': str(e)},
            'raw': str(e),
            'files': [],
            'charts': charts,
        }

def run_enhanced_analysis_agent(csv_path):
    """Run the enhanced statistical analysis agent with preprocessing"""
    try:
        # First preprocess the data
        df_cleaned, preprocessing_report = preprocess_for_analysis(csv_path)
        
        # Save cleaned data to temp file
        temp_dir = tempfile.mkdtemp()
        cleaned_csv_path = os.path.join(temp_dir, "cleaned_data.csv")
        df_cleaned.to_csv(cleaned_csv_path, index=False)
        
        # Run enhanced analysis on cleaned data
        result = run_enhanced_analysis(cleaned_csv_path)
        
        # Add preprocessing info to result
        result['preprocessing_report'] = preprocessing_report
        result['data_quality_score'] = calculate_data_quality_score(preprocessing_report)
        
        return result
        
    except Exception as e:
        print(f"Enhanced analysis failed: {str(e)}")
        # Fallback to basic analysis
        return run_system_agent_fallback(csv_path)

def calculate_data_quality_score(preprocessing_report):
    """Calculate a data quality score based on preprocessing results"""
    score = 100  # Start with perfect score
    
    # Deduct points for data issues
    if preprocessing_report.get('data_reduction_percentage', 0) > 10:
        score -= 20  # Significant data loss
    
    if len(preprocessing_report.get('data_quality_issues', [])) > 3:
        score -= 15  # Many quality issues
    
    # Add points for good data characteristics
    if preprocessing_report.get('data_reduction_percentage', 0) < 5:
        score += 5  # Minimal data loss
    
    return max(0, min(100, score))

def run_agent(agent_key, csv_path):
    if agent_key == 'enhanced_analysis':
        return run_enhanced_analysis_agent(csv_path)
    elif agent_key == 'system_agent':
        return run_system_agent(csv_path)
    elif agent_key == 'gpt4o_insights':
        return run_gpt4o_insights(csv_path)
    elif agent_key == 'gpt4o_domain_insights':
        return run_gpt4o_domain_insights(csv_path)
    elif agent_key == 'tree_of_thought_dashboard':
        return run_tree_of_thought_dashboard(csv_path)
    else:
        raise ValueError(f"Unknown agent key: {agent_key}")

def get_agent_choices():
    """Return a list of (key, user-friendly name) tuples for UI dropdowns."""
    return [(k, v['name']) for k, v in AGENT_MAP.items()] 

def generate_enhanced_charts_for_domain(csv_path, temp_dir):
    """Generate ML-based predictive analytics charts for domain insights"""
    try:
        df = pd.read_csv(csv_path)
        charts = []
        
        # Create charts directory
        charts_dir = os.path.join(temp_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Chart 1: Feature Importance Analysis (if enough numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            try:
                # Create a simple feature importance visualization
                # For demonstration, we'll use correlation with a target variable
                target_col = numeric_cols[0]  # Use first numeric column as target
                feature_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Use up to 5 features
                
                correlations = []
                for col in feature_cols:
                    corr = df[col].corr(df[target_col])
                    correlations.append(abs(corr))
                
                plt.figure(figsize=(10, 6))
                plt.barh(feature_cols, correlations)
                plt.title(f'Feature Importance Analysis\n(Correlation with {target_col})')
                plt.xlabel('Absolute Correlation')
                plt.ylabel('Features')
                plt.tight_layout()
                chart_path = os.path.join(charts_dir, "domain_feature_importance.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append({
                    'name': 'Domain Feature Importance',
                    'path': chart_path,
                    'type': 'chart'
                })
            except Exception as e:
                print(f"Error creating feature importance chart: {e}")
        
        # Chart 2: Predictive Trend Analysis
        if len(numeric_cols) > 1:
            try:
                # Create a trend analysis chart
                key_cols = numeric_cols[:min(3, len(numeric_cols))]
                fig, axes = plt.subplots(1, len(key_cols), figsize=(15, 5))
                if len(key_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(key_cols):
                    # Sort by the column values to show trend
                    sorted_data = df[col].dropna().sort_values()
                    axes[i].plot(range(len(sorted_data)), sorted_data.values, alpha=0.7)
                    axes[i].set_title(f'{col} Trend Analysis')
                    axes[i].set_xlabel('Data Points')
                    axes[i].set_ylabel(col)
                    
                    # Add trend line
                    if len(sorted_data) > 1:
                        z = np.polyfit(range(len(sorted_data)), sorted_data.values, 1)
                        p = np.poly1d(z)
                        axes[i].plot(range(len(sorted_data)), p(range(len(sorted_data))), "r--", alpha=0.8)
                
                plt.tight_layout()
                chart_path = os.path.join(charts_dir, "domain_trend_analysis.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append({
                    'name': 'Domain Trend Analysis',
                    'path': chart_path,
                    'type': 'chart'
                })
            except Exception as e:
                print(f"Error creating trend analysis chart: {e}")
        
        # Chart 3: Predictive Clustering Analysis
        if len(numeric_cols) > 1:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Prepare data for clustering
                cluster_data = df[numeric_cols].dropna()
                if len(cluster_data) > 10:  # Need enough data points
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=min(3, len(cluster_data)//3), random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    # Visualize clusters (use first two dimensions)
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis')
                    plt.title('Predictive Clustering Analysis')
                    plt.xlabel(f'{numeric_cols[0]} (Standardized)')
                    plt.ylabel(f'{numeric_cols[1]} (Standardized)')
                    plt.colorbar(scatter, label='Cluster')
                    plt.tight_layout()
                    chart_path = os.path.join(charts_dir, "domain_clustering_analysis.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    charts.append({
                        'name': 'Domain Clustering Analysis',
                        'path': chart_path,
                        'type': 'chart'
                    })
            except Exception as e:
                print(f"Error creating clustering analysis chart: {e}")
        
        # Chart 4: Predictive Outlier Detection
        if len(numeric_cols) > 0:
            try:
                from sklearn.ensemble import IsolationForest
                
                # Use first numeric column for outlier detection
                outlier_data = df[numeric_cols[0]].dropna()
                if len(outlier_data) > 10:
                    # Reshape for sklearn
                    X = outlier_data.values.reshape(-1, 1)
                    
                    # Detect outliers
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(X)
                    
                    # Create outlier visualization
                    plt.figure(figsize=(12, 6))
                    
                    # Plot normal points
                    normal_mask = outlier_labels == 1
                    plt.scatter(range(len(outlier_data)), outlier_data[normal_mask], 
                               c='blue', alpha=0.6, label='Normal Data')
                    
                    # Plot outliers
                    outlier_mask = outlier_labels == -1
                    if outlier_mask.sum() > 0:
                        plt.scatter(range(len(outlier_data))[outlier_mask], 
                                   outlier_data[outlier_mask], 
                                   c='red', alpha=0.8, label='Outliers')
                    
                    plt.title(f'Predictive Outlier Detection - {numeric_cols[0]}')
                    plt.xlabel('Data Points')
                    plt.ylabel(numeric_cols[0])
                    plt.legend()
                    plt.tight_layout()
                    chart_path = os.path.join(charts_dir, "domain_outlier_detection.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    charts.append({
                        'name': 'Domain Outlier Detection',
                        'path': chart_path,
                        'type': 'chart'
                    })
            except Exception as e:
                print(f"Error creating outlier detection chart: {e}")
        
        # Chart 5: Predictive Pattern Analysis (for categorical data)
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            try:
                # Analyze patterns between categorical and numeric variables
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Group by categorical variable and analyze numeric patterns
                grouped_data = df.groupby(cat_col)[num_col].agg(['mean', 'count']).head(10)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Mean values
                grouped_data['mean'].plot(kind='bar', ax=ax1, color='skyblue')
                ax1.set_title(f'Average {num_col} by {cat_col}')
                ax1.set_xlabel(cat_col)
                ax1.set_ylabel(f'Average {num_col}')
                ax1.tick_params(axis='x', rotation=45)
                
                # Count values
                grouped_data['count'].plot(kind='bar', ax=ax2, color='lightcoral')
                ax2.set_title(f'Count by {cat_col}')
                ax2.set_xlabel(cat_col)
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                chart_path = os.path.join(charts_dir, "domain_pattern_analysis.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append({
                    'name': 'Domain Pattern Analysis',
                    'path': chart_path,
                    'type': 'chart'
                })
            except Exception as e:
                print(f"Error creating pattern analysis chart: {e}")
        
        print(f"Generated {len(charts)} ML-based domain charts successfully")
        return charts
    except Exception as e:
        print(f"Error generating ML-based domain charts: {e}")
        return [] 