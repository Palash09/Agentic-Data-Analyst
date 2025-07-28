"""
Enhanced Analysis Agent with Advanced Statistical Capabilities
============================================================

This agent provides deeper, more novel insights through:
- Advanced statistical analysis
- Hypothesis generation and testing
- Domain-specific reasoning
- Multi-dimensional data exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import json
import tempfile
import os
from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class EnhancedAnalysisAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}},
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data scientist specializing in generating novel, 
            deep insights from data analysis results. Given statistical analysis results, 
            generate surprising insights that go beyond obvious patterns."""),
            ("human", """
            Dataset Overview: {dataset_overview}
            Statistical Analysis Results: {statistical_results}
            Correlations: {correlations}
            Clustering Results: {clustering}
            Anomalies: {anomalies}
            
            Generate novel insights in JSON format with:
            - surprising_findings: List of unexpected discoveries
            - domain_implications: Business/domain-specific implications
            - predictive_insights: Forward-looking predictions
            - actionable_recommendations: Specific actions based on findings
            - statistical_significance: Key statistical insights
            """)
        ])

    def analyze_dataset(self, csv_path: str) -> Dict[str, Any]:
        """Perform comprehensive analysis with statistical depth"""
        try:
            df = pd.read_csv(csv_path)
            
            # 1. Advanced Statistical Analysis
            statistical_results = self._perform_statistical_analysis(df)
            
            # 2. Correlation and Relationship Analysis
            correlations = self._analyze_correlations(df)
            
            # 3. Clustering and Segmentation
            clustering_results = self._perform_clustering(df)
            
            # 4. Anomaly Detection
            anomalies = self._detect_anomalies(df)
            
            # 5. Generate Enhanced Insights
            insights = self._generate_enhanced_insights(
                df, statistical_results, correlations, 
                clustering_results, anomalies
            )
            
            # 6. Create Advanced Visualizations
            charts = self._create_advanced_charts(
                df, statistical_results, clustering_results, anomalies
            )
            
            return {
                'agent': 'Enhanced Analysis Agent',
                'insights': insights,
                'statistical_analysis': statistical_results,
                'correlations': correlations,
                'clustering': clustering_results,
                'anomalies': anomalies,
                'charts': charts,
                'raw': {
                    'dataset_shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict()
                }
            }
            
        except Exception as e:
            return {
                'agent': 'Enhanced Analysis Agent',
                'error': str(e),
                'insights': {'error': f'Analysis failed: {str(e)}'},
                'charts': []
            }

    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced statistical analysis"""
        results = {}
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results['descriptive_stats'] = df[numeric_cols].describe().to_dict()
            
            # Normality tests
            results['normality_tests'] = {}
            for col in numeric_cols:
                if df[col].notna().sum() > 8:  # Need at least 8 samples
                    stat, p_value = stats.shapiro(df[col].dropna())
                    results['normality_tests'][col] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
            
            # Distribution analysis
            results['distributions'] = {}
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    results['distributions'][col] = {
                        'skewness': float(stats.skew(data)),
                        'kurtosis': float(stats.kurtosis(data)),
                        'variance': float(np.var(data)),
                        'cv': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0
                    }
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        results['categorical_analysis'] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            results['categorical_analysis'][col] = {
                'unique_values': len(value_counts),
                'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                'entropy': float(stats.entropy(value_counts.values)),
                'distribution': value_counts.head(10).to_dict()
            }
        
        return results

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations and relationships"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}
        
        # Pearson correlations
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                    })
        
        # Partial correlations (controlling for other variables)
        partial_corr = {}
        if len(numeric_cols) >= 3:
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    # Simple partial correlation (could be enhanced)
                    other_cols = [c for c in numeric_cols if c not in [col1, col2]]
                    if other_cols:
                        control_col = other_cols[0]  # Use first other column as control
                        try:
                            r12 = corr_matrix.loc[col1, col2]
                            r1c = corr_matrix.loc[col1, control_col]
                            r2c = corr_matrix.loc[col2, control_col]
                            
                            partial_r = (r12 - r1c * r2c) / np.sqrt((1 - r1c**2) * (1 - r2c**2))
                            partial_corr[f"{col1}_{col2}"] = float(partial_r)
                        except:
                            continue
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_corr,
            'partial_correlations': partial_corr
        }

    def _perform_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Not enough numeric columns for clustering'}
        
        # Prepare data
        data = df[numeric_cols].dropna()
        if len(data) < 4:
            return {'message': 'Not enough data points for clustering'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, min(8, len(data)))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simple method)
        optimal_k = 3  # Default
        if len(inertias) >= 2:
            diffs = np.diff(inertias)
            optimal_k = np.argmax(diffs) + 2  # +2 because we start from k=2
        
        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Analyze cluster characteristics
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters
        
        cluster_stats = {}
        for cluster_id in range(optimal_k):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data_with_clusters) * 100,
                'centroid': cluster_data[numeric_cols].mean().to_dict()
            }
        
        return {
            'optimal_clusters': optimal_k,
            'cluster_assignments': clusters.tolist(),
            'cluster_statistics': cluster_stats,
            'inertias': inertias
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'message': 'No numeric columns for anomaly detection'}
        
        data = df[numeric_cols].dropna()
        if len(data) < 4:
            return {'message': 'Not enough data for anomaly detection'}
        
        # Isolation Forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = isolation_forest.fit_predict(data)
        
        # Statistical outliers (IQR method)
        statistical_outliers = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            statistical_outliers[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100,
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            }
        
        return {
            'isolation_forest_anomalies': {
                'count': int(np.sum(anomaly_labels == -1)),
                'percentage': float(np.sum(anomaly_labels == -1) / len(anomaly_labels) * 100),
                'indices': np.where(anomaly_labels == -1)[0].tolist()
            },
            'statistical_outliers': statistical_outliers
        }

    def _generate_enhanced_insights(self, df: pd.DataFrame, statistical_results: Dict, 
                                  correlations: Dict, clustering_results: Dict, 
                                  anomalies: Dict) -> Dict[str, Any]:
        """Generate enhanced insights using LLM"""
        
        # Prepare context for LLM
        dataset_overview = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        try:
            response = self.llm.invoke(
                self.analysis_prompt.format_messages(
                    dataset_overview=json.dumps(dataset_overview),
                    statistical_results=json.dumps(statistical_results, default=str),
                    correlations=json.dumps(correlations, default=str),
                    clustering=json.dumps(clustering_results, default=str),
                    anomalies=json.dumps(anomalies, default=str)
                )
            )
            
            insights = json.loads(response.content)
            return insights
            
        except Exception as e:
            return {
                'error': f'Failed to generate insights: {str(e)}',
                'basic_insights': {
                    'dataset_size': df.shape,
                    'strong_correlations': len(correlations.get('strong_correlations', [])),
                    'anomaly_percentage': anomalies.get('isolation_forest_anomalies', {}).get('percentage', 0)
                }
            }

    def _create_advanced_charts(self, df: pd.DataFrame, statistical_results: Dict,
                              clustering_results: Dict, anomalies: Dict) -> List[str]:
        """Create advanced visualization charts"""
        charts = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Correlation Heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap', fontsize=14)
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'correlation_heatmap.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
            
            # 2. Clustering Visualization (PCA)
            if clustering_results.get('cluster_assignments') and len(numeric_cols) > 1:
                data = df[numeric_cols].dropna()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                
                if len(numeric_cols) > 2:
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(scaled_data)
                else:
                    pca_data = scaled_data
                
                plt.figure(figsize=(10, 8))
                clusters = np.array(clustering_results['cluster_assignments'])
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
                plt.xlabel('First Principal Component' if len(numeric_cols) > 2 else numeric_cols[0])
                plt.ylabel('Second Principal Component' if len(numeric_cols) > 2 else numeric_cols[1])
                plt.title('Cluster Analysis Visualization')
                plt.colorbar(scatter)
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'cluster_analysis.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
            
            # 3. Anomaly Detection Visualization
            if len(numeric_cols) > 1 and anomalies.get('isolation_forest_anomalies'):
                data = df[numeric_cols].dropna()
                anomaly_indices = anomalies['isolation_forest_anomalies']['indices']
                
                plt.figure(figsize=(10, 8))
                # Plot first two numeric columns
                col1, col2 = numeric_cols[0], numeric_cols[1]
                normal_mask = ~data.index.isin(anomaly_indices)
                
                plt.scatter(data.loc[normal_mask, col1], data.loc[normal_mask, col2], 
                           c='blue', alpha=0.6, label='Normal')
                if anomaly_indices:
                    anomaly_data = data.iloc[anomaly_indices]
                    plt.scatter(anomaly_data[col1], anomaly_data[col2], 
                               c='red', alpha=0.8, label='Anomaly')
                
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.title('Anomaly Detection')
                plt.legend()
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'anomaly_detection.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                charts.append(chart_path)
                
        except Exception as e:
            print(f"Error creating advanced charts: {e}")
        
        return charts


def run_enhanced_analysis(csv_path: str) -> Dict[str, Any]:
    """Main function to run enhanced analysis"""
    agent = EnhancedAnalysisAgent()
    return agent.analyze_dataset(csv_path) 