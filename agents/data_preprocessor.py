"""
Advanced Data Preprocessing and Validation Module
===============================================

Handles data cleaning, validation, and preparation for enhanced analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
from datetime import datetime
import re

class DataPreprocessor:
    def __init__(self):
        self.preprocessing_log = []
        self.data_issues = []
        
    def preprocess_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data preprocessing pipeline
        
        Returns:
            - Cleaned DataFrame
            - Preprocessing report
        """
        original_shape = df.shape
        report = {
            'original_shape': original_shape,
            'preprocessing_steps': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        # 1. Data Type Detection and Conversion
        df_cleaned = self._detect_and_convert_types(df, report)
        
        # 2. Handle Missing Values
        df_cleaned = self._handle_missing_values(df_cleaned, report)
        
        # 3. Remove Duplicates
        df_cleaned = self._handle_duplicates(df_cleaned, report)
        
        # 4. Outlier Detection and Handling
        df_cleaned = self._handle_outliers(df_cleaned, report)
        
        # 5. Data Validation
        self._validate_data_consistency(df_cleaned, report)
        
        # 6. Feature Engineering Suggestions
        self._suggest_feature_engineering(df_cleaned, report)
        
        report['final_shape'] = df_cleaned.shape
        report['data_reduction_percentage'] = (
            (original_shape[0] - df_cleaned.shape[0]) / original_shape[0] * 100
        )
        
        return df_cleaned, report
    
    def _detect_and_convert_types(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Intelligently detect and convert data types"""
        df_processed = df.copy()
        type_changes = []
        
        for col in df.columns:
            original_type = df[col].dtype
            
            # Try to convert to numeric if it looks numeric
            if df[col].dtype == 'object':
                # Check if it's actually numeric (with potential formatting issues)
                sample_values = df[col].dropna().astype(str).head(100)
                
                # Remove common formatting (commas, currency symbols, percentages)
                cleaned_sample = sample_values.str.replace(r'[$,€£¥%]', '', regex=True)
                cleaned_sample = cleaned_sample.str.replace(r'\s+', '', regex=True)
                
                # Check if numeric after cleaning
                numeric_count = 0
                for val in cleaned_sample:
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        continue
                
                if numeric_count / len(cleaned_sample) > 0.8:  # 80% numeric threshold
                    try:
                        # Clean the entire column
                        cleaned_col = df[col].astype(str).str.replace(r'[$,€£¥%]', '', regex=True)
                        cleaned_col = cleaned_col.str.replace(r'\s+', '', regex=True)
                        df_processed[col] = pd.to_numeric(cleaned_col, errors='coerce')
                        type_changes.append(f"{col}: {original_type} -> numeric")
                    except:
                        pass
                
                # Check if it's a date
                elif self._is_date_column(df[col]):
                    try:
                        df_processed[col] = pd.to_datetime(df[col], errors='coerce')
                        type_changes.append(f"{col}: {original_type} -> datetime")
                    except:
                        pass
                
                # Check if it's categorical with low cardinality
                elif df[col].nunique() / len(df) < 0.1 and df[col].nunique() < 50:
                    df_processed[col] = df[col].astype('category')
                    type_changes.append(f"{col}: {original_type} -> category")
        
        if type_changes:
            report['preprocessing_steps'].append({
                'step': 'Type Detection & Conversion',
                'changes': type_changes
            })
        
        return df_processed
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date-like data"""
        sample = series.dropna().astype(str).head(20)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or MM/DD/YYYY
        ]
        
        for pattern in date_patterns:
            matches = sample.str.match(pattern).sum()
            if matches / len(sample) > 0.7:  # 70% match threshold
                return True
        
        return False
    
    def _handle_missing_values(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Handle missing values intelligently"""
        df_processed = df.copy()
        missing_info = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = missing_count / len(df) * 100
            
            if missing_count > 0:
                missing_info.append({
                    'column': col,
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2)
                })
                
                # Strategy based on missing percentage and data type
                if missing_percentage > 70:
                    # Drop columns with too many missing values
                    df_processed = df_processed.drop(columns=[col])
                    report['data_quality_issues'].append(
                        f"Dropped column '{col}' - {missing_percentage:.1f}% missing"
                    )
                elif missing_percentage > 5:
                    if df[col].dtype in ['int64', 'float64']:
                        # Use median for numeric data
                        df_processed[col].fillna(df[col].median(), inplace=True)
                    else:
                        # Use mode for categorical data
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df_processed[col].fillna(mode_val[0], inplace=True)
                        else:
                            df_processed[col].fillna('Unknown', inplace=True)
        
        if missing_info:
            report['preprocessing_steps'].append({
                'step': 'Missing Value Handling',
                'missing_data_info': missing_info
            })
        
        return df_processed
    
    def _handle_duplicates(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Remove duplicate rows"""
        original_count = len(df)
        df_processed = df.drop_duplicates()
        duplicate_count = original_count - len(df_processed)
        
        if duplicate_count > 0:
            report['preprocessing_steps'].append({
                'step': 'Duplicate Removal',
                'duplicates_removed': duplicate_count,
                'duplicate_percentage': round(duplicate_count / original_count * 100, 2)
            })
        
        return df_processed
    
    def _handle_outliers(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Detect and handle outliers"""
        df_processed = df.copy()
        outlier_info = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_percentage = outlier_count / len(df) * 100
                outlier_info.append({
                    'column': col,
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': round(outlier_percentage, 2)
                })
                
                # Cap outliers instead of removing (preserves data)
                if outlier_percentage < 5:  # Only cap if < 5% outliers
                    df_processed.loc[df[col] < lower_bound, col] = lower_bound
                    df_processed.loc[df[col] > upper_bound, col] = upper_bound
        
        if outlier_info:
            report['preprocessing_steps'].append({
                'step': 'Outlier Handling',
                'outlier_info': outlier_info
            })
        
        return df_processed
    
    def _validate_data_consistency(self, df: pd.DataFrame, report: Dict):
        """Validate data consistency and identify potential issues"""
        issues = []
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            issues.append(f"Constant columns detected: {', '.join(constant_cols)}")
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > 0.9:  # Each row almost unique
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            issues.append(f"High cardinality categorical columns: {', '.join(high_cardinality_cols)}")
        
        # Check for potential ID columns
        potential_id_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['id', 'key', 'index']):
                if df[col].nunique() / len(df) > 0.9:
                    potential_id_cols.append(col)
        
        if potential_id_cols:
            issues.append(f"Potential ID columns (consider excluding from analysis): {', '.join(potential_id_cols)}")
        
        report['data_quality_issues'].extend(issues)
    
    def _suggest_feature_engineering(self, df: pd.DataFrame, report: Dict):
        """Suggest potential feature engineering opportunities"""
        suggestions = []
        
        # Date columns - extract components
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            suggestions.append(f"Extract year/month/day from date columns: {', '.join(date_cols)}")
        
        # Text columns - length features
        text_cols = df.select_dtypes(include=['object']).columns
        long_text_cols = []
        for col in text_cols:
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 50:  # Longer text fields
                long_text_cols.append(col)
        
        if long_text_cols:
            suggestions.append(f"Create text length features for: {', '.join(long_text_cols)}")
        
        # Numeric columns - ratios and interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            suggestions.append("Consider creating ratio features between numeric columns")
        
        # Categorical combinations
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 1:
            suggestions.append("Consider creating interaction features between categorical variables")
        
        if suggestions:
            report['recommendations'] = suggestions

def preprocess_for_analysis(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to preprocess a CSV file for analysis
    
    Returns:
        - Cleaned DataFrame
        - Preprocessing report
    """
    df = pd.read_csv(csv_path)
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_dataset(df) 