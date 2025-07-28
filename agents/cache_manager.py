"""
Caching and Performance Monitoring System
========================================

Provides intelligent caching for analysis results and performance monitoring.
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import threading
from functools import wraps

class CacheManager:
    def __init__(self, cache_dir: str = ".cache", max_cache_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.performance_log = []
        self.lock = threading.Lock()
        
        # Clean up old cache files on initialization
        self._cleanup_old_cache()
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for CSV file content"""
        hasher = hashlib.md5()
        
        # Include file size and modification time for quick check
        stat = os.stat(file_path)
        hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode())
        
        # Include first and last few rows for content verification
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Hash first 5 and last 5 rows
                sample_data = pd.concat([df.head(5), df.tail(5)])
                hasher.update(sample_data.to_string().encode())
        except:
            # Fallback to file path if reading fails
            hasher.update(file_path.encode())
        
        return hasher.hexdigest()
    
    def get_cache_key(self, csv_path: str, agent_type: str, additional_params: Dict = None) -> str:
        """Generate cache key for analysis results"""
        file_hash = self.get_file_hash(csv_path)
        
        key_components = [file_hash, agent_type]
        if additional_params:
            key_components.append(json.dumps(additional_params, sort_keys=True))
        
        return hashlib.md5("_".join(key_components).encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis result"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired (24 hours)
            if time.time() - cache_file.stat().st_mtime > 24 * 3600:
                cache_file.unlink()  # Delete expired cache
                return None
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Update access time
            cache_file.touch()
            
            self._log_performance("cache_hit", cache_key)
            return cached_data
            
        except Exception as e:
            print(f"Error reading cache: {e}")
            return None
    
    def cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result"""
        try:
            # Check cache size limits
            self._enforce_cache_size_limit()
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            self._log_performance("cache_store", cache_key)
            
        except Exception as e:
            print(f"Error caching result: {e}")
    
    def _enforce_cache_size_limit(self):
        """Remove oldest cache files if size limit exceeded"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        
        if total_size > self.max_cache_size:
            # Get all cache files sorted by access time
            cache_files = sorted(
                self.cache_dir.glob("*.pkl"),
                key=lambda f: f.stat().st_atime
            )
            
            # Remove oldest files until under limit
            for cache_file in cache_files:
                cache_file.unlink()
                total_size -= cache_file.stat().st_size
                
                if total_size <= self.max_cache_size * 0.8:  # Leave some buffer
                    break
    
    def _cleanup_old_cache(self):
        """Remove cache files older than 7 days"""
        cutoff_time = time.time() - 7 * 24 * 3600  # 7 days ago
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
    
    def _log_performance(self, action: str, cache_key: str):
        """Log performance metrics"""
        with self.lock:
            self.performance_log.append({
                'timestamp': datetime.now(),
                'action': action,
                'cache_key': cache_key[:8]  # Only first 8 chars for privacy
            })
            
            # Keep only last 1000 entries
            if len(self.performance_log) > 1000:
                self.performance_log = self.performance_log[-1000:]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Analyze performance log
        recent_log = [
            entry for entry in self.performance_log 
            if entry['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        hits = len([e for e in recent_log if e['action'] == 'cache_hit'])
        stores = len([e for e in recent_log if e['action'] == 'cache_store'])
        
        return {
            'cache_files_count': len(cache_files),
            'total_cache_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_hit_rate': round(hits / max(hits + stores, 1) * 100, 1),
            'recent_hits': hits,
            'recent_stores': stores
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


# Global cache manager instance
cache_manager = CacheManager()


def cached_analysis(agent_type: str, additional_params: Dict = None):
    """Decorator for caching analysis results"""
    def decorator(func):
        @wraps(func)
        def wrapper(csv_path: str, *args, **kwargs):
            # Generate cache key
            params = additional_params or {}
            if args:
                params['args'] = str(args)
            if kwargs:
                params['kwargs'] = str(kwargs)
            
            cache_key = cache_manager.get_cache_key(csv_path, agent_type, params)
            
            # Try to get cached result
            cached_result = cache_manager.get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Run analysis and cache result
            start_time = time.time()
            result = func(csv_path, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Add performance metadata
            if isinstance(result, dict):
                result['_cache_info'] = {
                    'execution_time': round(execution_time, 2),
                    'cached': False,
                    'cache_key': cache_key[:8]
                }
            
            # Cache the result
            cache_manager.cache_result(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
    
    def track_execution(self, operation_name: str):
        """Context manager for tracking operation execution time"""
        return ExecutionTracker(self, operation_name)
    
    def log_metric(self, operation: str, duration: float, success: bool = True, metadata: Dict = None):
        """Log performance metric"""
        with self.lock:
            self.metrics.append({
                'timestamp': datetime.now(),
                'operation': operation,
                'duration': duration,
                'success': success,
                'metadata': metadata or {}
            })
            
            # Keep only last 10000 metrics
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-10000:]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {'message': 'No recent metrics available'}
        
        # Group by operation
        by_operation = {}
        for metric in recent_metrics:
            op = metric['operation']
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(metric)
        
        summary = {}
        for operation, metrics in by_operation.items():
            durations = [m['duration'] for m in metrics if m['success']]
            success_count = len([m for m in metrics if m['success']])
            total_count = len(metrics)
            
            if durations:
                summary[operation] = {
                    'avg_duration': round(sum(durations) / len(durations), 2),
                    'min_duration': round(min(durations), 2),
                    'max_duration': round(max(durations), 2),
                    'success_rate': round(success_count / total_count * 100, 1),
                    'total_executions': total_count
                }
        
        return summary


class ExecutionTracker:
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        metadata = {}
        if not success:
            metadata['error_type'] = exc_type.__name__ if exc_type else 'Unknown'
        
        self.monitor.log_metric(self.operation_name, duration, success, metadata)


# Global performance monitor
performance_monitor = PerformanceMonitor() 