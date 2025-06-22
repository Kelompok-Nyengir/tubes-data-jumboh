import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CreditCardVisualizer:
    """
    Comprehensive visualization class for credit card default analysis
    Creates research-grade visualizations and interactive dashboards
    """

    def __init__(self, figsize=(12, 8), style='seaborn-v0_8'):
        """Initialize visualizer with styling"""
        plt.style.use(style)
        sns.set_palette("Set2")
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#85DCBA',
            'excellent': '#2E8B57',
            'good': '#32CD32',
            'fair': '#FFD700',
            'poor': '#FF6347',
            'deteriorating': '#DC143C'
        }

    def convert_spark_to_pandas(self, df_spark, sample_size=None, seed=42):
        """Convert Spark DataFrame to pandas for visualization"""
        try:
            if sample_size and df_spark.count() > sample_size:
                df_pandas = df_spark.sample(sample_size / df_spark.count(), seed=seed).toPandas()
                logger.info(f"Sampled {len(df_pandas):,} rows for visualization")
            else:
                df_pandas = df_spark.toPandas()
                logger.info(f"Converted {len(df_pandas):,} rows to pandas")

            return df_pandas

        except Exception as e:
            logger.error(f"Error converting to pandas: {e}")
            raise

    def create_research_overview_dashboard(self, df_pandas: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive research variable overview dashboard"""
        logger.info("Creating research overview dashboard...")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

        # 1. Target Variable Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        target_counts = df_pandas['default payment next month'].value_counts()
        colors = [self.colors['info'], self.colors['warning']]
        wedges, texts, autotexts = ax1.pie(target_counts.values,
                                          labels=['No Default', 'Default'],
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax1.set_title('Target Distribution\n(Default Payment Next Month)', fontweight='bold')

        # 2. Gender Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        gender_counts = df_pandas['SEX'].map({1: 'Male', 2: 'Female'}).value_counts()
        bars = ax2.bar(gender_counts.index, gender_counts.values,
                      color=[self.colors['primary'], self.colors['secondary']])
        ax2.set_title('Gender Distribution (X2)', fontweight='bold')
        ax2.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:,}', ha='center', va='bottom')

        # 3. Education Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        edu_mapping = {1: 'Grad School', 2: 'University', 3: 'High School', 4: 'Others'}
        edu_counts = df_pandas['EDUCATION'].map(edu_mapping).value_counts()
        bars = ax3.bar(edu_counts.index, edu_counts.values, color=self.colors['success'])
        ax3.set_title('Education Distribution (X3)', fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:,}', ha='center', va='bottom')

        # 4. Age Distribution
        ax4 = fig.add_subplot(gs[0, 3])
        n, bins, patches = ax4.hist(df_pandas['AGE'], bins=25, color=self.colors['info'],
                                   alpha=0.7, edgecolor='black')
        ax4.set_title('Age Distribution (X5)', fontweight='bold')
        ax4.set_xlabel('Age (years)')
        ax4.set_ylabel('Frequency')
        ax4.axvline(df_pandas['AGE'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_pandas["AGE"].mean():.1f}')
        ax4.legend()

        # 5. Payment Status Correlation Heatmap
        ax5 = fig.add_subplot(gs[1, :2])
        pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
        months = ['Apr 2005', 'May 2005', 'Jun 2005', 'Jul 2005', 'Aug 2005', 'Sep 2005']

        if all(col in df_pandas.columns for col in pay_cols):
            pay_corr = df_pandas[pay_cols].corr()
            pay_corr.index = months
            pay_corr.columns = months

            sns.heatmap(pay_corr, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax5,
                       cbar_kws={'label': 'Correlation Coefficient'}, square=True)
            ax5.set_title('Payment Status Correlation Across Months (X6-X11)', fontweight='bold')

        # 6. Temporal Bill vs Payment Trends
        ax6 = fig.add_subplot(gs[1, 2:])
        bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
        pay_amt_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']

        if all(col in df_pandas.columns for col in bill_cols + pay_amt_cols):
            avg_bills = df_pandas[bill_cols].mean() / 1000  # Convert to thousands
            avg_payments = df_pandas[pay_amt_cols].mean() / 1000

            ax6.plot(months, avg_bills.values, marker='s', label='Avg Bill Amount (X12-X17)',
                    linewidth=3, markersize=8, color=self.colors['warning'])
            ax6.plot(months, avg_payments.values, marker='o', label='Avg Payment Amount (X18-X23)',
                    linewidth=3, markersize=8, color=self.colors['success'])
            ax6.set_title('Bill vs Payment Amounts Over Time', fontweight='bold')
            ax6.set_ylabel('Amount (NT$ thousands)')
            ax6.set_xlabel('Month')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45)

        # 7. Risk Score Distribution (if available)
        if 'RISK_SCORE_CATEGORY' in df_pandas.columns:
            ax7 = fig.add_subplot(gs[2, :2])
            risk_dist = df_pandas['RISK_SCORE_CATEGORY'].value_counts()
            risk_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            risk_dist = risk_dist.reindex([r for r in risk_order if r in risk_dist.index])

            bars = ax7.bar(risk_dist.index, risk_dist.values,
                          color=[self.colors['excellent'], self.colors['good'],
                                self.colors['fair'], self.colors['poor'], self.colors['deteriorating']])
            ax7.set_title('Temporal Risk Score Distribution', fontweight='bold')
            ax7.set_ylabel('Count')
            ax7.tick_params(axis='x', rotation=45)

            for bar in bars:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:,}', ha='center', va='bottom')

        # 8. Payment Behavior vs Default Rate
        if 'PAYMENT_BEHAVIOR_TYPE' in df_pandas.columns:
            ax8 = fig.add_subplot(gs[2, 2:])
            behavior_default = df_pandas.groupby('PAYMENT_BEHAVIOR_TYPE')['default payment next month'].agg(['mean', 'count'])
            behavior_default['default_rate'] = behavior_default['mean'] * 100

            bars = ax8.bar(behavior_default.index, behavior_default['default_rate'],
                          color=[self.colors['excellent'], self.colors['deteriorating'],
                                self.colors['fair'], self.colors['good'],
                                self.colors['poor'], self.colors['success']][:len(behavior_default)])
            ax8.set_title('Default Rate by Payment Behavior Type', fontweight='bold')
            ax8.set_ylabel('Default Rate (%)')
            ax8.tick_params(axis='x', rotation=45)

            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = behavior_default.iloc[i]['count']
                ax8.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f}%\n(n={count:,})', ha='center', va='bottom')

        # 9. Feature Correlation Matrix
        ax9 = fig.add_subplot(gs[3, :])
        key_features = ['LIMIT_BAL', 'AGE', 'default payment next month']

        # Add temporal features if available
        temporal_features = ['RECENT_AVG_DELAY', 'CREDIT_UTILIZATION_RATIO', 'TEMPORAL_RISK_SCORE',
                           'PAYMENT_IMPROVEMENT_SCORE', 'AVG_PAYMENT_EFFICIENCY']
        available_temporal = [f for f in temporal_features if f in df_pandas.columns]
        key_features.extend(available_temporal)

        available_features = [f for f in key_features if f in df_pandas.columns]

        if len(available_features) > 3:
            corr_matrix = df_pandas[available_features].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, ax=ax9, square=True, linewidths=0.5,
                       cbar_kws={'label': 'Correlation Coefficient'})
            ax9.set_title('Key Features Correlation Matrix', fontweight='bold')

        plt.suptitle('Enhanced Credit Card Default Analysis - Research Variable Overview\n' +
                    f'Analysis Date: 2025-06-20 15:52:57 UTC | Analyst: ardzz',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Research overview dashboard saved to: {save_path}")

        plt.show()

        return fig

    def create_temporal_analysis_dashboard(self, df_pandas: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive temporal analysis dashboard"""
        logger.info("Creating temporal analysis dashboard...")

        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('Temporal Analysis Dashboard - 6-Month Payment History Insights\n' +
                    f'Analysis Date: 2025-06-20 15:52:57 UTC | Analyst: ardzz',
                    fontsize=16, fontweight='bold')

        pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
        months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

        # 1. Payment Status Trends
        if all(col in df_pandas.columns for col in pay_cols):
            pay_status_means = df_pandas[pay_cols].mean()
            pay_status_means.index = months

            axes[0,0].plot(months, pay_status_means.values, marker='o', linewidth=3,
                          markersize=10, color=self.colors['warning'])
            axes[0,0].fill_between(months, pay_status_means.values, alpha=0.3,
                                  color=self.colors['warning'])
            axes[0,0].set_title('Average Payment Delay by Month (X6-X11)', fontweight='bold')
            axes[0,0].set_ylabel('Average Payment Status')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_ylim(bottom=0)

        # 2. Payment Improvement Distribution
        if 'PAYMENT_IMPROVEMENT_SCORE' in df_pandas.columns:
            n, bins, patches = axes[0,1].hist(df_pandas['PAYMENT_IMPROVEMENT_SCORE'],
                                             bins=40, alpha=0.7, color=self.colors['primary'],
                                             edgecolor='black')
            axes[0,1].set_title('Payment Improvement Score Distribution', fontweight='bold')
            axes[0,1].set_xlabel('Improvement Score (Historical - Recent)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
            axes[0,1].axvline(df_pandas['PAYMENT_IMPROVEMENT_SCORE'].mean(),
                             color='green', linestyle=':', linewidth=2,
                             label=f'Mean: {df_pandas["PAYMENT_IMPROVEMENT_SCORE"].mean():.2f}')
            axes[0,1].legend()

        # 3. Credit Utilization vs Default
        if 'CREDIT_UTILIZATION_RATIO' in df_pandas.columns:
            default_0 = df_pandas[df_pandas['default payment next month'] == 0]['CREDIT_UTILIZATION_RATIO']
            default_1 = df_pandas[df_pandas['default payment next month'] == 1]['CREDIT_UTILIZATION_RATIO']

            axes[1,0].hist([default_0, default_1], bins=30, alpha=0.7,
                          label=['No Default', 'Default'],
                          color=[self.colors['success'], self.colors['warning']])
            axes[1,0].set_title('Credit Utilization vs Default Status', fontweight='bold')
            axes[1,0].set_xlabel('Credit Utilization Ratio')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].legend()

        # 4. Payment Efficiency Analysis
        if 'AVG_PAYMENT_EFFICIENCY' in df_pandas.columns:
            efficiency_default = df_pandas[df_pandas['default payment next month'] == 1]['AVG_PAYMENT_EFFICIENCY']
            efficiency_no_default = df_pandas[df_pandas['default payment next month'] == 0]['AVG_PAYMENT_EFFICIENCY']

            axes[1,1].hist([efficiency_no_default, efficiency_default], bins=30, alpha=0.7,
                          label=['No Default', 'Default'],
                          color=[self.colors['success'], self.colors['warning']])
            axes[1,1].set_title('Payment Efficiency vs Default Status', fontweight='bold')
            axes[1,1].set_xlabel('Average Payment Efficiency')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()

        # 5. Risk Score Distribution by Default
        if 'TEMPORAL_RISK_SCORE' in df_pandas.columns:
            default_scores = df_pandas[df_pandas['default payment next month'] == 1]['TEMPORAL_RISK_SCORE']
            no_default_scores = df_pandas[df_pandas['default payment next month'] == 0]['TEMPORAL_RISK_SCORE']

            axes[2,0].hist([no_default_scores, default_scores], bins=30, alpha=0.7,
                          label=['No Default', 'Default'],
                          color=[self.colors['success'], self.colors['warning']])
            axes[2,0].set_title('Temporal Risk Score by Default Status', fontweight='bold')
            axes[2,0].set_xlabel('Temporal Risk Score')
            axes[2,0].set_ylabel('Frequency')
            axes[2,0].legend()

        # 6. Customer Segment Analysis
        if 'CUSTOMER_SEGMENT' in df_pandas.columns:
            segment_analysis = df_pandas.groupby('CUSTOMER_SEGMENT').agg({
                'default payment next month': ['mean', 'count']
            }).round(4)
            segment_analysis.columns = ['default_rate', 'count']
            segment_analysis['default_rate'] *= 100

            bars = axes[2,1].bar(segment_analysis.index, segment_analysis['default_rate'],
                               color=self.colors['secondary'], alpha=0.8)
            axes[2,1].set_title('Default Rate by Customer Segment', fontweight='bold')
            axes[2,1].set_ylabel('Default Rate (%)')
            axes[2,1].tick_params(axis='x', rotation=45)

            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = segment_analysis.iloc[i]['count']
                axes[2,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                              f'{height:.1f}%\n(n={count:,})', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Temporal analysis dashboard saved to: {save_path}")

        plt.show()

        return fig

    def create_interactive_plotly_dashboard(self, df_pandas: pd.DataFrame, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard"""
        logger.info("Creating interactive Plotly dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Payment Status Over Time', 'Risk Score Distribution',
                           'Credit Utilization Analysis', 'Payment Behavior Analysis',
                           'Default Rate by Demographics', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )

        # 1. Payment Status Over Time
        months = ['Apr 2005', 'May 2005', 'Jun 2005', 'Jul 2005', 'Aug 2005', 'Sep 2005']
        pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']

        if all(col in df_pandas.columns for col in pay_cols):
            avg_payment_status = df_pandas[pay_cols].mean()

            fig.add_trace(
                go.Scatter(x=months, y=avg_payment_status.values,
                          mode='lines+markers', name='Avg Payment Status',
                          line=dict(width=3), marker=dict(size=8)),
                row=1, col=1
            )

        # 2. Risk Score Distribution
        if 'TEMPORAL_RISK_SCORE' in df_pandas.columns:
            fig.add_trace(
                go.Histogram(x=df_pandas['TEMPORAL_RISK_SCORE'],
                           name='Risk Score Distribution',
                           opacity=0.7),
                row=1, col=2
            )

        # 3. Credit Utilization Scatter
        if 'CREDIT_UTILIZATION_RATIO' in df_pandas.columns:
            colors = df_pandas['default payment next month'].map({0: 'green', 1: 'red'})

            fig.add_trace(
                go.Scatter(x=df_pandas['CREDIT_UTILIZATION_RATIO'],
                          y=df_pandas['LIMIT_BAL'],
                          mode='markers',
                          marker=dict(color=colors, opacity=0.6),
                          name='Credit Utilization',
                          text=df_pandas['default payment next month'].map({0: 'No Default', 1: 'Default'})),
                row=2, col=1
            )

        # 4. Payment Behavior Analysis
        if 'PAYMENT_BEHAVIOR_TYPE' in df_pandas.columns:
            behavior_counts = df_pandas['PAYMENT_BEHAVIOR_TYPE'].value_counts()

            fig.add_trace(
                go.Bar(x=behavior_counts.index, y=behavior_counts.values,
                      name='Payment Behavior Distribution'),
                row=2, col=2
            )

        # 5. Default Rate by Education
        edu_mapping = {1: 'Grad School', 2: 'University', 3: 'High School', 4: 'Others'}
        if 'EDUCATION' in df_pandas.columns:
            edu_default = df_pandas.groupby('EDUCATION')['default payment next month'].mean() * 100
            edu_labels = [edu_mapping.get(edu, f'Code {edu}') for edu in edu_default.index]

            fig.add_trace(
                go.Bar(x=edu_labels, y=edu_default.values,
                      name='Default Rate by Education'),
                row=3, col=1
            )

        # 6. Correlation Heatmap
        numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 numeric columns
        if len(numeric_cols) > 2:
            corr_matrix = df_pandas[numeric_cols].corr()

            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          colorscale='RdYlBu',
                          name='Correlation Matrix'),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Interactive Credit Card Default Analysis Dashboard<br>" +
                      f"<sub>Analysis Date: 2025-06-20 15:52:57 UTC | Analyst: ardzz</sub>",
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to: {save_path}")

        fig.show()

        return fig

    def create_model_performance_visualization(self, model_results: Dict, save_path: Optional[str] = None):
        """Create model performance comparison visualization"""
        logger.info("Creating model performance visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Machine Learning Model Performance Comparison\n' +
                    f'Analysis Date: 2025-06-20 15:52:57 UTC | Analyst: ardzz',
                    fontsize=16, fontweight='bold')

        # Prepare data
        models = list(model_results.keys())
        metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']

        # 1. Overall Performance Comparison
        x = np.arange(len(models))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            axes[0,0].bar(x + i*width, values, width, label=metric, alpha=0.8)

        axes[0,0].set_title('Overall Model Performance Comparison', fontweight='bold')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_xticks(x + width * 2)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. AUC Comparison
        auc_scores = [model_results[model]['AUC'] for model in models]
        bars = axes[0,1].bar(models, auc_scores, color=[self.colors['primary'], self.colors['secondary'], self.colors['success']])
        axes[0,1].set_title('AUC Score Comparison', fontweight='bold')
        axes[0,1].set_ylabel('AUC Score')
        axes[0,1].set_ylim(0.7, 1.0)

        for i, (bar, score) in enumerate(zip(bars, auc_scores)):
            axes[0,1].text(bar.get_x() + bar.get_width()/2., score + 0.01,
                          f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        # 3. Precision vs Recall
        precision_scores = [model_results[model]['Precision'] for model in models]
        recall_scores = [model_results[model]['Recall'] for model in models]

        for i, model in enumerate(models):
            axes[1,0].scatter(recall_scores[i], precision_scores[i],
                            s=200, alpha=0.7, label=model)
            axes[1,0].annotate(model, (recall_scores[i], precision_scores[i]),
                             xytext=(5, 5), textcoords='offset points')

        axes[1,0].set_title('Precision vs Recall Trade-off', fontweight='bold')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()

        # 4. F1 Score Ranking
        f1_scores = [model_results[model]['F1'] for model in models]
        model_f1_pairs = list(zip(models, f1_scores))
        model_f1_pairs.sort(key=lambda x: x[1], reverse=True)

        sorted_models, sorted_f1 = zip(*model_f1_pairs)
        colors_ranked = [self.colors['excellent'], self.colors['good'], self.colors['fair']]

        bars = axes[1,1].barh(sorted_models, sorted_f1, color=colors_ranked)
        axes[1,1].set_title('F1 Score Ranking', fontweight='bold')
        axes[1,1].set_xlabel('F1 Score')

        for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
            axes[1,1].text(score + 0.005, bar.get_y() + bar.get_height()/2.,
                          f'{score:.4f}', va='center', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance visualization saved to: {save_path}")

        plt.show()

        return fig

    def save_all_visualizations(self, df_pandas: pd.DataFrame, model_results: Optional[Dict] = None,
                               output_dir: str = "outputs/figures"):
        """Save all visualizations to specified directory"""
        logger.info(f"Saving all visualizations to: {output_dir}")

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save static visualizations
        self.create_research_overview_dashboard(df_pandas, f"{output_dir}/research_overview_dashboard.png")
        self.create_temporal_analysis_dashboard(df_pandas, f"{output_dir}/temporal_analysis_dashboard.png")

        # Save interactive dashboard
        self.create_interactive_plotly_dashboard(df_pandas, f"{output_dir}/interactive_dashboard.html")

        # Save model performance if available
        if model_results:
            self.create_model_performance_visualization(model_results, f"{output_dir}/model_performance_comparison.png")

        logger.info("✅ All visualizations saved successfully")

# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = CreditCardVisualizer()

    # Example with dummy data
    print("✅ Credit Card Visualization Module initialized successfully")
    print("📊 Available visualization methods:")
    print("   - create_research_overview_dashboard()")
    print("   - create_temporal_analysis_dashboard()")
    print("   - create_interactive_plotly_dashboard()")
    print("   - create_model_performance_visualization()")
    print("   - save_all_visualizations()")