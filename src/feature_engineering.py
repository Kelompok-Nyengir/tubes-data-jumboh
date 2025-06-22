from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from functools import reduce
from operator import add
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Advanced temporal feature engineering class for credit card default analysis
    Creates sophisticated features from 6-month payment history and behavioral patterns
    """

    def __init__(self, spark_session=None):
        """Initialize with optional Spark session"""
        if spark_session:
            self.spark = spark_session
        else:
            self.spark = SparkSession.getActiveSession()

        if not self.spark:
            raise ValueError("No active Spark session found. Please provide a Spark session.")

        logger.info("✅ TemporalFeatureEngineer initialized successfully")

    def _safe_column_sum(self, columns_list):
        """Safely sum a list of columns using reduce and add operator"""
        if not columns_list:
            return lit(0.0)
        if len(columns_list) == 1:
            return col(columns_list[0])
        return reduce(add, [col(col_name) for col_name in columns_list])

    def _safe_column_mean(self, columns_list):
        """Safely calculate mean of a list of columns"""
        if not columns_list:
            return lit(0.0)
        column_sum = self._safe_column_sum(columns_list)
        return column_sum / len(columns_list)

    def _safe_variance_calculation(self, columns_list, mean_col):
        """Safely calculate variance for a list of columns"""
        if not columns_list:
            return lit(0.0)
        if len(columns_list) == 1:
            return lit(0.0)

        squared_diffs = [pow(col(col_name) - mean_col, 2) for col_name in columns_list]
        variance_sum = reduce(add, squared_diffs)
        return variance_sum / len(columns_list)

    def create_payment_trend_features(self, df):
        """
        Create payment trend and volatility features from 6-month payment history
        """
        logger.info("Creating payment trend and volatility features...")

        df_trends = df

        # Payment status columns (PAY_0 is most recent, PAY_6 is oldest)
        pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']

        # Check if payment columns exist
        available_pay_cols = [col_name for col_name in pay_cols if col_name in df.columns]

        if len(available_pay_cols) >= 3:
            # Create payment trend slope using linear regression approach
            # Calculate the trend of payment delays over time
            df_trends = df_trends.withColumn(
                "PAYMENT_TREND_SLOPE",
                # Simple linear trend calculation
                (col('PAY_0') - col('PAY_6')) / 6.0
            )

            # Calculate payment status volatility (standard deviation)
            # Create array of payment statuses for volatility calculation
            pay_array = array(*[col(pay_col) for pay_col in available_pay_cols])

            # Custom volatility calculation using available payments
            payment_mean = self._safe_column_mean(available_pay_cols)

            df_trends = df_trends.withColumn(
                "payment_array", pay_array
            ).withColumn(
                "payment_mean", payment_mean
            )

            # Calculate volatility using safe variance calculation
            variance = self._safe_variance_calculation(available_pay_cols, col("payment_mean"))

            df_trends = df_trends.withColumn(
                "PAYMENT_STATUS_VOLATILITY",
                sqrt(variance)
            )

            # Maximum and minimum payment delays
            df_trends = df_trends.withColumn(
                "MAX_PAYMENT_DELAY",
                greatest(*[col(pay_col) for pay_col in available_pay_cols])
            ).withColumn(
                "MIN_PAYMENT_DELAY",
                least(*[col(pay_col) for pay_col in available_pay_cols])
            )

            # Payment delay range
            df_trends = df_trends.withColumn(
                "PAYMENT_DELAY_RANGE",
                col("MAX_PAYMENT_DELAY") - col("MIN_PAYMENT_DELAY")
            )

            # Drop temporary columns
            df_trends = df_trends.drop("payment_array", "payment_mean")

        else:
            logger.warning(f"Insufficient payment columns found. Available: {available_pay_cols}")
            # Create default values if insufficient data
            df_trends = df_trends.withColumn("PAYMENT_TREND_SLOPE", lit(0.0)) \
                .withColumn("PAYMENT_STATUS_VOLATILITY", lit(0.0)) \
                .withColumn("MAX_PAYMENT_DELAY", lit(0)) \
                .withColumn("MIN_PAYMENT_DELAY", lit(0)) \
                .withColumn("PAYMENT_DELAY_RANGE", lit(0))

        logger.info("✅ Payment trend and volatility features created")
        return df_trends

    def create_temporal_segmentation_features(self, df):
        """
        Create temporal segmentation features (recent vs historical patterns)
        """
        logger.info("Creating temporal segmentation features...")

        df_temporal = df

        # Recent payments (last 3 months): PAY_0, PAY_2, PAY_3
        recent_cols = ['PAY_0', 'PAY_2', 'PAY_3']
        # Historical payments (older 3 months): PAY_4, PAY_5, PAY_6
        historical_cols = ['PAY_4', 'PAY_5', 'PAY_6']

        available_recent = [col_name for col_name in recent_cols if col_name in df.columns]
        available_historical = [col_name for col_name in historical_cols if col_name in df.columns]

        if available_recent and available_historical:
            # Recent average delay
            df_temporal = df_temporal.withColumn(
                "RECENT_AVG_DELAY",
                self._safe_column_mean(available_recent)
            )

            # Historical average delay
            df_temporal = df_temporal.withColumn(
                "HISTORICAL_AVG_DELAY",
                self._safe_column_mean(available_historical)
            )

            # Payment improvement score (negative means improvement)
            df_temporal = df_temporal.withColumn(
                "PAYMENT_IMPROVEMENT_SCORE",
                col("HISTORICAL_AVG_DELAY") - col("RECENT_AVG_DELAY")
            )

        else:
            logger.warning("Insufficient payment columns for temporal segmentation")
            df_temporal = df_temporal.withColumn("RECENT_AVG_DELAY", lit(0.0)) \
                .withColumn("HISTORICAL_AVG_DELAY", lit(0.0)) \
                .withColumn("PAYMENT_IMPROVEMENT_SCORE", lit(0.0))

        # Recovery instances (count of times payment status improved)
        pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
        available_pay_cols = [col_name for col_name in pay_cols if col_name in df.columns]

        if len(available_pay_cols) >= 2:
            recovery_conditions = []
            for i in range(len(available_pay_cols) - 1):
                # Recovery when payment status decreases (improvement)
                recovery_conditions.append(
                    when(col(available_pay_cols[i]) > col(available_pay_cols[i + 1]), 1).otherwise(0)
                )

            if recovery_conditions:
                df_temporal = df_temporal.withColumn(
                    "RECOVERY_INSTANCES",
                    reduce(add, recovery_conditions)
                )
            else:
                df_temporal = df_temporal.withColumn("RECOVERY_INSTANCES", lit(0))
        else:
            df_temporal = df_temporal.withColumn("RECOVERY_INSTANCES", lit(0))

        logger.info("✅ Temporal segmentation features created")
        return df_temporal

    def create_bill_statement_features(self, df):
        """
        Create bill statement and financial trend features
        """
        logger.info("Creating bill statement and financial features...")

        df_bills = df

        # Bill amount columns (BILL_AMT1 is most recent, BILL_AMT6 is oldest)
        bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
        available_bill_cols = [col_name for col_name in bill_cols if col_name in df.columns]

        if len(available_bill_cols) >= 3:
            # Bill trend slope (increasing/decreasing bills)
            df_bills = df_bills.withColumn(
                "BILL_TREND_SLOPE",
                (col('BILL_AMT1') - col('BILL_AMT6')) / 6.0
            )

            # Bill amount volatility
            bill_mean = self._safe_column_mean(available_bill_cols)
            df_bills = df_bills.withColumn("bill_mean_temp", bill_mean)

            # Calculate bill volatility
            bill_variance = self._safe_variance_calculation(available_bill_cols, col("bill_mean_temp"))
            df_bills = df_bills.withColumn(
                "BILL_AMOUNT_VOLATILITY",
                sqrt(bill_variance)
            )

            # Average bill amount
            df_bills = df_bills.withColumn(
                "AVG_BILL_AMOUNT",
                bill_mean
            )

            # Debt accumulation rate (recent vs historical)
            recent_bills = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']
            historical_bills = ['BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

            available_recent_bills = [col_name for col_name in recent_bills if col_name in df.columns]
            available_historical_bills = [col_name for col_name in historical_bills if col_name in df.columns]

            if available_recent_bills and available_historical_bills:
                recent_avg = self._safe_column_mean(available_recent_bills)
                historical_avg = self._safe_column_mean(available_historical_bills)

                df_bills = df_bills.withColumn(
                    "DEBT_ACCUMULATION_RATE",
                    when(col("bill_mean_temp") > 0,
                         (recent_avg - historical_avg) / (historical_avg + 1)).otherwise(0.0)
                )
            else:
                df_bills = df_bills.withColumn("DEBT_ACCUMULATION_RATE", lit(0.0))

            # Drop temporary column
            df_bills = df_bills.drop("bill_mean_temp")

        else:
            logger.warning(f"Insufficient bill columns found. Available: {available_bill_cols}")
            df_bills = df_bills.withColumn("BILL_TREND_SLOPE", lit(0.0)) \
                .withColumn("BILL_AMOUNT_VOLATILITY", lit(0.0)) \
                .withColumn("AVG_BILL_AMOUNT", lit(0.0)) \
                .withColumn("DEBT_ACCUMULATION_RATE", lit(0.0))

        logger.info("✅ Bill statement and financial features created")
        return df_bills

    def create_payment_efficiency_features(self, df):
        """
        Create payment efficiency and consistency features
        """
        logger.info("Creating payment efficiency and consistency features...")

        df_efficiency = df

        # Payment amount columns
        pay_amt_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
        bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']

        available_pay_amt_cols = [col_name for col_name in pay_amt_cols if col_name in df.columns]
        available_bill_cols = [col_name for col_name in bill_cols if col_name in df.columns]

        if available_pay_amt_cols and available_bill_cols:
            # Average payment amount
            df_efficiency = df_efficiency.withColumn(
                "AVG_PAYMENT_AMOUNT",
                self._safe_column_mean(available_pay_amt_cols)
            )

            # Payment efficiency ratio (payment amount / bill amount)
            efficiency_ratios = []
            for pay_col, bill_col in zip(available_pay_amt_cols, available_bill_cols):
                if pay_col in df.columns and bill_col in df.columns:
                    efficiency_ratios.append(
                        when(col(bill_col) > 0, col(pay_col) / col(bill_col)).otherwise(0.0)
                    )

            if efficiency_ratios:
                if len(efficiency_ratios) == 1:
                    avg_efficiency = efficiency_ratios[0]
                else:
                    avg_efficiency = reduce(add, efficiency_ratios) / len(efficiency_ratios)

                df_efficiency = df_efficiency.withColumn(
                    "AVG_PAYMENT_EFFICIENCY",
                    avg_efficiency
                )

                # Payment efficiency trend
                if len(efficiency_ratios) >= 2:
                    recent_ratios = efficiency_ratios[:3] if len(efficiency_ratios) >= 3 else efficiency_ratios[:len(
                        efficiency_ratios) // 2]
                    historical_ratios = efficiency_ratios[-3:] if len(efficiency_ratios) >= 3 else efficiency_ratios[
                                                                                                   len(efficiency_ratios) // 2:]

                    if recent_ratios and historical_ratios:
                        recent_efficiency = reduce(add, recent_ratios) / len(recent_ratios) if len(
                            recent_ratios) > 1 else recent_ratios[0]
                        historical_efficiency = reduce(add, historical_ratios) / len(historical_ratios) if len(
                            historical_ratios) > 1 else historical_ratios[0]

                        df_efficiency = df_efficiency.withColumn(
                            "PAYMENT_EFFICIENCY_TREND",
                            recent_efficiency - historical_efficiency
                        )
                    else:
                        df_efficiency = df_efficiency.withColumn("PAYMENT_EFFICIENCY_TREND", lit(0.0))
                else:
                    df_efficiency = df_efficiency.withColumn("PAYMENT_EFFICIENCY_TREND", lit(0.0))

                # Payment consistency score (inverse of payment volatility)
                pay_mean = self._safe_column_mean(available_pay_amt_cols)
                df_efficiency = df_efficiency.withColumn("pay_mean_temp", pay_mean)

                pay_variance = self._safe_variance_calculation(available_pay_amt_cols, col("pay_mean_temp"))
                pay_volatility = sqrt(pay_variance)

                df_efficiency = df_efficiency.withColumn(
                    "PAYMENT_CONSISTENCY_SCORE",
                    when(pay_volatility > 0, 1.0 / (1.0 + pay_volatility)).otherwise(1.0)
                )

                # Drop temporary column
                df_efficiency = df_efficiency.drop("pay_mean_temp")

            else:
                df_efficiency = df_efficiency.withColumn("AVG_PAYMENT_EFFICIENCY", lit(0.0)) \
                    .withColumn("PAYMENT_EFFICIENCY_TREND", lit(0.0)) \
                    .withColumn("PAYMENT_CONSISTENCY_SCORE", lit(0.0))
        else:
            logger.warning("Insufficient payment/bill columns for efficiency calculation")
            df_efficiency = df_efficiency.withColumn("AVG_PAYMENT_AMOUNT", lit(0.0)) \
                .withColumn("AVG_PAYMENT_EFFICIENCY", lit(0.0)) \
                .withColumn("PAYMENT_EFFICIENCY_TREND", lit(0.0)) \
                .withColumn("PAYMENT_CONSISTENCY_SCORE", lit(0.0))

        logger.info("✅ Payment efficiency and consistency features created")
        return df_efficiency

    def create_credit_utilization_features(self, df):
        """
        Create credit utilization and financial capacity features
        """
        logger.info("Creating credit utilization features...")

        df_credit = df

        # Credit utilization ratio (average bill / credit limit)
        if 'AVG_BILL_AMOUNT' in df.columns and 'LIMIT_BAL' in df.columns:
            df_credit = df_credit.withColumn(
                "CREDIT_UTILIZATION_RATIO",
                when(col('LIMIT_BAL') > 0, col('AVG_BILL_AMOUNT') / col('LIMIT_BAL')).otherwise(0.0)
            )
        elif 'LIMIT_BAL' in df.columns:
            # Calculate average bill if not available
            bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            available_bill_cols = [col_name for col_name in bill_cols if col_name in df.columns]

            if available_bill_cols:
                avg_bill = self._safe_column_mean(available_bill_cols)
                df_credit = df_credit.withColumn(
                    "CREDIT_UTILIZATION_RATIO",
                    when(col('LIMIT_BAL') > 0, avg_bill / col('LIMIT_BAL')).otherwise(0.0)
                )
            else:
                df_credit = df_credit.withColumn("CREDIT_UTILIZATION_RATIO", lit(0.0))
        else:
            df_credit = df_credit.withColumn("CREDIT_UTILIZATION_RATIO", lit(0.0))

        # Credit buffer (remaining credit limit)
        if 'LIMIT_BAL' in df.columns and 'AVG_BILL_AMOUNT' in df.columns:
            df_credit = df_credit.withColumn(
                "CREDIT_BUFFER",
                col('LIMIT_BAL') - col('AVG_BILL_AMOUNT')
            )
        elif 'LIMIT_BAL' in df.columns:
            df_credit = df_credit.withColumn("CREDIT_BUFFER", col('LIMIT_BAL'))
        else:
            df_credit = df_credit.withColumn("CREDIT_BUFFER", lit(0.0))

        # Credit utilization trend
        bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
        available_bill_cols = [col_name for col_name in bill_cols if
                               col_name in df.columns and 'LIMIT_BAL' in df.columns]

        if len(available_bill_cols) >= 3 and 'LIMIT_BAL' in df.columns:
            recent_bills = available_bill_cols[:3]
            historical_bills = available_bill_cols[-3:]

            recent_utilization = self._safe_column_mean(recent_bills) / col('LIMIT_BAL')
            historical_utilization = self._safe_column_mean(historical_bills) / col('LIMIT_BAL')

            df_credit = df_credit.withColumn(
                "CREDIT_UTILIZATION_TREND",
                when(col('LIMIT_BAL') > 0, recent_utilization - historical_utilization).otherwise(0.0)
            )
        else:
            df_credit = df_credit.withColumn("CREDIT_UTILIZATION_TREND", lit(0.0))

        logger.info("✅ Credit utilization features created")
        return df_credit

    def create_behavioral_classification_features(self, df):
        """
        Create behavioral classification and customer segmentation features
        """
        logger.info("Creating behavioral classification features...")

        df_behavior = df

        # Payment behavior type classification
        if 'RECENT_AVG_DELAY' in df.columns and 'PAYMENT_STATUS_VOLATILITY' in df.columns:
            df_behavior = df_behavior.withColumn(
                "PAYMENT_BEHAVIOR_TYPE",
                when((col('RECENT_AVG_DELAY') <= 0) & (col('PAYMENT_STATUS_VOLATILITY') <= 1), "Excellent Payer")
                .when((col('RECENT_AVG_DELAY') <= 1) & (col('PAYMENT_STATUS_VOLATILITY') <= 2), "Good Payer")
                .when((col('RECENT_AVG_DELAY') <= 2) & (col('PAYMENT_STATUS_VOLATILITY') <= 3), "Fair Payer")
                .when(col('PAYMENT_STATUS_VOLATILITY') > 3, "Erratic Payer")
                .when(col('RECENT_AVG_DELAY') > 2, "Poor Payer")
                .otherwise("Unclassified")
            )
        else:
            df_behavior = df_behavior.withColumn("PAYMENT_BEHAVIOR_TYPE", lit("Unclassified"))

        # Temporal risk level
        if 'PAYMENT_IMPROVEMENT_SCORE' in df.columns and 'CREDIT_UTILIZATION_RATIO' in df.columns:
            df_behavior = df_behavior.withColumn(
                "TEMPORAL_RISK_LEVEL",
                when((col('PAYMENT_IMPROVEMENT_SCORE') > 1) & (col('CREDIT_UTILIZATION_RATIO') < 0.3), "Very Low")
                .when((col('PAYMENT_IMPROVEMENT_SCORE') > 0) & (col('CREDIT_UTILIZATION_RATIO') < 0.5), "Low")
                .when((col('PAYMENT_IMPROVEMENT_SCORE') > -1) & (col('CREDIT_UTILIZATION_RATIO') < 0.8), "Medium")
                .when((col('PAYMENT_IMPROVEMENT_SCORE') > -2) | (col('CREDIT_UTILIZATION_RATIO') < 0.9), "High")
                .otherwise("Very High")
            )
        else:
            df_behavior = df_behavior.withColumn("TEMPORAL_RISK_LEVEL", lit("Medium"))

        # Customer segment based on multiple factors
        if all(col_name in df.columns for col_name in ['AGE', 'CREDIT_UTILIZATION_RATIO', 'PAYMENT_BEHAVIOR_TYPE']):
            df_behavior = df_behavior.withColumn(
                "CUSTOMER_SEGMENT",
                when((col('AGE') < 30) & (col('CREDIT_UTILIZATION_RATIO') < 0.3), "Young Conservative")
                .when((col('AGE') < 30) & (col('CREDIT_UTILIZATION_RATIO') >= 0.3), "Young Active")
                .when((col('AGE') >= 30) & (col('AGE') < 50) & (col('CREDIT_UTILIZATION_RATIO') < 0.5),
                      "Prime Conservative")
                .when((col('AGE') >= 30) & (col('AGE') < 50) & (col('CREDIT_UTILIZATION_RATIO') >= 0.5), "Prime Active")
                .when((col('AGE') >= 50) & (col('CREDIT_UTILIZATION_RATIO') < 0.4), "Mature Conservative")
                .when((col('AGE') >= 50) & (col('CREDIT_UTILIZATION_RATIO') >= 0.4), "Mature Active")
                .otherwise("Unclassified")
            )
        else:
            df_behavior = df_behavior.withColumn("CUSTOMER_SEGMENT", lit("Unclassified"))

        logger.info("✅ Behavioral classification features created")
        return df_behavior

    def create_risk_scoring_features(self, df):
        """
        Create comprehensive risk scoring features
        """
        logger.info("Creating risk scoring features...")

        df_risk = df

        # Temporal risk score (0-1 scale, higher = more risk)
        risk_components = []

        # Payment delay component
        if 'RECENT_AVG_DELAY' in df.columns:
            risk_components.append(
                when(col('RECENT_AVG_DELAY') <= 0, 0.0)
                .when(col('RECENT_AVG_DELAY') <= 1, 0.2)
                .when(col('RECENT_AVG_DELAY') <= 2, 0.4)
                .when(col('RECENT_AVG_DELAY') <= 3, 0.6)
                .when(col('RECENT_AVG_DELAY') <= 4, 0.8)
                .otherwise(1.0)
            )

        # Credit utilization component
        if 'CREDIT_UTILIZATION_RATIO' in df.columns:
            risk_components.append(
                when(col('CREDIT_UTILIZATION_RATIO') <= 0.3, 0.0)
                .when(col('CREDIT_UTILIZATION_RATIO') <= 0.5, 0.2)
                .when(col('CREDIT_UTILIZATION_RATIO') <= 0.7, 0.4)
                .when(col('CREDIT_UTILIZATION_RATIO') <= 0.9, 0.6)
                .when(col('CREDIT_UTILIZATION_RATIO') <= 1.0, 0.8)
                .otherwise(1.0)
            )

        # Payment efficiency component
        if 'AVG_PAYMENT_EFFICIENCY' in df.columns:
            risk_components.append(
                when(col('AVG_PAYMENT_EFFICIENCY') >= 0.8, 0.0)
                .when(col('AVG_PAYMENT_EFFICIENCY') >= 0.6, 0.2)
                .when(col('AVG_PAYMENT_EFFICIENCY') >= 0.4, 0.4)
                .when(col('AVG_PAYMENT_EFFICIENCY') >= 0.2, 0.6)
                .when(col('AVG_PAYMENT_EFFICIENCY') >= 0.1, 0.8)
                .otherwise(1.0)
            )

        # Payment improvement component
        if 'PAYMENT_IMPROVEMENT_SCORE' in df.columns:
            risk_components.append(
                when(col('PAYMENT_IMPROVEMENT_SCORE') > 1, 0.0)
                .when(col('PAYMENT_IMPROVEMENT_SCORE') > 0, 0.2)
                .when(col('PAYMENT_IMPROVEMENT_SCORE') > -1, 0.4)
                .when(col('PAYMENT_IMPROVEMENT_SCORE') > -2, 0.6)
                .when(col('PAYMENT_IMPROVEMENT_SCORE') > -3, 0.8)
                .otherwise(1.0)
            )

        if risk_components:
            # Calculate weighted average risk score
            if len(risk_components) == 1:
                risk_score = risk_components[0]
            else:
                risk_score = reduce(add, risk_components) / len(risk_components)

            df_risk = df_risk.withColumn(
                "TEMPORAL_RISK_SCORE",
                risk_score
            )

            # Risk score categories
            df_risk = df_risk.withColumn(
                "RISK_SCORE_CATEGORY",
                when(col('TEMPORAL_RISK_SCORE') <= 0.2, "Very Low")
                .when(col('TEMPORAL_RISK_SCORE') <= 0.4, "Low")
                .when(col('TEMPORAL_RISK_SCORE') <= 0.6, "Medium")
                .when(col('TEMPORAL_RISK_SCORE') <= 0.8, "High")
                .otherwise("Very High")
            )
        else:
            df_risk = df_risk.withColumn("TEMPORAL_RISK_SCORE", lit(0.5)) \
                .withColumn("RISK_SCORE_CATEGORY", lit("Medium"))

        logger.info("✅ Risk scoring features created")
        return df_risk

    def create_all_features(self, df):
        """
        Create all temporal features in the correct order
        """
        logger.info("🚀 Creating all temporal features...")

        # Apply feature engineering phases in order
        df_enhanced = self.create_payment_trend_features(df)
        df_enhanced = self.create_temporal_segmentation_features(df_enhanced)
        df_enhanced = self.create_bill_statement_features(df_enhanced)
        df_enhanced = self.create_payment_efficiency_features(df_enhanced)
        df_enhanced = self.create_credit_utilization_features(df_enhanced)
        df_enhanced = self.create_behavioral_classification_features(df_enhanced)
        df_enhanced = self.create_risk_scoring_features(df_enhanced)

        # Count new features
        original_cols = len(df.columns)
        enhanced_cols = len(df_enhanced.columns)
        new_features = enhanced_cols - original_cols

        logger.info(f"✅ All temporal features created successfully")
        logger.info(f"   Original columns: {original_cols}")
        logger.info(f"   Enhanced columns: {enhanced_cols}")
        logger.info(f"   New features: {new_features}")

        return df_enhanced

    def get_feature_summary(self, df):
        """
        Get summary of all created features
        """
        feature_categories = {
            'Payment Trends': ['PAYMENT_TREND_SLOPE', 'PAYMENT_STATUS_VOLATILITY',
                               'MAX_PAYMENT_DELAY', 'MIN_PAYMENT_DELAY', 'PAYMENT_DELAY_RANGE'],
            'Temporal Segmentation': ['RECENT_AVG_DELAY', 'HISTORICAL_AVG_DELAY',
                                      'PAYMENT_IMPROVEMENT_SCORE', 'RECOVERY_INSTANCES'],
            'Financial Analysis': ['BILL_TREND_SLOPE', 'BILL_AMOUNT_VOLATILITY',
                                   'DEBT_ACCUMULATION_RATE', 'AVG_BILL_AMOUNT'],
            'Payment Efficiency': ['AVG_PAYMENT_EFFICIENCY', 'PAYMENT_EFFICIENCY_TREND',
                                   'PAYMENT_CONSISTENCY_SCORE', 'AVG_PAYMENT_AMOUNT'],
            'Credit Utilization': ['CREDIT_UTILIZATION_RATIO', 'CREDIT_BUFFER',
                                   'CREDIT_UTILIZATION_TREND'],
            'Behavioral Classification': ['PAYMENT_BEHAVIOR_TYPE', 'TEMPORAL_RISK_LEVEL',
                                          'CUSTOMER_SEGMENT'],
            'Risk Scoring': ['TEMPORAL_RISK_SCORE', 'RISK_SCORE_CATEGORY']
        }

        summary = {}
        for category, features in feature_categories.items():
            available_features = [f for f in features if f in df.columns]
            summary[category] = {
                'total_features': len(features),
                'available_features': len(available_features),
                'feature_list': available_features
            }

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize Spark session for testing
    spark = SparkSession.builder.appName("TemporalFeatureEngineerTest").getOrCreate()

    # Initialize feature engineer
    engineer = TemporalFeatureEngineer(spark)

    print("✅ TemporalFeatureEngineer class created successfully")
    print("📊 Available methods:")
    methods = [method for method in dir(engineer) if not method.startswith('_')]
    for method in methods:
        print(f"   - {method}()")

    spark.stop()