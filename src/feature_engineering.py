from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditCardDataProcessor:
    """
    Comprehensive data processing class for credit card default analysis
    Implements research-standard variable mapping and data quality checks
    """

    def __init__(self, spark_session):
        """Initialize with Spark session"""
        self.spark = spark_session
        self.variable_mapping = self._create_variable_mapping()

    def _create_variable_mapping(self):
        """Create research variable mapping (X1-X23)"""
        return {
            # Demographics (X1-X5)
            'LIMIT_BAL': 'X1', 'SEX': 'X2', 'EDUCATION': 'X3', 'MARRIAGE': 'X4', 'AGE': 'X5',
            # Payment History (X6-X11)
            'PAY_0': 'X6', 'PAY_2': 'X7', 'PAY_3': 'X8', 'PAY_4': 'X9', 'PAY_5': 'X10', 'PAY_6': 'X11',
            # Bill Statements (X12-X17)
            'BILL_AMT1': 'X12', 'BILL_AMT2': 'X13', 'BILL_AMT3': 'X14',
            'BILL_AMT4': 'X15', 'BILL_AMT5': 'X16', 'BILL_AMT6': 'X17',
            # Payment Amounts (X18-X23)
            'PAY_AMT1': 'X18', 'PAY_AMT2': 'X19', 'PAY_AMT3': 'X20',
            'PAY_AMT4': 'X21', 'PAY_AMT5': 'X22', 'PAY_AMT6': 'X23'
        }

    def load_data(self, file_path, file_format="csv"):
        """Load dataset with error handling"""
        try:
            if file_format.lower() == "csv":
                df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            elif file_format.lower() == "parquet":
                df = self.spark.read.parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            logger.info(f"Data loaded successfully: {df.count():,} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_schema(self, df):
        """Validate dataset schema against research variables"""
        missing_vars = []
        for orig_col, research_var in self.variable_mapping.items():
            if orig_col not in df.columns:
                missing_vars.append(f"{orig_col} ({research_var})")

        if missing_vars:
            logger.warning(f"Missing variables: {', '.join(missing_vars)}")
        else:
            logger.info("✅ All research variables present in dataset")

        return len(missing_vars) == 0

    def quality_assessment(self, df):
        """Comprehensive data quality assessment"""
        logger.info("Performing data quality assessment...")

        quality_report = {
            'total_records': df.count(),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicates': 0,
            'data_types': dict(df.dtypes)
        }

        # Check missing values
        for col_name in df.columns:
            missing_count = df.filter(col(col_name).isNull()).count()
            if missing_count > 0:
                quality_report['missing_values'][col_name] = missing_count

        # Check duplicates
        quality_report['duplicates'] = df.count() - df.dropDuplicates().count()

        # Log quality summary
        logger.info(f"Quality Assessment Summary:")
        logger.info(f"  Total records: {quality_report['total_records']:,}")
        logger.info(f"  Missing values: {len(quality_report['missing_values'])} columns affected")
        logger.info(f"  Duplicate records: {quality_report['duplicates']:,}")

        return quality_report

    def clean_data(self, df):
        """Apply data cleaning transformations"""
        logger.info("Applying data cleaning transformations...")

        df_clean = df

        # Clean categorical variables
        if 'EDUCATION' in df.columns:
            df_clean = df_clean.withColumn(
                "EDUCATION",
                when(col("EDUCATION").isin([0, 5, 6]), 4).otherwise(col("EDUCATION"))
            )

        if 'MARRIAGE' in df.columns:
            df_clean = df_clean.withColumn(
                "MARRIAGE",
                when(col("MARRIAGE") == 0, 3).otherwise(col("MARRIAGE"))
            )

        # Handle outliers in financial variables
        financial_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]

        for col_name in financial_cols:
            if col_name in df.columns:
                # Cap extreme values at 99th percentile
                upper_bound = df.select(percentile_approx(col_name, 0.99)).collect()[0][0]
                df_clean = df_clean.withColumn(
                    col_name,
                    when(col(col_name) > upper_bound, upper_bound).otherwise(col(col_name))
                )

        logger.info("✅ Data cleaning completed")
        return df_clean

    def save_processed_data(self, df, output_path, format="parquet"):
        """Save processed dataset"""
        try:
            if format.lower() == "parquet":
                df.write.mode("overwrite").parquet(output_path)
            elif format.lower() == "csv":
                df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

            logger.info(f"✅ Processed data saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("DataProcessing").getOrCreate()

    # Initialize processor
    processor = CreditCardDataProcessor(spark)

    # Process data
    df = processor.load_data("data/sample.csv")
    processor.validate_schema(df)
    quality_report = processor.quality_assessment(df)
    df_clean = processor.clean_data(df)
    processor.save_processed_data(df_clean, "data/processed/clean_data.parquet")

    print("✅ Data processing pipeline completed successfully")