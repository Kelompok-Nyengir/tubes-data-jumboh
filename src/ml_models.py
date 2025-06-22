from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditDefaultMLPipeline:
    """
    Comprehensive machine learning pipeline for credit card default analysis
    Compatible with the 05_machine_learning.ipynb notebook
    """

    def __init__(self, spark_session):
        """Initialize with Spark session"""
        self.spark = spark_session
        self.training_times = {}
        self.trained_models = {}
        self.preprocessing_pipeline = None
        self.feature_names = []

        # Feature categorization
        self.demographic_features = ['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE']
        self.payment_history_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        self.bill_features = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        self.payment_features = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

        logger.info("✅ CreditDefaultMLPipeline initialized successfully")

    def prepare_features(self, df, feature_selection_method='all'):
        """
        Prepare features for machine learning
        Returns: (numerical_features, categorical_features)
        """
        logger.info("Preparing features for machine learning...")

        # Get all columns except target
        target_col = "default payment next month"
        all_columns = [col for col in df.columns if col != target_col]

        # Identify numerical and categorical features
        numerical_features = []
        categorical_features = []

        for col_name in all_columns:
            col_type = dict(df.dtypes)[col_name]

            # Check if categorical based on column name or data type
            if col_name in ['SEX', 'EDUCATION', 'MARRIAGE']:
                categorical_features.append(col_name)
            elif col_name in ['PAYMENT_BEHAVIOR_TYPE', 'TEMPORAL_RISK_LEVEL', 'CUSTOMER_SEGMENT',
                              'RISK_SCORE_CATEGORY']:
                categorical_features.append(col_name)
            elif col_type in ['int', 'bigint', 'double', 'float']:
                # Check if it's actually categorical (small number of unique values)
                if col_name not in ['SEX', 'EDUCATION', 'MARRIAGE']:
                    try:
                        unique_count = df.select(col_name).distinct().count()
                        if unique_count <= 10 and col_name.startswith('PAY_'):
                            # Payment status columns are categorical but we'll treat as numerical
                            numerical_features.append(col_name)
                        else:
                            numerical_features.append(col_name)
                    except:
                        numerical_features.append(col_name)

        # Feature selection based on method
        if feature_selection_method == 'all':
            # Use all available features
            pass
        elif feature_selection_method == 'core':
            # Use only core features
            core_features = (self.demographic_features + self.payment_history_features +
                             self.bill_features + self.payment_features)
            numerical_features = [f for f in numerical_features if f in core_features]
            categorical_features = [f for f in categorical_features if f in core_features]

        # Remove any features that don't exist in the dataset
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        logger.info(f"Feature preparation completed:")
        logger.info(f"  Numerical features: {len(numerical_features)}")
        logger.info(f"  Categorical features: {len(categorical_features)}")

        return numerical_features, categorical_features

    def split_data(self, df, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Split data into train, validation, and test sets
        Returns: (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/validation/test sets...")

        # Ensure ratios sum to 1
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
            train_ratio /= total_ratio
            validation_ratio /= total_ratio
            test_ratio /= total_ratio

        # Split data
        train_df, temp_df = df.randomSplit([train_ratio, validation_ratio + test_ratio], seed=seed)

        # Calculate new ratios for val/test split
        val_ratio_adjusted = validation_ratio / (validation_ratio + test_ratio)
        val_df, test_df = temp_df.randomSplit([val_ratio_adjusted, 1 - val_ratio_adjusted], seed=seed)

        logger.info(f"Data split completed:")
        logger.info(f"  Training: {train_df.count():,} records")
        logger.info(f"  Validation: {val_df.count():,} records")
        logger.info(f"  Test: {test_df.count():,} records")

        return train_df, val_df, test_df

    def create_preprocessing_pipeline(self, numerical_features, categorical_features):
        """
        Create preprocessing pipeline
        Returns: Spark ML Pipeline
        """
        logger.info("Creating preprocessing pipeline...")

        stages = []

        # Handle categorical features
        indexed_categorical = []
        if categorical_features:
            for cat_feature in categorical_features:
                indexer = StringIndexer(
                    inputCol=cat_feature,
                    outputCol=f"{cat_feature}_indexed",
                    handleInvalid="keep"
                )
                stages.append(indexer)
                indexed_categorical.append(f"{cat_feature}_indexed")

        # Combine all features
        all_feature_cols = numerical_features + indexed_categorical
        self.feature_names = all_feature_cols

        # Vector assembler
        assembler = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        stages.append(assembler)

        # Feature scaling
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaledFeatures",
            withStd=True,
            withMean=False
        )
        stages.append(scaler)

        # Create pipeline
        pipeline = Pipeline(stages=stages)

        logger.info(f"Preprocessing pipeline created with {len(stages)} stages")
        return pipeline

    def initialize_models(self):
        """
        Initialize ML models
        Returns: Dictionary of model instances
        """
        logger.info("Initializing machine learning models...")

        models = {
            "Logistic Regression": LogisticRegression(
                featuresCol="scaledFeatures",
                labelCol="default payment next month",
                maxIter=100
            ),
            "Random Forest": RandomForestClassifier(
                featuresCol="scaledFeatures",
                labelCol="default payment next month",
                numTrees=100,
                seed=42
            ),
            "Gradient Boosting": GBTClassifier(
                featuresCol="scaledFeatures",
                labelCol="default payment next month",
                maxIter=100,
                seed=42
            ),
            "Neural Network": MultilayerPerceptronClassifier(
                featuresCol="scaledFeatures",
                labelCol="default payment next month",
                layers=[10, 5, 2],  # Will be adjusted based on actual feature count
                seed=42,
                maxIter=100
            )
        }

        logger.info(f"Initialized {len(models)} models")
        return models

    def create_hyperparameter_grids(self):
        """
        Create hyperparameter grids for tuning
        Returns: Dictionary of parameter grids
        """
        logger.info("Creating hyperparameter grids...")

        param_grids = {
            "Logistic Regression": ParamGridBuilder() \
                .addGrid(LogisticRegression.regParam, [0.01, 0.1, 1.0]) \
                .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
                .build(),

            "Random Forest": ParamGridBuilder() \
                .addGrid(RandomForestClassifier.numTrees, [50, 100, 200]) \
                .addGrid(RandomForestClassifier.maxDepth, [5, 10, 15]) \
                .build(),

            "Gradient Boosting": ParamGridBuilder() \
                .addGrid(GBTClassifier.maxIter, [50, 100]) \
                .addGrid(GBTClassifier.maxDepth, [5, 10]) \
                .build(),

            "Neural Network": ParamGridBuilder() \
                .build()  # Keep simple for neural network
        }

        logger.info(f"Created parameter grids for {len(param_grids)} models")
        return param_grids

    def train_models(self, train_df, val_df, use_hyperparameter_tuning=True):
        """
        Train models with optional hyperparameter tuning
        Returns: Dictionary of trained models
        """
        logger.info("Starting model training...")

        models = self.initialize_models()
        param_grids = self.create_hyperparameter_grids()
        trained_models = {}

        # Create evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="default payment next month",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            start_time = time.time()

            try:
                if use_hyperparameter_tuning and model_name in param_grids:
                    # Use cross-validation for hyperparameter tuning
                    cv = CrossValidator(
                        estimator=model,
                        estimatorParamMaps=param_grids[model_name],
                        evaluator=evaluator,
                        numFolds=3,
                        seed=42
                    )

                    # Fit the cross-validator
                    cv_model = cv.fit(train_df)
                    trained_model = cv_model.bestModel
                else:
                    # Train without hyperparameter tuning
                    trained_model = model.fit(train_df)

                # Store the trained model
                trained_models[model_name] = trained_model

                # Record training time
                training_time = time.time() - start_time
                self.training_times[model_name] = training_time

                logger.info(f"✅ {model_name} training completed in {training_time:.1f}s")

            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                # Record failed training time
                self.training_times[model_name] = time.time() - start_time
                continue

        self.trained_models = trained_models
        logger.info(f"Model training completed. Successfully trained {len(trained_models)} models.")

        return trained_models

    def evaluate_models(self, test_df):
        """
        Evaluate trained models
        Returns: Dictionary with model metrics
        """
        logger.info("Evaluating trained models...")

        if not self.trained_models:
            logger.error("No trained models available for evaluation")
            return {}

        # Create evaluators
        auc_evaluator = BinaryClassificationEvaluator(
            labelCol="default payment next month",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="default payment next month",
            predictionCol="prediction",
            metricName="accuracy"
        )

        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="default payment next month",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )

        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="default payment next month",
            predictionCol="prediction",
            metricName="weightedRecall"
        )

        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="default payment next month",
            predictionCol="prediction",
            metricName="f1"
        )

        evaluation_results = {}

        for model_name, model in self.trained_models.items():
            logger.info(f"Evaluating {model_name}...")

            try:
                # Make predictions
                predictions = model.transform(test_df)

                # Calculate metrics
                auc = auc_evaluator.evaluate(predictions)
                accuracy = accuracy_evaluator.evaluate(predictions)
                precision = precision_evaluator.evaluate(predictions)
                recall = recall_evaluator.evaluate(predictions)
                f1 = f1_evaluator.evaluate(predictions)

                # Store results
                evaluation_results[model_name] = {
                    'AUC': auc,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'Training_Time': self.training_times.get(model_name, 0)
                }

                logger.info(f"✅ {model_name} evaluation completed - AUC: {auc:.4f}")

            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {e}")
                continue

        logger.info(f"Model evaluation completed for {len(evaluation_results)} models")
        return evaluation_results

    def extract_feature_importance(self, model_name):
        """
        Extract feature importance from tree-based models
        Returns: List of (feature_name, importance) tuples
        """
        if model_name not in self.trained_models:
            logger.warning(f"Model {model_name} not found in trained models")
            return []

        model = self.trained_models[model_name]

        try:
            if hasattr(model, 'featureImportances'):
                # Get feature importances
                importances = model.featureImportances.toArray()

                # Create feature names if not available
                if not self.feature_names:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                else:
                    feature_names = self.feature_names[:len(importances)]

                # Create list of (feature, importance) tuples
                feature_importance = list(zip(feature_names, importances))

                # Sort by importance (descending)
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                logger.info(f"Feature importance extracted for {model_name}")
                return feature_importance
            else:
                logger.warning(f"Model {model_name} does not support feature importance")
                return []

        except Exception as e:
            logger.error(f"Error extracting feature importance for {model_name}: {e}")
            return []

    def create_business_insights(self, test_df):
        """
        Create business insights and analysis
        Returns: Dictionary with business analysis
        """
        logger.info("Creating business insights...")

        if not self.trained_models:
            logger.warning("No trained models available for business insights")
            return None

        try:
            # Find best model by AUC
            evaluation_results = self.evaluate_models(test_df)
            if not evaluation_results:
                return None

            best_model_name = max(evaluation_results.keys(), key=lambda k: evaluation_results[k]['AUC'])
            best_model = self.trained_models[best_model_name]

            # Basic statistics
            total_customers = test_df.count()
            actual_defaults = test_df.filter(col("default payment next month") == 1).count()

            # Get predictions from best model
            predictions = best_model.transform(test_df)
            predicted_defaults = predictions.filter(col("prediction") == 1.0).count()

            business_insights = {
                'best_model': best_model_name,
                'model_performance': evaluation_results[best_model_name],
                'total_customers': total_customers,
                'actual_defaults': actual_defaults,
                'predicted_defaults': predicted_defaults
            }

            # Risk distribution analysis (if risk columns exist)
            if 'RISK_SCORE_CATEGORY' in test_df.columns:
                try:
                    risk_analysis = predictions.groupBy('RISK_SCORE_CATEGORY') \
                        .pivot('default payment next month') \
                        .count() \
                        .fillna(0)

                    # Calculate default rates
                    risk_analysis = risk_analysis.withColumn(
                        'total', col('0') + col('1')
                    ).withColumn(
                        'default_rate',
                        when(col('total') > 0, col('1') / col('total') * 100).otherwise(0)
                    )

                    business_insights['risk_distribution'] = risk_analysis.collect()
                except Exception as e:
                    logger.warning(f"Could not create risk distribution analysis: {e}")

            # Customer segment analysis (if segment columns exist)
            if 'CUSTOMER_SEGMENT' in test_df.columns:
                try:
                    segment_analysis = predictions.groupBy('CUSTOMER_SEGMENT').agg(
                        count('*').alias('total_customers'),
                        avg('default payment next month').alias('actual_default_rate'),
                        avg('prediction').alias('predicted_default_rate')
                    )

                    business_insights['segment_analysis'] = segment_analysis.collect()
                except Exception as e:
                    logger.warning(f"Could not create segment analysis: {e}")

            logger.info("Business insights created successfully")
            return business_insights

        except Exception as e:
            logger.error(f"Error creating business insights: {e}")
            return None

    def save_models(self, output_dir):
        """
        Save trained models to specified directory
        """
        logger.info(f"Saving trained models to {output_dir}...")

        if not self.trained_models:
            logger.warning("No trained models to save")
            return

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            for model_name, model in self.trained_models.items():
                # Create safe filename
                safe_name = model_name.replace(" ", "_").lower()
                model_path = os.path.join(output_dir, safe_name)

                try:
                    # Save Spark model
                    model.write().overwrite().save(model_path)
                    logger.info(f"✅ Saved {model_name} to {model_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save {model_name}: {e}")

            # Save training times and metadata
            metadata = {
                'training_times': self.training_times,
                'model_count': len(self.trained_models),
                'feature_names': self.feature_names
            }

            import json
            metadata_path = os.path.join(output_dir, 'training_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Model saving completed")

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Initialize Spark session for testing
    spark = SparkSession.builder.appName("CreditDefaultMLPipelineTest").getOrCreate()

    # Initialize ML pipeline
    pipeline = CreditDefaultMLPipeline(spark)

    print("✅ CreditDefaultMLPipeline class created successfully")
    print("📊 Available methods:")
    methods = [method for method in dir(pipeline) if not method.startswith('_')]
    for method in methods:
        print(f"   - {method}()")

    spark.stop()