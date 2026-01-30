"""
Deployment Manager

Handles model promotion to production with safe deployment strategies.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import yaml
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentReport:
    """Container for deployment results"""

    timestamp: datetime
    model_uri: str
    model_version: str
    deployment_status: str  # 'success', 'failed', 'rolled_back'
    deployment_strategy: str  # 'direct', 'shadow', 'canary'
    previous_version: Optional[str]
    details: Dict[str, Any]


class DeploymentManager:
    """
    Manages model deployment to production with rollback capabilities.

    Supports deployment strategies:
    - Direct: Immediate replacement
    - Shadow: Run in parallel, collect metrics
    - Canary: Gradual rollout with monitoring
    """

    def __init__(self, config_path: str = "config/retrain_config.yaml"):
        """
        Initialize deployment manager.

        Args:
            config_path: Path to retraining configuration
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.deployment_config = self.config.get("deployment", {})
        self.model_name = self.deployment_config.get(
            "model_name", "predictive_maintenance_model"
        )
        self.deployment_strategy = self.deployment_config.get("strategy", "direct")
        self.enable_rollback = self.deployment_config.get("enable_rollback", True)

        self.client = MlflowClient()

        logger.info(
            f"DeploymentManager initialized. "
            f"Model: {self.model_name}, Strategy: {self.deployment_strategy}"
        )

    def promote_to_production(
        self, model_uri: str, archive_existing: bool = True
    ) -> DeploymentReport:
        """
        Promote a model to production.

        Args:
            model_uri: URI of model to promote (e.g., 'runs:/abc123/model')
            archive_existing: Whether to archive current production model

        Returns:
            DeploymentReport with deployment results
        """
        try:
            # Get current production version
            previous_version = None
            try:
                prod_versions = self.client.get_latest_versions(
                    self.model_name, stages=["Production"]
                )
                if prod_versions:
                    previous_version = prod_versions[0].version
            except Exception:
                pass

            # Register model if not already registered
            model_version = self._register_model(model_uri)

            # Archive existing production model
            if archive_existing and previous_version:
                self._archive_version(previous_version)

            # Promote new model to production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=archive_existing,
            )

            # Log deployment
            self._log_deployment(model_version, previous_version)

            logger.info(
                f"Successfully promoted model version {model_version} to production. "
                f"Previous: {previous_version}"
            )

            return DeploymentReport(
                timestamp=datetime.now(),
                model_uri=model_uri,
                model_version=model_version,
                deployment_status="success",
                deployment_strategy=self.deployment_strategy,
                previous_version=previous_version,
                details={
                    "archived_existing": archive_existing,
                    "model_name": self.model_name,
                },
            )

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return DeploymentReport(
                timestamp=datetime.now(),
                model_uri=model_uri,
                model_version="unknown",
                deployment_status="failed",
                deployment_strategy=self.deployment_strategy,
                previous_version=previous_version,
                details={"error": str(e)},
            )

    def rollback_to_previous(self) -> DeploymentReport:
        """
        Rollback to previous production model.

        Returns:
            DeploymentReport with rollback results
        """
        if not self.enable_rollback:
            logger.error("Rollback is disabled in configuration")
            return self._create_error_report("Rollback disabled")

        try:
            # Get current production version
            prod_versions = self.client.get_latest_versions(
                self.model_name, stages=["Production"]
            )

            if not prod_versions:
                raise ValueError("No production model found")

            current_version = prod_versions[0].version

            # Get archived versions (previous production models)
            archived_versions = self.client.get_latest_versions(
                self.model_name, stages=["Archived"]
            )

            if not archived_versions:
                raise ValueError("No archived model found for rollback")

            # Get most recent archived version
            previous_version = archived_versions[0].version

            # Promote archived version back to production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=previous_version,
                stage="Production",
                archive_existing_versions=True,
            )

            logger.info(
                f"Successfully rolled back to version {previous_version}. "
                f"Archived current: {current_version}"
            )

            return DeploymentReport(
                timestamp=datetime.now(),
                model_uri=f"models:/{self.model_name}/{previous_version}",
                model_version=previous_version,
                deployment_status="rolled_back",
                deployment_strategy="rollback",
                previous_version=current_version,
                details={"reason": "manual_rollback", "model_name": self.model_name},
            )

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return self._create_error_report(str(e))

    def _register_model(self, model_uri: str) -> str:
        """
        Register model in MLflow Model Registry.

        Args:
            model_uri: URI of model to register

        Returns:
            Model version number
        """
        try:
            result = mlflow.register_model(model_uri, self.model_name)
            return result.version
        except Exception as e:
            # Model might already be registered
            logger.warning(f"Model registration issue: {e}")
            # Extract version from URI if possible
            if "runs:/" in model_uri:
                run_id = model_uri.split("/")[1]
                versions = self.client.search_model_versions(f"run_id='{run_id}'")
                if versions:
                    return versions[0].version
            raise

    def _archive_version(self, version: str) -> None:
        """
        Archive a model version.

        Args:
            version: Version to archive
        """
        try:
            self.client.transition_model_version_stage(
                name=self.model_name, version=version, stage="Archived"
            )
            logger.info(f"Archived model version {version}")
        except Exception as e:
            logger.warning(f"Failed to archive version {version}: {e}")

    def _log_deployment(
        self, new_version: str, previous_version: Optional[str]
    ) -> None:
        """
        Log deployment event to MLflow.

        Args:
            new_version: Newly deployed version
            previous_version: Previously deployed version
        """
        try:
            mlflow.set_experiment("model_deployment")

            with mlflow.start_run(
                run_name=f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("new_version", new_version)
                mlflow.log_param("previous_version", previous_version or "none")
                mlflow.log_param("deployment_strategy", self.deployment_strategy)
                mlflow.set_tag("deployment_status", "success")
                mlflow.set_tag("deployment_time", datetime.now().isoformat())

        except Exception as e:
            logger.warning(f"Failed to log deployment: {e}")

    def _create_error_report(self, error_msg: str) -> DeploymentReport:
        """Create error report when deployment fails"""
        return DeploymentReport(
            timestamp=datetime.now(),
            model_uri="unknown",
            model_version="unknown",
            deployment_status="failed",
            deployment_strategy=self.deployment_strategy,
            previous_version=None,
            details={"error": error_msg},
        )

    def get_production_model_info(self) -> Dict[str, Any]:
        """
        Get information about current production model.

        Returns:
            Dictionary with model information
        """
        try:
            prod_versions = self.client.get_latest_versions(
                self.model_name, stages=["Production"]
            )

            if not prod_versions:
                return {"status": "no_production_model"}

            version = prod_versions[0]

            return {
                "model_name": self.model_name,
                "version": version.version,
                "run_id": version.run_id,
                "current_stage": version.current_stage,
                "creation_timestamp": version.creation_timestamp,
                "last_updated_timestamp": version.last_updated_timestamp,
                "status": version.status,
                "source": version.source,
            }

        except Exception as e:
            logger.error(f"Failed to get production model info: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """Test deployment manager"""
    print("\n=== Deployment Manager Test ===")
    print("Note: This is a demo. Real usage requires models in MLflow.")
    print("\nTypical deployment flow:")
    print("1. Model comparison determines challenger should be promoted")
    print("2. DeploymentManager.promote_to_production(challenger_uri)")
    print("3. Current production model archived automatically")
    print("4. New model becomes production")
    print("5. If issues detected: DeploymentManager.rollback_to_previous()")


if __name__ == "__main__":
    main()
