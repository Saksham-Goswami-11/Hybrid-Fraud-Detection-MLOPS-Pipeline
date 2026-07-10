"""
Pydantic schemas for the Fraud Detection API.

Defines request and response models with validation
for transaction scoring and health check endpoints.
"""


from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """
    Input schema for a single credit card transaction.

    Features V1–V28 are PCA-transformed (anonymized).
    Amount and Time are raw transaction fields.
    """

    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction")
    card_id: str = Field("card_default_123", description="Credit card identifier for velocity tracking")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "V1": -1.3598071336738,
                    "V2": -0.0727811733098497,
                    "V3": 2.53634673796914,
                    "V4": 1.37815522427443,
                    "V5": -0.338320769942518,
                    "V6": 0.462387777762292,
                    "V7": 0.239598554061257,
                    "V8": 0.0986979012610507,
                    "V9": 0.363786969611213,
                    "V10": 0.0907941719789316,
                    "V11": -0.551599533260813,
                    "V12": -0.617800855762348,
                    "V13": -0.991389847235408,
                    "V14": -0.311169353699879,
                    "V15": 1.46817697209427,
                    "V16": -0.470400525259478,
                    "V17": 0.207971241929242,
                    "V18": 0.0257905801985591,
                    "V19": 0.403992960255733,
                    "V20": 0.251412098239705,
                    "V21": -0.018306777944153,
                    "V22": 0.277837575558899,
                    "V23": -0.110473910188767,
                    "V24": 0.0669280749146731,
                    "V25": 0.128539358273528,
                    "V26": -0.189114843888824,
                    "V27": 0.133558376740387,
                    "V28": -0.0210530534538215,
                    "Amount": 149.62,
                    "Time": 0.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single transaction prediction."""

    fraud_probability: float = Field(
        ..., ge=0, le=1, description="Probability of fraud (0 to 1)"
    )
    is_fraud: bool = Field(..., description="Whether the transaction is flagged as fraud")
    threshold: float = Field(..., description="Decision threshold used")
    shap_values: dict[str, float] = Field(
        ..., description="SHAP contribution per feature"
    )
    top_risk_factors: list[str] = Field(
        ..., description="Top 5 features driving the fraud prediction"
    )
    rule_triggered: bool = Field(
        False, description="Whether a deterministic rule overrode the model's output"
    )
    rule_reasons: list[str] = Field(
        [], description="Reasons for rule override (if any)"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    transactions: list[TransactionInput] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions to score"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse]
    total: int
    fraud_count: int


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="API status")
    model_version: str = Field(..., description="Loaded model identifier")
    model_type: str = Field(..., description="Type of the loaded model")
    timestamp: str = Field(..., description="Current server timestamp")


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""

    model_type: str
    model_name: str
    threshold: float
    features: list[str]
    n_features: int
