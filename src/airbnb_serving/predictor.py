import pandas as pd

from airbnb_serving.schema import ListingFeatures, PredictionResponse

FEATURE_COLS = [
    "property_type", "room_type", "accommodates", "bathrooms",
    "bedrooms", "beds", "listing_price", "minimum_nights",
    "maximum_nights", "instant_bookable", "is_superhost",
    "host_listing_count", "neighbourhood_name",
    "total_reviews_before_cutoff", "unique_reviewers_before_cutoff",
    "avg_comment_len_before_cutoff", "max_comment_len_before_cutoff",
    "days_since_last_review", "available_days_last_90d",
    "available_rate_last_90d", "avg_minimum_nights_calendar_last_90d",
    "avg_maximum_nights_calendar_last_90d", "available_days_last_30d",
    "available_rate_last_30d", "avg_minimum_nights_calendar_last_30d",
    "avg_maximum_nights_calendar_last_30d", "future_available_rate_30d",
]


def predict_single(features: ListingFeatures, model, run_id: str) -> PredictionResponse:
    """Predict for a single listing."""
    row = features.model_dump()
    df = pd.DataFrame([row])[FEATURE_COLS]

    pred_label = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[:, 1][0])

    return PredictionResponse(
        prediction=pred_label,
        probability_high_demand=proba,
        model_run_id=run_id,
    )


def predict_batch(features_list: list[ListingFeatures], model, run_id: str) -> list[PredictionResponse]:
    """Predict for a batch of listings in one step."""
    rows = [f.model_dump() for f in features_list]
    df = pd.DataFrame(rows)[FEATURE_COLS]

    pred_labels = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    results = []
    for pred, proba in zip(pred_labels, probas):
        results.append(PredictionResponse(
            prediction=int(pred),
            probability_high_demand=float(proba),
            model_run_id=run_id,
        ))
    return results