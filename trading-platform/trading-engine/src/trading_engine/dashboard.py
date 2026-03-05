from trading_engine.models.registry import get_model

SELECTABLE_MODELS = [
    "IBIT_AMMA",
    "IBIT_GRAND_STACK",
]


def run_selected_model(data, model_name: str):
    model = get_model(model_name)
    return model(data)
