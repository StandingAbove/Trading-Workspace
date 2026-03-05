from trading_engine.models.catalogue.ibit_amma import generate_positions as ibit_amma_generate_positions
from trading_engine.models.catalogue.ibit_grand_stack import generate_positions as ibit_grand_stack_generate_positions

MODEL_REGISTRY = {
    "IBIT_AMMA": ibit_amma_generate_positions,
    "IBIT_GRAND_STACK": ibit_grand_stack_generate_positions,
}


def get_model(model_name: str):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_name}'") from exc
