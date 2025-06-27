def load_model_safely(model, state_dict):

    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    if model_keys == loaded_keys:
        # Exact match
        model.load_state_dict(state_dict)
    elif all("model." + k in model_keys for k in loaded_keys):
        # Need to add "model." prefix to loaded keys
        state_dict = {"model." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    elif all(k.startswith("model.") for k in loaded_keys):
        # Need to remove "model." prefix from loaded keys
        stripped_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        if set(stripped_state_dict.keys()) == model_keys:
            model.load_state_dict(stripped_state_dict)
        else:
            raise RuntimeError("Stripped keys still don't match the model keys.")
    else:
        raise RuntimeError("Mismatch between model and state_dict keys.")

    return model