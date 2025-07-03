def load_model_safely(model, state_dict):
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    # Exact match
    if model_keys == loaded_keys:
        model.load_state_dict(state_dict)
        return model

    # Case 1: Loaded keys missing 'model.' prefix
    if all("model." + k in model_keys for k in loaded_keys):
        print("[INFO] Adding 'model.' prefix to loaded keys.")
        state_dict = {"model." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    # Case 2: Loaded keys have extra 'model.' prefix
    if all(k.startswith("model.") for k in loaded_keys):
        stripped_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        if set(stripped_state_dict.keys()) == model_keys:
            print("[INFO] Stripping 'model.' prefix from loaded keys.")
            model.load_state_dict(stripped_state_dict)
            return model

    # Case 3: Loaded keys have 'base_model.' prefix but model expects 'model.'
    if all(k.startswith("base_model.") for k in loaded_keys):
        converted_state_dict = {k.replace("base_model.", "model.", 1): v for k, v in state_dict.items()}
        if set(converted_state_dict.keys()) == model_keys:
            print("[INFO] Renaming 'base_model.' prefix to 'model.' in loaded keys.")
            model.load_state_dict(converted_state_dict)
            return model

    # Case 4: Loaded keys have 'model.' prefix but model expects 'base_model.'
    if all(k.startswith("model.") for k in loaded_keys) and all(k.startswith("base_model.") for k in model_keys):
        converted_state_dict = {k.replace("model.", "base_model.", 1): v for k, v in state_dict.items()}
        if set(converted_state_dict.keys()) == model_keys:
            print("[INFO] Renaming 'model.' prefix to 'base_model.' in loaded keys.")
            model.load_state_dict(converted_state_dict)
            return model

    # If none match, fail
    raise RuntimeError("Mismatch between model and state_dict keys after all prefix checks.")


