def load_model_safely(model, state_dict):
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    # If keys start with 'base_model.', rename them to 'model.'
    if all(k.startswith("base_model.") for k in loaded_keys):
        print("[INFO] Renaming 'base_model.' keys to 'model.' for compatibility")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("base_model.", "model.", 1)
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        loaded_keys = set(state_dict.keys())

    # Now check keys match
    if model_keys == loaded_keys:
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError("Mismatch between model and state_dict keys after renaming.")

    return model
