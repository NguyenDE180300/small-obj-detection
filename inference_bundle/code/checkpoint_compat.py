"""Small inference-only shims for older serialized custom modules."""

from __future__ import annotations


def patch_legacy_modules(model) -> list[str]:
    """Populate attributes introduced after older checkpoints were serialized.

    Older SRMF2Guidance objects predate the optional saturation branch. If the
    branch modules and parameter were not saved, the historically correct
    behavior is to disable that branch. If they are present, enable it.
    """
    patched = []
    inner = getattr(model, "model", model)
    modules = inner.modules() if hasattr(inner, "modules") else ()
    for module in modules:
        if type(module).__name__ == "SRMF2Guidance" and not hasattr(module, "use_saturation_guidance"):
            module.use_saturation_guidance = bool(
                hasattr(module, "saturation_mask_net") and hasattr(module, "gamma_saturation_raw")
            )
            patched.append(type(module).__name__)
    return patched
