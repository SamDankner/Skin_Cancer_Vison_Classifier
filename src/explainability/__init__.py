"""On-demand image attribution utilities for the deployment demo."""

from .service import AttributionResult, generate_attribution
from .visualization import overlay_heatmap

__all__ = ["AttributionResult", "generate_attribution", "overlay_heatmap"]
