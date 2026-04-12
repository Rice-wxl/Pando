"""Neuronpedia API helper with caching for SAE feature descriptions and densities.

Supports two methods:
1. Bulk download from S3 (fast, recommended for many features)
2. Individual API calls (fallback, for small numbers of features)
"""

import gzip
import io
import json
import logging
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Cache file path - can be overridden via environment variable
DEFAULT_CACHE_PATH = Path(__file__).parent.parent / "sae_feature_cache.json"
CACHE_FILE_PATH = Path(os.getenv("SAE_FEATURE_CACHE_PATH", DEFAULT_CACHE_PATH))

# Bulk cache directory for S3 downloads
DEFAULT_BULK_CACHE_DIR = Path(__file__).parent.parent / "neuronpedia_bulk_cache"
BULK_CACHE_DIR = Path(os.getenv("NEURONPEDIA_BULK_CACHE_DIR", DEFAULT_BULK_CACHE_DIR))


# =============================================================================
# Bulk downloader (from S3)
# =============================================================================

@dataclass
class NeuronpediaFeature:
    """Represents a feature from Neuronpedia dataset."""
    id: str
    model_id: str
    layer: str
    index: str
    description: str
    author_id: Optional[str] = None
    created_at: Optional[str] = None
    embedding: Optional[list[float]] = None

    @classmethod
    def from_json(cls, data: dict) -> "NeuronpediaFeature":
        """Create a NeuronpediaFeature from JSON data."""
        return cls(
            id=data.get("id", ""),
            model_id=data.get("modelId", ""),
            layer=data.get("layer", ""),
            index=data.get("index", ""),
            description=data.get("description", ""),
            author_id=data.get("authorId"),
            created_at=data.get("createdAt"),
            embedding=data.get("embedding"),
        )


class NeuronpediaDownloader:
    """Handles downloading and caching of Neuronpedia data from S3."""

    S3_BASE_URL = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1"

    def __init__(self, cache_dir: Path):
        """Initialize the downloader with a cache directory."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_explanation_urls(self, model: str, sae: str, suffix: str = "-prev") -> list[str]:
        """Get list of explanation file URLs for a model/SAE combination."""
        prefix = f"v1/{model}/{sae}/explanations{suffix}/"
        list_url = f"https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/?prefix={prefix}&max-keys=1000"

        logger.info(f"Listing files from S3 with prefix: {prefix}")

        try:
            with urllib.request.urlopen(list_url, timeout=30) as response:
                xml_data = response.read().decode("utf-8")

            root = ET.fromstring(xml_data)

            # Extract all Keys that match our pattern
            namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            keys = []
            for contents in root.findall(".//s3:Contents", namespace):
                key_elem = contents.find("s3:Key", namespace)
                if key_elem is not None and key_elem.text:
                    key = key_elem.text
                    if "batch-" in key and key.endswith(".jsonl.gz"):
                        keys.append(key)

            # If no namespace, try without it
            if not keys:
                for contents in root.findall(".//Contents"):
                    key_elem = contents.find("Key")
                    if key_elem is not None and key_elem.text:
                        key = key_elem.text
                        if "batch-" in key and key.endswith(".jsonl.gz"):
                            keys.append(key)

            urls = [
                f"https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/{key}"
                for key in sorted(keys)
            ]
            logger.info(f"Found {len(urls)} explanation files")
            return urls

        except Exception as e:
            logger.error(f"Failed to list S3 bucket: {e}")
            return []

    def download_file(self, url: str, force: bool = False, old: bool = False) -> Optional[Path]:
        """Download a file from URL with caching."""
        filename = url.split("/")[-1]
        model = url.split("/")[-4]
        sae = url.split("/")[-3]
        if old:
            filename = 'old_' + filename

        cache_path = self.cache_dir / model / sae / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not force:
            logger.debug(f"Using cached file: {cache_path}")
            return cache_path

        logger.info(f"Downloading: {filename}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()

            with open(cache_path, "wb") as f:
                f.write(data)

            logger.debug(f"Downloaded to: {cache_path}")
            return cache_path

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def load_features_for_layer(
        self,
        model: str,
        sae: str,
        max_features: Optional[int] = None,
        old: bool = True,
    ) -> dict[int, NeuronpediaFeature]:
        """Load features from Neuronpedia dataset for a single SAE.

        Returns:
            Dict mapping feature index to NeuronpediaFeature.
        """
        # Get list of explanation files
        if not old:
            urls = self.get_explanation_urls(model, sae, suffix="")
        else:
            urls = self.get_explanation_urls(model, sae, suffix="-prev")
            if not urls:
                urls = self.get_explanation_urls(model, sae, suffix="-old")

        if not urls:
            logger.warning(f"No explanation files found for {model}/{sae}")
            return {}

        features = {}

        for url in urls:
            file_path = self.download_file(url, old=old)
            if not file_path:
                continue

            try:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            feature = NeuronpediaFeature.from_json(data)
                            features[int(feature.index)] = feature

                            if max_features and len(features) >= max_features:
                                return features

            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(features)} features for {sae}")
        return features


# Global bulk cache: model -> layer -> feature_index -> NeuronpediaFeature
_bulk_feature_cache: dict[str, dict[int, dict[int, NeuronpediaFeature]]] = {}
_bulk_downloader: NeuronpediaDownloader | None = None


def get_bulk_downloader() -> NeuronpediaDownloader:
    """Get or create the bulk downloader instance."""
    global _bulk_downloader
    if _bulk_downloader is None:
        _bulk_downloader = NeuronpediaDownloader(cache_dir=BULK_CACHE_DIR)
    return _bulk_downloader


def load_bulk_features_for_layer(
    layer: int,
    model: str = "gemma-2-2b",
    old: bool = True,
    format_str: str = "{layer}-gemmascope-res-16k",
    cache: bool = False,
) -> dict[int, NeuronpediaFeature]:
    """Load all features for a layer using bulk S3 download.

    Args:
        layer: Layer number.
        width_k: SAE width in thousands.
        model: Model name.
        old: Use old/prev explanations (more complete).

    Returns:
        Dict mapping feature index to NeuronpediaFeature.
    """
    global _bulk_feature_cache

    if cache and model not in _bulk_feature_cache:
        _bulk_feature_cache[model] = {}

    if cache and layer in _bulk_feature_cache[model]:
        return _bulk_feature_cache[model][layer]

    downloader = get_bulk_downloader()
    sae = format_str.format(layer=layer)

    print(f"\rBulk loading features for {model}/{sae}...", end="", flush=True)
    features = downloader.load_features_for_layer(model, sae, old=old)
    if cache:
        _bulk_feature_cache[model][layer] = features
    print(f" Loaded {len(features)} features            ", end="", flush=True)

    return features


def preload_bulk_features(
    layers: list[int],
    model: str = "gemma-2-2b",
    old: bool = True,
    format_str: str = "{layer}-gemmascope-res-16k",
    cache: bool = True,
) -> None:
    """Preload bulk features for multiple layers.

    Call this at agent init to load all features upfront.
    """
    for layer in layers:
        load_bulk_features_for_layer(layer, model, old, format_str, cache=cache)
    print()  # Final newline after all layers loaded


def get_bulk_feature_description(
    feature_index: int,
    layer: int,
    model: str = "gemma-2-2b",
) -> str | None:
    """Get feature description from bulk cache.

    Returns None if not in cache (caller should fall back to API).
    """
    global _bulk_feature_cache

    if model not in _bulk_feature_cache:
        return None
    if layer not in _bulk_feature_cache[model]:
        return None

    feature = _bulk_feature_cache[model][layer].get(feature_index)
    if feature is None:
        return None

    return feature.description if feature.description else None


# =============================================================================
# Feature density loading (from S3 /features/ path)
# =============================================================================

# In-memory density cache: model -> layer -> {feature_idx: frac_nonzero}
_density_cache: dict[str, dict[int, dict[int, float]]] = {}


def load_feature_densities(
    layer: int,
    model: str = "gemma-2-2b",
    format_str: str = "{layer}-gemmascope-res-16k",
) -> dict[int, float]:
    """Load feature densities (frac_nonzero) from Neuronpedia S3 features dataset.

    Downloads from the /features/ S3 path (not /explanations/), extracts
    frac_nonzero per feature, and caches as compact JSON on disk.

    Args:
        layer: Layer number.
        model: Model name (e.g., "gemma-2-2b").
        format_str: SAE name format string with {layer} placeholder.

    Returns:
        Dict mapping feature index to frac_nonzero (0-1 range).
    """
    global _density_cache

    # Check in-memory cache
    if model in _density_cache and layer in _density_cache[model]:
        return _density_cache[model][layer]

    sae_name = format_str.format(layer=layer)
    cache_path = BULK_CACHE_DIR / model / sae_name / "densities.json"

    # Check disk cache
    if cache_path.exists():
        with open(cache_path, "r") as f:
            densities = {int(k): v for k, v in json.load(f).items()}
        _density_cache.setdefault(model, {})[layer] = densities
        return densities

    # Download from S3 /features/ path
    prefix = f"v1/{model}/{sae_name}/features/"
    list_url = (
        f"https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
        f"?prefix={prefix}&max-keys=1000"
    )

    try:
        with urllib.request.urlopen(list_url, timeout=30) as response:
            xml_data = response.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
        keys = []
        for contents in root.findall(f'{ns}Contents'):
            key_elem = contents.find(f'{ns}Key')
            if key_elem is not None and key_elem.text:
                key = key_elem.text
                if key.endswith('.jsonl.gz') or key.endswith('.jsonl'):
                    keys.append(key)

        if not keys:
            logger.warning(f"No feature files found at {prefix}")
            return {}

        # Download each file, extract frac_nonzero, discard raw data
        densities = {}
        print(
            f"Downloading feature densities for {model}/{sae_name} "
            f"({len(keys)} files)...",
            end="", flush=True,
        )

        for key in keys:
            url = f"https://neuronpedia-datasets.s3.amazonaws.com/{key}"
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()

            if key.endswith('.gz'):
                f = gzip.open(io.BytesIO(data), 'rt', encoding='utf-8')
            else:
                f = io.StringIO(data.decode('utf-8'))

            with f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        idx = int(obj.get("index", -1))
                        frac = obj.get("frac_nonzero")
                        if idx >= 0 and frac is not None:
                            densities[idx] = float(frac)
            print(".", end="", flush=True)

        print(f" {len(densities)} features")

        # Save compact density cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(densities, f)

        _density_cache.setdefault(model, {})[layer] = densities
        return densities

    except Exception as e:
        logger.error(f"Failed to download feature densities: {e}")
        return {}


def preload_feature_densities(
    layers: list[int],
    model: str = "gemma-2-2b",
    format_str: str = "{layer}-gemmascope-res-16k",
) -> dict[int, dict[int, float]]:
    """Preload feature densities for multiple layers.

    Returns:
        Dict mapping layer -> {feature_idx: frac_nonzero}.
    """
    result = {}
    for layer in layers:
        densities = load_feature_densities(layer, model, format_str)
        if densities:
            result[layer] = densities
    return result


# =============================================================================
# Original API-based caching (kept as fallback)
# =============================================================================

_sae_feature_cache: dict | None = None


def load_sae_feature_cache() -> dict:
    """Load the SAE feature cache from disk."""
    global _sae_feature_cache
    if _sae_feature_cache is not None:
        return _sae_feature_cache

    try:
        if CACHE_FILE_PATH.exists():
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                _sae_feature_cache = json.load(f)
            print(
                f"Loaded SAE feature cache with {len(_sae_feature_cache.get('features', {}))} cached features"
            )
        else:
            _sae_feature_cache = {
                "cache_version": "1.0",
                "created": datetime.now().isoformat(),
                "features": {},
            }
            print(f"Created new SAE feature cache at {CACHE_FILE_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load SAE feature cache: {e}")
        _sae_feature_cache = {
            "cache_version": "1.0",
            "created": datetime.now().isoformat(),
            "features": {},
        }

    return _sae_feature_cache


def get_cache_key(feature_index: int, layer: int, width_k: int, model: str = "gemma-2-2b") -> str:
    """Generate a cache key for a feature."""
    return f"{model}-{layer}-{width_k}k-{feature_index}"


def save_sae_feature_cache() -> None:
    """Save the SAE feature cache to disk."""
    global _sae_feature_cache
    if _sae_feature_cache is None:
        return

    try:
        _sae_feature_cache["last_updated"] = datetime.now().isoformat()
        CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(_sae_feature_cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save SAE feature cache: {e}")


def fetch_sae_feature_description(
    feature_index: int,
    layer: int = 20,
    width_k: int = 16,
    model: str = "gemma-2-2b",
) -> str:
    """Fetch SAE feature description with multi-level caching.

    Checks in order:
    1. Bulk S3 cache (if preloaded)
    2. Local JSON cache
    3. Neuronpedia API (and caches result)

    Args:
        feature_index: The SAE feature index
        layer: The layer number (default: 20)
        width_k: The width in thousands (default: 16)
        model: The model name for Neuronpedia (default: gemma-2-2b)

    Returns:
        Feature description string, or "No description available" if not found
    """
    # 1. Check bulk cache first (fastest, if preloaded)
    bulk_desc = get_bulk_feature_description(feature_index, layer, model)
    if bulk_desc is not None:
        return bulk_desc

    # 2. Check local JSON cache
    cache = load_sae_feature_cache()
    cache_key = get_cache_key(feature_index, layer, width_k, model)

    if cache_key in cache["features"] and "description" in cache["features"][cache_key]:
        return cache["features"][cache_key]["description"]

    # 3. Fall back to API
    try:
        # Neuronpedia URL format: gemma-2-2b/{layer}-gemmascope-res-{width_k}k/{feature_index}
        url = f"https://www.neuronpedia.org/api/feature/{model}/{layer}-gemmascope-res-{width_k}k/{feature_index}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Try to extract description from the response
        description = "No description available"
        explanations = data.get("explanations", [])
        if explanations and len(explanations) > 0:
            first_explanation = explanations[0]
            if "description" in first_explanation:
                description = first_explanation["description"].strip()

        # Fallback: check if there's a direct description field
        if description == "No description available" and "description" in data:
            description = data["description"].strip()

        # Cache the result and immediately save to disk
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = description
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
        save_sae_feature_cache()

        return description

    except requests.RequestException as e:
        print(f"Warning: Failed to fetch description for feature {feature_index}: {e}")
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = "No description available"
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
        save_sae_feature_cache()
        return "No description available"
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Unexpected response format for feature {feature_index}: {e}")
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = "No description available"
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
        save_sae_feature_cache()
        return "No description available"


def get_neuronpedia_url(
    feature_index: int,
    layer: int = 20,
    width_k: int = 16,
    model: str = "gemma-2-2b",
) -> str:
    """Get the Neuronpedia URL for a feature.

    Args:
        feature_index: The SAE feature index
        layer: The layer number
        width_k: The width in thousands
        model: The model name

    Returns:
        URL string to the Neuronpedia feature page
    """
    return f"https://neuronpedia.org/{model}/{layer}-gemmascope-res-{width_k}k/{feature_index}?embed=true&embedexplanation=true&embedplots=true"


def batch_fetch_descriptions(
    feature_indices: list[int],
    layer: int = 20,
    width_k: int = 16,
    model: str = "gemma-2-2b",
) -> dict[int, str]:
    """Fetch descriptions for multiple features.

    Args:
        feature_indices: List of feature indices to fetch
        layer: The layer number
        width_k: The width in thousands
        model: The model name

    Returns:
        Dict mapping feature index to description
    """
    results = {}
    for idx in feature_indices:
        results[idx] = fetch_sae_feature_description(idx, layer, width_k, model)
    return results
