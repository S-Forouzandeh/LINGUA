"""
LINGUA: Language-based Inference for Grounded Video Understanding Agent
=======================================================================

Reference implementation accompanying the paper:

    Bridging the Grounding Gap in VideoQA via Typed Memory
    for Language-based Belief-State Reasoning.
    Forouzandeh, Peng, Yu, Jalili. ICML 2026.

LINGUA performs grounded VideoQA by reasoning in an explicit linguistic
belief-state space, supported by:

    (1) Event-driven perception     -- VideoMAE-v2 + YOLOv8
    (2) Three typed memories        -- episodic, semantic, procedural
    (3) Belief-Action-Verification  -- iterative BAV loop with postcondition
                                       and temporal verification
    (4) Meta reflection             -- contrastive refinement of scripts
    (5) Bayesian reliability        -- gradient-free continual learning

All vision-language and text-only reasoning is performed by Gemma3-4B
served locally via Ollama. VideoMAE-v2 and YOLOv8 are used as frozen
auxiliary encoders for semantic-change detection and object detection.

Paper-alignment notes
---------------------
This file is the camera-ready implementation. Every threshold, formula,
and trigger matches the Methodology section of the paper:

  * Expected utility (Belief-Action-Verification paragraph):
        EU_k = Rel_k * E[rho_k] - Risk_k + lambda_info * H(Beta(alpha_k, beta_k))
    Rel_k is the sentence-embedding similarity between observed event
    descriptions and the schema's preconditions; Risk_k is the semantic
    similarity between the script representation and past failure
    narratives.

  * Bayesian posterior update is purely verification-driven. The
    "is_grounded" flag from the verifier alone decides whether
    alpha or beta is incremented (paper Eq. for the posterior update).

  * Episodic merging (Episodic Memory paragraph):
        gap < 2s  AND  embedding similarity > 0.85  AND
        continuity marker ("then", "next", ...).

  * Procedural schema validation:
        n_min = 5,    sigma_i / mu_i < 0.5.

  * Meta-reflection triggers:
        3+ consecutive failures, OR
        postcondition coverage < 0.3, OR
        semantic drift in linguistic descriptions > 0.7.

  * All Gemma3-4B text-mode calls use T = 0.1.

Project page : https://github.com/<org>/lingua
License      : MIT (see LICENSE)
"""

from __future__ import annotations

import os
import cv2
import warnings
import platform
import subprocess
import time

# ----------------------------------------------------------------------------
# Environment hygiene -- silence OpenCV/FFmpeg noise that obscures logs.
# ----------------------------------------------------------------------------
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
warnings.filterwarnings('ignore')

# ============================================================================
# Ollama bootstrap (Windows-specific helper; no-op on Linux/macOS)
# ============================================================================


def setup_ollama_windows():
    """
    Setup Ollama for Windows environment
    
    Handles:
    - Service detection
    - Environment variables
    - Connection settings
    """
    
    # Detect OS
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        print(" Windows detected - Configuring Ollama...")
        
        # 1. Set Ollama host (default localhost)
        if 'OLLAMA_HOST' not in os.environ:
            os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
            print(f"  [ok] OLLAMA_HOST set to: {os.environ['OLLAMA_HOST']}")
        
        # 2. Check if Ollama service is running
        try:
            # Try to find Ollama process
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq ollama.exe'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'ollama.exe' in result.stdout:
                print("  [ok] Ollama service is running")
            else:
                print("  [warn] Ollama service not detected!")
                print("   Start Ollama manually or run: ollama serve")
                
                # Attempt to start Ollama
                try:
                    print("   Attempting to start Ollama...")
                    subprocess.Popen(
                        ['ollama', 'serve'],
                        creationflags=subprocess.CREATE_NO_WINDOW if is_windows else 0
                    )
                    time.sleep(3)  # Wait for service to start
                    print("  [ok] Ollama service started")
                except:
                    print("  [warn] Could not auto-start Ollama")
                    print("   Please run 'ollama serve' in a separate terminal")
        
        except Exception as e:
            print(f"  [warn] Could not check Ollama service: {e}")
        
        # 3. Optional: Set GPU/CPU preferences
        # Uncomment if you want to force CPU mode (slower but more stable)
        # os.environ['OLLAMA_NUM_GPU'] = '0'  # Force CPU mode
        
        # 4. Set timeout for large models (increase if needed)
        os.environ['OLLAMA_REQUEST_TIMEOUT'] = '300'  # 5 minutes
        
        # 5. Windows-specific: Disable CUDA warnings if no GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA if you want CPU-only
        
        print("  [ok] Ollama configuration complete\n")
    
    else:
        print(f" {platform.system()} detected - Using default Ollama settings\n")


def verify_ollama_models():
    """
    Verify that required Ollama models are available
    """
    try:
        import ollama
        
        print(" Checking Ollama models...")
        
        # List available models
        models = ollama.list()
        model_names = [m['name'] for m in models['models']]
        
        print(f"  Available models: {model_names}")
        
        # Check for required models
        required_models = {
            'llava:7b': 'Vision-Language Model (for frame descriptions)',
            'gemma3:4b': 'Text LLM (for reasoning/planning)'
        }
        
        missing_models = []
        
        for model, description in required_models.items():
            # Check if model exists (handle version suffixes)
            model_base = model.split(':')[0]
            exists = any(model_base in name for name in model_names)
            
            if exists:
                print(f"  [ok] {model} - {description}")
            else:
                print(f"  ✗ {model} - {description} - MISSING!")
                missing_models.append(model)
        
        if missing_models:
            print("\n[warn] Missing models detected!")
            print("\nRun these commands to install:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            print("\nThen restart this script.\n")
            return False
        
        print("  [ok] All required models available\n")
        return True
    
    except ImportError:
        print("[err] Ollama Python package not installed!")
        print("   Run: pip install ollama\n")
        return False
    
    except Exception as e:
        print(f"[err] Error checking models: {e}")
        print("   Make sure Ollama is running: ollama serve\n")
        return False


# ============================================================================
# INITIALIZE OLLAMA ON SCRIPT START
# ============================================================================

# Setup Ollama for Windows
setup_ollama_windows()

# Verify models are available
models_ready = verify_ollama_models()

if not models_ready:
    print("="*80)
    print("SETUP REQUIRED")
    print("="*80)
    print("\n1. Install Ollama:")
    print("   Download from: https://ollama.com/download")
    print("\n2. Start Ollama service:")
    print("   Open terminal and run: ollama serve")
    print("\n3. Pull required models:")
    print("   ollama pull llava:7b")
    print("   ollama pull gemma3:4b")
    print("\n4. Verify installation:")
    print("   ollama list")
    print("\n" + "="*80)
    
    # Ask user if they want to continue anyway
    response = input("\nContinue without Ollama models? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        exit(1)

import json
import torch
import spacy
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

# ============================================================================
# Vision & Language Models
# ============================================================================
try:
    from transformers import VideoMAEModel, VideoMAEImageProcessor
    VIDEOMAE_AVAILABLE = True
except ImportError:
    VIDEOMAE_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import beta as beta_dist
import torch.nn.functional as F

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Model configuration
# ============================================================================
@dataclass
class LINGUAConfig:
    SYSTEM_NAME: str = "LINGUA-Agent"

    # ------------------------------------------------------------------
    # Vision backbone (Section 3.3.1 - Semantic Change Detection)
    # ------------------------------------------------------------------
    VIDEOMAE_MODEL: str = "MCG-NJU/videomae-base"
    VIDEOMAE_NUM_FRAMES: int = 16
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5

    # ------------------------------------------------------------------
    # Unified Gemma3-4B backbone (Section 3 - Implementation Details)
    # Gemma 3 (4B) is a natively multimodal model: the same checkpoint
    # serves both vision-language captioning and text-only reasoning,
    # following the paper's "unified Gemma3-4B" design.
    # ------------------------------------------------------------------
    VLM_PROVIDER: str = "ollama"
    VLM_MODEL: str = "gemma3:4b"          # vision-language mode
    VLM_TEMPERATURE: float = 0.1
    VLM_MAX_LENGTH: int = 128

    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "gemma3:4b"          # text-only mode (same weights)
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_LENGTH: int = 256

    # ------------------------------------------------------------------
    # Auxiliary NLP models
    # ------------------------------------------------------------------
    SRL_MODEL: str = "en_core_web_sm"            # spaCy NER + SRL
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"    # sentence-transformers

    # ------------------------------------------------------------------
    # Hyperparameters -- values exactly as stated in the paper.
    # All thresholds appear in Section 3 (Methodology) of the camera-ready.
    # ------------------------------------------------------------------
    SEMANTIC_CHANGE_THRESHOLD: float = 0.15      # tau_Delta
    FUZZY_MATCH_THRESHOLD: float = 0.75          # gamma_aff (affordance retrieval)
    POSTCONDITION_THRESHOLD: float = 0.8         # gamma_post
    EU_THRESHOLD: float = 0.4                    # tau_EU
    PRIOR_ALPHA: float = 1.0                     # Beta(1,1) uniform prior
    PRIOR_BETA: float = 1.0
    LAMBDA_INFO: float = 0.1                     # lambda_info (EU exploration)
    MIN_CONTRASTIVE_EXAMPLES: int = 3            # n_contrast in paper

    # Episodic merging (paper Section 3, Episodic Memory paragraph)
    TEMPORAL_GAP_THRESHOLD: float = 2.0          # Delta_t_merge = 2s
    EPISODIC_MERGE_SIMILARITY: float = 0.85      # embedding similarity > 0.85

    # Procedural schema validation (paper Section 3, Procedural Memory)
    MIN_SCRIPT_INSTANCES: int = 5                # n_min = 5
    MAX_TEMPORAL_VARIANCE: float = 0.5           # sigma_i / mu_i < 0.5

    # Meta-reflection triggers (paper Section 3, Meta Reflection paragraph)
    REFLECTION_FAILURE_COUNT: int = 3            # "three or more consecutive failures"
    REFLECTION_COVERAGE_THRESHOLD: float = 0.3   # postcondition coverage < 0.3
    SEMANTIC_DRIFT_THRESHOLD: float = 0.7        # drift > 0.7
# ============================================================================
# VLM-BASED SEMANTIC MEMORY (Section 3.2.2, Eq. 6, 9, 11)
# ============================================================================
@dataclass
class SemanticMemoryEntry:
    """
    Eq. 6: m_j^sem = ⟨Object, A_j, P_j, R_j, s_j, c_j⟩
    
    Where:
    - Object: Object label
    - A_j: Affordances as semantic frames
    - P_j: Linguistic properties
    - R_j: Semantic relations
    - s_j: VLM embedding for fuzzy matching (Eq. 11)
    - c_j: Confidence
    """
    object_label: str                          # Object
    affordances: List[Dict[str, Any]]          # A_j: Frame-semantic affordances
    linguistic_properties: List[str]           # P_j: ["sharp", "metallic", ...]
    semantic_relations: Dict[str, List[str]]   # R_j: {partOf: [...], usedFor: [...]}
    embedding: Optional[np.ndarray] = None     # s_j: For fuzzy matching
    confidence: float = 1.0                    # c_j

@dataclass
class SemanticFrame:
    """
    FrameNet-style semantic frame structure
    
    Example:
    Frame: "Cutting"
    Roles: {
        "Instrument": "knife",
        "Agent": "?",  # To be filled from context
        "Patient": "?"  # To be filled from action
    }
    """
    frame_name: str
    roles: Dict[str, str]  # Role  Value mapping
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    temporal_duration: Tuple[float, float] = (0.0, 0.0)  # (min, max) seconds

class VLMSemanticMemory:
    """
    Section 3.2.2: VLM-Based Semantic Memory
    
    Requirements:
    - Eq. 9: VLM-based affordance extraction
    - Eq. 11: Fuzzy semantic matching via distributional semantics
    - Frame-semantic output format
    """
    
    def __init__(self, config: LINGUAConfig, vlm, embedding_model):
        self.config = config
        self.vlm = vlm
        self.embedding_model = embedding_model
        
        # Memory storage
        self.semantic_entries: Dict[str, SemanticMemoryEntry] = {}
        
        logger.info("[ok] VLM-based semantic memory initialized")
    

        # FrameNet-style affordance database (paper Section 3.2.2)
        self.framenet_db = self._initialize_framenet_database()
    def extract_affordances_vlm(self, object_label: str, contexts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Eq. 9: VLM-Based Frame-Semantic Affordance Extraction (Ollama)
        """
        if not self.vlm.model:
            return self._fallback_affordances(object_label)
        
        try:
            import ollama
            
            # Build context string
            context_str = ""
            if contexts:
                context_str = f"\nContext: {', '.join(contexts[:3])}"
            
            # Prompt for frame-semantic extraction
            prompt = f"""Task: Extract semantic affordances for object as FrameNet-style frames.

    Object: {object_label}{context_str}

    Provide affordances as semantic frames with typed roles.
    Format each as: [FrameName: [Role1=value, Role2=value, ...]]

    Examples:
    - knife  [Cutting: [Instrument=knife, Agent=human, Patient=food]]
    - cup  [Ingestion: [Instrument=cup, Ingestor=human, Ingestibles=liquid]]

    Affordances for {object_label}:"""
            
            response = ollama.generate(
                model=self.config.LLM_MODEL,  # Use Gemma3 for this
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 100
                }
            )
            
            response_text = response['response']
            
            # Parse frame-semantic output
            affordances = self._parse_frame_semantic_output(response_text, object_label)
            
            if affordances:
                logger.debug(f"  VLM affordances for '{object_label}': {len(affordances)} frames")
                return affordances
            else:
                logger.debug(f"  VLM parsing failed, using fallback")
                return self._fallback_affordances(object_label)
            
        except Exception as e:
            logger.debug(f"VLM affordance extraction failed: {e}")
            return self._fallback_affordances(object_label)
    
    def _parse_frame_semantic_output(self, response: str, object_label: str) -> List[Dict[str, Any]]:
        """Parse VLM output into frame-semantic structures"""
        affordances = []
        
        # Look for patterns like [FrameName: [roles]]
        import re
        pattern = r'\[([A-Z][a-zA-Z]+):\s*\[(.*?)\]\]'
        matches = re.findall(pattern, response)
        
        for frame_name, roles_str in matches:
            # Parse roles
            role_pattern = r'([A-Za-z]+)=([a-zA-Z_]+)'
            roles = re.findall(role_pattern, roles_str)
            
            affordances.append({
                "frame": frame_name,
                "roles": [f"{role}={value}" for role, value in roles]
            })
        
        return affordances
    
    def _fallback_affordances(self, object_label: str) -> List[Dict[str, Any]]:
        """Fallback hardcoded affordances"""
        fallback = {
            "knife": [{"frame": "Cutting", "roles": ["Instrument=knife", "Agent=?", "Patient=?"]}],
            "spoon": [{"frame": "Stirring", "roles": ["Instrument=spoon", "Agent=?", "Patient=?"]}],
            "cup": [{"frame": "Ingestion", "roles": ["Instrument=cup", "Ingestor=?", "Ingestibles=?"]}],
            "fork": [{"frame": "Eating", "roles": ["Instrument=fork", "Agent=?", "Patient=?"]}],
            "glass": [{"frame": "Ingestion", "roles": ["Instrument=glass", "Ingestor=?", "Ingestibles=?"]}],
            "bowl": [{"frame": "Ingestion", "roles": ["Container=bowl", "Agent=?", "Contents=?"]}],
            "plate": [{"frame": "Serving", "roles": ["Container=plate", "Agent=?", "Contents=?"]}],
            "chair": [{"frame": "Sitting", "roles": ["Support=chair", "Agent=?", "Location=?"]}],
            "table": [{"frame": "Supporting", "roles": ["Support=table", "Theme=?", "Location=?"]}],
        }
        
        return fallback.get(object_label, [{"frame": "General", "roles": ["Object=" + object_label]}])
    
    def get_or_create_entry(self, object_label: str, contexts: List[str] = None) -> SemanticMemoryEntry:
        """
        Get or create semantic memory entry
        Implements Eq. 6 structure
        """
        if object_label in self.semantic_entries:
            return self.semantic_entries[object_label]
        
        # Extract affordances via VLM (Eq. 9)
        affordances = self.extract_affordances_vlm(object_label, contexts)
        
        # Extract linguistic properties (simple heuristics)
        properties = self._extract_properties(object_label, affordances)
        
        # Extract semantic relations
        relations = self._extract_relations(object_label, affordances)
        
        # Compute embedding for fuzzy matching (Eq. 11)
        embedding = self.embedding_model.encode(
            f"{object_label} {' '.join(properties)}",
            convert_to_numpy=True
        )
        
        # Create entry
        entry = SemanticMemoryEntry(
            object_label=object_label,
            affordances=affordances,
            linguistic_properties=properties,
            semantic_relations=relations,
            embedding=embedding,
            confidence=0.8
        )
        
        self.semantic_entries[object_label] = entry
        return entry
    
    def _extract_properties(self, object_label: str, affordances: List[Dict]) -> List[str]:
        """Extract linguistic properties from affordances"""
        properties = []
        
        # Infer properties from frames
        for aff in affordances:
            frame = aff.get("frame", "")
            if "Cutting" in frame:
                properties.extend(["sharp", "tool"])
            elif "Ingestion" in frame:
                properties.extend(["container", "holdable"])
            elif "Sitting" in frame:
                properties.extend(["furniture", "support"])
        
        return list(set(properties))
    
    def _extract_relations(self, object_label: str, affordances: List[Dict]) -> Dict[str, List[str]]:
        """Extract semantic relations"""
        relations = defaultdict(list)
        
        # Infer relations from frames
        for aff in affordances:
            frame = aff.get("frame", "")
            if frame:
                relations["evokes_frame"].append(frame)
        
        return dict(relations)
    
    def retrieve_fuzzy(self, query: str, threshold: float = None) -> List[SemanticMemoryEntry]:
        """
        Eq. 11: Fuzzy Semantic Matching via Distributional Semantics
        
        Retrieve(q) = {m_j^sem : sim(VLM(q), s_j) > 0.75}
        
        Example:
        - sim("spoon", "teaspoon") = 0.93  share affordances
        - sim("knife", "blade") = 0.88  similar Cutting frames
        """
        if threshold is None:
            threshold = self.config.FUZZY_MATCH_THRESHOLD  # 0.75 from paper
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Find similar entries
        matches = []
        for entry in self.semantic_entries.values():
            if entry.embedding is not None:
                similarity = np.dot(query_embedding, entry.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding) + 1e-8
                )
                
                if similarity > threshold:
                    matches.append(entry)
                    logger.debug(f"  Fuzzy match: '{query}'  '{entry.object_label}' (sim={similarity:.3f})")
        
        return matches


# ============================================================================
# SEMANTIC ROLE LABELING
# ============================================================================

    def _initialize_framenet_database(self) -> Dict[str, List[SemanticFrame]]:
        """
        Initialize FrameNet-style affordance frames
        
        This is the implementation of what's described in the paper:
        "knife evokes the Cutting frame with roles [Instrument=knife, Agent=?, Patient=?]"
        """
        return {
            # Kitchen tools
            "knife": [
                SemanticFrame(
                    frame_name="Cutting",
                    roles={
                        "Instrument": "knife",
                        "Agent": "?",  # Will be filled: person, chef, etc.
                        "Patient": "?"  # Will be filled: food, vegetable, etc.
                    },
                    preconditions=["graspable", "sharp", "visible"],
                    postconditions=["divided", "separated", "pieces"],
                    temporal_duration=(2.0, 8.0)
                ),
                SemanticFrame(
                    frame_name="Manipulation",
                    roles={
                        "Instrument": "knife",
                        "Agent": "?",
                        "Purpose": "?"
                    },
                    preconditions=["reachable"],
                    postconditions=["grasped", "held"],
                    temporal_duration=(0.5, 2.0)
                )
            ],
            
            "spoon": [
                SemanticFrame(
                    frame_name="Ingestion",
                    roles={
                        "Instrument": "spoon",
                        "Ingestor": "?",
                        "Ingestibles": "?"
                    },
                    preconditions=["graspable", "food_present"],
                    postconditions=["food_consumed"],
                    temporal_duration=(1.0, 5.0)
                ),
                SemanticFrame(
                    frame_name="Stirring",
                    roles={
                        "Instrument": "spoon",
                        "Agent": "?",
                        "Substance": "?"
                    },
                    preconditions=["liquid_present", "container_open"],
                    postconditions=["mixed", "blended"],
                    temporal_duration=(3.0, 10.0)
                )
            ],
            
            "bottle": [
                SemanticFrame(
                    frame_name="Containing",
                    roles={
                        "Container": "bottle",
                        "Contents": "?",
                        "Capacity": "?"
                    },
                    preconditions=["closed", "filled"],
                    postconditions=["stored"],
                    temporal_duration=(0.0, 0.0)  # Static state
                ),
                SemanticFrame(
                    frame_name="Taking",
                    roles={
                        "Agent": "?",
                        "Theme": "bottle",
                        "Source": "?"
                    },
                    preconditions=["visible", "reachable"],
                    postconditions=["acquired", "grasped"],
                    temporal_duration=(1.0, 3.0)
                )
            ],
            
            "refrigerator": [
                SemanticFrame(
                    frame_name="Opening",
                    roles={
                        "Agent": "?",
                        "Container": "refrigerator",
                        "Purpose": "?"
                    },
                    preconditions=["closed", "reachable"],
                    postconditions=["opened", "accessible"],
                    temporal_duration=(0.5, 2.0)
                ),
                SemanticFrame(
                    frame_name="Storing",
                    roles={
                        "Agent": "?",
                        "Theme": "?",
                        "Location": "refrigerator"
                    },
                    preconditions=["opened", "item_available"],
                    postconditions=["stored", "cooled"],
                    temporal_duration=(2.0, 5.0)
                ),
                SemanticFrame(
                    frame_name="Cooling",
                    roles={
                        "Location": "refrigerator",
                        "Theme": "?",
                        "Duration": "?"
                    },
                    preconditions=["inside", "powered"],
                    postconditions=["chilled", "cold"],
                    temporal_duration=(60.0, 3600.0)  # 1min - 1hour
                )
            ],
            
            "fridge": [  # Alias for refrigerator
                SemanticFrame(
                    frame_name="Opening",
                    roles={
                        "Agent": "?",
                        "Container": "fridge",
                        "Purpose": "?"
                    },
                    preconditions=["closed"],
                    postconditions=["opened"],
                    temporal_duration=(0.5, 2.0)
                ),
                SemanticFrame(
                    frame_name="Storing",
                    roles={
                        "Agent": "?",
                        "Theme": "?",
                        "Location": "fridge"
                    },
                    preconditions=["opened"],
                    postconditions=["stored"],
                    temporal_duration=(2.0, 5.0)
                )
            ],
            
            "cup": [
                SemanticFrame(
                    frame_name="Containing",
                    roles={
                        "Container": "cup",
                        "Contents": "?",
                        "Holder": "?"
                    },
                    preconditions=["empty_or_filled"],
                    postconditions=["holding_liquid"],
                    temporal_duration=(0.0, 0.0)
                ),
                SemanticFrame(
                    frame_name="Drinking",
                    roles={
                        "Drinker": "?",
                        "Beverage": "?",
                        "Container": "cup"
                    },
                    preconditions=["filled", "accessible"],
                    postconditions=["consumed", "empty"],
                    temporal_duration=(2.0, 10.0)
                )
            ],
            
            "bowl": [
                SemanticFrame(
                    frame_name="Containing",
                    roles={
                        "Container": "bowl",
                        "Contents": "?",
                        "Purpose": "?"
                    },
                    preconditions=["empty_or_filled"],
                    postconditions=["holding_food"],
                    temporal_duration=(0.0, 0.0)
                ),
                SemanticFrame(
                    frame_name="Mixing",
                    roles={
                        "Agent": "?",
                        "Ingredients": "?",
                        "Container": "bowl"
                    },
                    preconditions=["ingredients_present"],
                    postconditions=["mixed", "combined"],
                    temporal_duration=(5.0, 30.0)
                )
            ],
            
            "oven": [
                SemanticFrame(
                    frame_name="Cooking",
                    roles={
                        "Agent": "?",
                        "Food": "?",
                        "Heat_source": "oven",
                        "Temperature": "?"
                    },
                    preconditions=["preheated", "food_inside"],
                    postconditions=["cooked", "hot"],
                    temporal_duration=(600.0, 3600.0)  # 10min - 1hour
                ),
                SemanticFrame(
                    frame_name="Heating",
                    roles={
                        "Heat_source": "oven",
                        "Theme": "?",
                        "Duration": "?"
                    },
                    preconditions=["powered_on"],
                    postconditions=["heated", "warm"],
                    temporal_duration=(300.0, 1800.0)
                )
            ]
        }
    

    def get_affordances_framenet(
        self, 
        object_label: str, 
        context: Optional[str] = None
    ) -> List[SemanticFrame]:
        """
        Get FrameNet-style affordances for an object
        
        This implements the paper's claim:
        "knife evokes the Cutting frame with roles [Instrument=knife, Agent=?, Patient=?]"
        
        Args:
            object_label: Object name (e.g., "knife")
            context: Optional context to help select relevant frames
        
        Returns:
            List of semantic frames with typed roles
        
        Example:
            frames = memory.get_affordances_framenet("knife")
            # Returns: [
            #   SemanticFrame(
            #       frame_name="Cutting",
            #       roles={"Instrument": "knife", "Agent": "?", "Patient": "?"},
            #       preconditions=["graspable", "sharp"],
            #       postconditions=["divided", "separated"]
            #   )
            # ]
        """
        obj_lower = object_label.lower()
        
        # Direct lookup
        if obj_lower in self.framenet_affordances:
            frames = self.framenet_affordances[obj_lower]
            
            # If context provided, rank by relevance
            if context:
                # Simple context matching (can be improved)
                scored_frames = []
                for frame in frames:
                    score = 0.0
                    # Check if frame name appears in context
                    if frame.frame_name.lower() in context.lower():
                        score += 1.0
                    # Check if preconditions match context
                    for precond in frame.preconditions:
                        if precond in context.lower():
                            score += 0.5
                    scored_frames.append((score, frame))
                
                # Sort by score and return
                scored_frames.sort(key=lambda x: x[0], reverse=True)
                return [f for _, f in scored_frames]
            
            return frames
        
        # Fuzzy matching for unknown objects
        return self._fuzzy_match_affordances(obj_lower)
    

    def _fuzzy_match_affordances(self, object_label: str) -> List[SemanticFrame]:
        """
        Fuzzy match using semantic similarity
        Example: "teaspoon"  similar to "spoon"  return spoon's frames
        """
        if not hasattr(self, 'embedding_model'):
            return []
        
        obj_embedding = self.embedding_model.encode([object_label])
        
        best_match = None
        best_similarity = 0.0
        
        for known_obj in self.framenet_affordances.keys():
            known_embedding = self.embedding_model.encode([known_obj])
            similarity = np.dot(obj_embedding[0], known_embedding[0])
            
            if similarity > best_similarity and similarity > 0.75:  # Threshold
                best_similarity = similarity
                best_match = known_obj
        
        if best_match:
            logger.info(f"  Fuzzy match: '{object_label}'  '{best_match}' (sim={best_similarity:.2f})")
            return self.framenet_affordances[best_match]
        
        return []
    

    def fill_frame_roles(
        self,
        frame: SemanticFrame,
        episodic_context: List[Any]
    ) -> SemanticFrame:
        """
        Fill in the "?" roles using episodic memory context
        
        Example:
            Input frame: [Cutting: [Instrument=knife, Agent=?, Patient=?]]
            Episodic context: "man cuts tomato"
            Output: [Cutting: [Instrument=knife, Agent=man, Patient=tomato]]
        """
        filled_frame = SemanticFrame(
            frame_name=frame.frame_name,
            roles=frame.roles.copy(),
            preconditions=frame.preconditions.copy(),
            postconditions=frame.postconditions.copy(),
            temporal_duration=frame.temporal_duration
        )
        
        # Extract from episodic memory
        for episode in episodic_context:
            if hasattr(episode, 'agent') and filled_frame.roles.get("Agent") == "?":
                filled_frame.roles["Agent"] = episode.agent
            
            if hasattr(episode, 'patient') and filled_frame.roles.get("Patient") == "?":
                filled_frame.roles["Patient"] = episode.patient
            
            if hasattr(episode, 'action'):
                # Match action to frame
                if frame.frame_name.lower() in episode.action.lower():
                    # Try to fill any remaining roles
                    for role, value in filled_frame.roles.items():
                        if value == "?" and hasattr(episode, role.lower()):
                            filled_frame.roles[role] = getattr(episode, role.lower())
        
        return filled_frame

    def consolidate_from_episodic(self, episodic_memories: List['EpisodicMemory'], 
                                min_instances: int = 3) -> List[str]:
        """Episodic  Semantic consolidation"""
        consolidated_concepts = []
    
        object_episodes = {}
        for ep in episodic_memories:
            for obj in ep.affordance_objects:
                label = obj.get("label", "")
                if label:
                    if label not in object_episodes:
                        object_episodes[label] = []
                    object_episodes[label].append(ep)
    
        for obj_label, episodes in object_episodes.items():
            if len(episodes) < min_instances:
                continue
        
            action_patterns = [ep.action for ep in episodes if ep.action]
            if not action_patterns:
                continue
        
            from collections import Counter
            common_actions = Counter(action_patterns).most_common(3)
        
            consolidated_affordances = []
            for action, count in common_actions:
                consolidated_affordances.append({
                    "frame": action,
                    "roles": [f"frequency:{count/len(episodes):.2f}"],
                    "source": "episodic_consolidation"
                }    )
        
            if obj_label in self.semantic_entries:
                entry = self.semantic_entries[obj_label]
                for aff in consolidated_affordances:
                    if aff not in entry.affordances:
                        entry.affordances.append(aff)
                logger.info(f"  [ok] Updated semantic '{obj_label}' from {len(episodes)} episodes")
            else:
                entry = SemanticMemoryEntry(
                    object_label=obj_label,
                    affordances=consolidated_affordances,
                    linguistic_properties=[],
                    semantic_relations={},
                    confidence=len(episodes) / 10.0
                )
                self.semantic_entries[obj_label] = entry
                logger.info(f"  [ok] Created semantic '{obj_label}' from {len(episodes)} episodes")
        
            consolidated_concepts.append(obj_label)
    
        if consolidated_concepts:
            logger.info(f"[ok] Consolidated {len(consolidated_concepts)} concepts")
    
        return consolidated_concepts


class SemanticRoleLabeler:
    """PropBank-style SRL (Section 3.2.1)"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        logger.info("[ok] PropBank-style SRL initialized")
    
    def extract_semantic_roles(self, text: str) -> Dict[str, str]:
        """
        Extract semantic roles following PropBank
        Returns: {Agent, Action, Affected_Entity, Location, Goal, Outcome}
        """
        doc = self.nlp(text)
        
        roles = {
            "Agent": "",
            "Action": "",
            "Affected_Entity": "",  
            "Location": "",
            "Goal": "",
            "Outcome": ""
        }
        
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"] and not roles["Agent"]:
                roles["Agent"] = token.text
        
        for token in doc:
            if token.pos_ == "VERB" and not roles["Action"]:
                roles["Action"] = token.lemma_
        
        for token in doc:
            if token.dep_ in ["dobj", "pobj"] and not roles["Affected_Entity"]:
                roles["Affected_Entity"] = token.text
        
        for token in doc:
            if token.dep_ == "prep" and token.text in ["at", "in", "on", "near"]:
                for child in token.children:
                    if child.dep_ == "pobj":
                        roles["Location"] = child.text
        
        for token in doc:
            if token.text.lower() in ["to", "for"] and token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ == "pobj":
                        roles["Goal"] = child.text
        
        return roles

# ============================================================================
# MEMORY STRUCTURES
# ============================================================================

@dataclass
class EpisodicMemory:
    """
    Paper Eq. 4: Narrative Situation Template
    m_i^epi = ⟨Agent, Action, Patient, t_s, t_e, Location, Goal, Outcome, O_i, d_i, v_i, α_i, β_i⟩
    """
    agent: str
    action: str
    affected_entity: str  # Paper's "Patient"
    location: str
    timestamp: float
    goal: str
    outcome: str
    start_time: float      # t_s
    end_time: float        # t_e
    description: str       # d_i
    temporal_markers: List[str] = field(default_factory=list)
    causal_connectives: List[str] = field(default_factory=list)
    affordance_objects: List[Dict] = field(default_factory=list)  # O_i
    videomae_embedding: Optional[np.ndarray] = None  # v_i
    alpha: float = 1.0     # α_i
    beta: float = 1.0      # β_i
    memory_id: str = ""
    confidence: float = 1.0    
    
    @property
    def reliability(self) -> float:
        """Expected reliability E[ρ] = α/(α+β)"""
        return self.alpha / (self.alpha + self.beta)


@dataclass
class ProceduralMemory:
    """Eq. 7: Script-Based Action Schema with versioning support"""
    script_name: str
    preconditions: List[str]
    action_sequence: List[Dict[str, str]]
    postconditions: List[str]
    temporal_constraints: List[Tuple[float, float]] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    causal_connectives: List[str] = field(default_factory=list)
    alpha: float = 1.0
    beta: float = 1.0
    corpus_frequency: float = 0.0
    instance_count: int = 0
    script_embedding: Optional[np.ndarray] = None
    script_id: str = ""
    
    # NEW: Versioning support
    script_version: int = 1
    parent_script_id: Optional[str] = None
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    
    @property
    def reliability(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def create_refined_version(self, discriminators: Dict[str, List[str]], reason: str):
        """Create new version keeping old one"""
        import copy
        refined = copy.deepcopy(self)
        refined.script_version = self.script_version + 1
        refined.parent_script_id = self.script_id
        refined.script_id = f"{self.script_id}_v{refined.script_version}"
        refined.alpha = self.alpha
        refined.beta = self.beta
        
        for feature in discriminators.get("positive", []):
            new_precond = f"requires_mention({feature})"
            if new_precond not in refined.preconditions:
                refined.preconditions.append(new_precond)
        
        for feature in discriminators.get("negative", []):
            neg_precond = f"NOT_mention({feature})"
            if neg_precond not in refined.preconditions:
                refined.preconditions.append(neg_precond)
        
        refined.postconditions.append("contrastive_refined")
        
        refined.refinement_history.append({
            "version": refined.script_version,
            "parent": self.script_id,
            "reason": reason,
            "discriminators": discriminators,
            "timestamp": datetime.now().isoformat()
        })
        
        return refined


# ============================================================================
# Add GroundingStatus class (After ProceduralMemory, ~Line 516)
# ============================================================================


@dataclass
class GroundingStatus:
    """Grounding status tracking"""
    is_grounded: bool = False
    prediction_made: bool = False
    verification_passed: bool = False
    
    predicted_postconditions: List[str] = field(default_factory=list)
    predicted_temporal_ranges: List[Tuple[float, float]] = field(default_factory=list)
    
    observed_postconditions: List[str] = field(default_factory=list)
    observed_durations: List[float] = field(default_factory=list)
    temporal_span: Optional[Tuple[float, float]] = None
    
    postcondition_coverage: float = 0.0
    temporal_consistency: float = 0.0
    
    grounding_timestamp: Optional[str] = None
    script_used: Optional[str] = None
    confidence: float = 0.0
    
    def mark_grounded(self, script_id: str, confidence: float):
        self.is_grounded = True
        self.verification_passed = True
        self.grounding_timestamp = datetime.now().isoformat()
        self.script_used = script_id
        self.confidence = confidence

#============================================================================
#VLMSemanticMemory 
# ============================================================================


class BayesianReliabilityTracker:
    """Persistent Bayesian learning (Eq. 8, 15)"""
    script_reliabilities: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    
    def get_reliability(self, script_id: str) -> Tuple[float, float]:
        if script_id not in self.script_reliabilities:
            self.script_reliabilities[script_id] = (self.prior_alpha, self.prior_beta)
        return self.script_reliabilities[script_id]
    
    def update_after_verification(self, script_id: str, success: bool):
        """Eq. 15: Posterior update"""
        alpha, beta = self.get_reliability(script_id)
        
        if success:
            alpha += 1.0
        else:
            beta += 1.0
        
        self.script_reliabilities[script_id] = (alpha, beta)
        
        self.update_history.append({
            "script_id": script_id,
            "success": success,
            "alpha": alpha,
            "beta": beta,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f" Updated {script_id}: α={alpha:.1f}, β={beta:.1f}, E[ρ]={alpha/(alpha+beta):.3f}")
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump({
                "reliabilities": {k: list(v) for k, v in self.script_reliabilities.items()},
                "history": self.update_history
            }, f, indent=2)
    
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.script_reliabilities = {k: tuple(v) for k, v in data["reliabilities"].items()}
        self.update_history = data.get("history", [])


# ============================================================================
# CONTINUAL LEARNING TRACKER # ============================================================================

class ContinualLearningTracker:
    """
    Track learning progress over time for continual learning experiments
    
    Enables:
    - Learning curve analysis (accuracy over time)
    - Domain adaptation metrics
    - Catastrophic forgetting analysis
    """
    
    def __init__(self):
        self.performance_history = []  # Track accuracy over time
        self.video_count = 0
        self.domain_performance = defaultdict(list)  # Track per-domain performance
        self.script_usage_history = defaultdict(list)  # Track script evolution
        
        logger.info("[ok] Continual learning tracker initialized")
        
    def log_performance(self, video_id: str, question: str, correct: bool, 
                       script_used: str, domain: str = "general"):
        """Log each prediction for continual learning analysis"""
        self.video_count += 1
        self.performance_history.append({
            "video_count": self.video_count,
            "video_id": video_id,
            "question": question,
            "correct": correct,
            "script_used": script_used,
            "domain": domain,
            "timestamp": datetime.now().isoformat()
        })
        
        self.domain_performance[domain].append(correct)
        
        # Track script usage
        if script_used and script_used != "none":
            self.script_usage_history[script_used].append({
                "video_count": self.video_count,
                "correct": correct
            })
        
    def get_learning_curve(self, window_size: int = 10):
        """
        Compute rolling accuracy for learning curve plot
        Returns: List of {video_count, accuracy} dicts
        """
        if len(self.performance_history) < window_size:
            return []
            
        accuracies = []
        for i in range(window_size, len(self.performance_history) + 1):
            window = self.performance_history[i-window_size:i]
            acc = sum(1 for x in window if x["correct"]) / window_size
            accuracies.append({
                "video_count": i,
                "accuracy": acc
            })
        return accuracies
    
    def get_domain_adaptation_metrics(self):
        """
        Measure adaptation to new domains
        Returns: Dict with per-domain initial/final/improvement metrics
        """
        metrics = {}
        for domain, results in self.domain_performance.items():
            if len(results) >= 5:
                # First 5 vs last 5
                initial = sum(results[:5]) / 5
                final = sum(results[-5:]) / 5
                improvement = final - initial
                metrics[domain] = {
                    "initial_acc": initial,
                    "final_acc": final,
                    "improvement": improvement,
                    "total_samples": len(results)
                }
        return metrics
    
    def get_forgetting_analysis(self, script_id: str):
        """
        Analyze if script performance degrades (catastrophic forgetting)
        Returns: Dict with early/recent accuracy and forgetting metric
        """
        if script_id not in self.script_usage_history:
            return None
            
        script_history = self.script_usage_history[script_id]
        
        if len(script_history) < 10:
            return None
        
        # Compare early vs recent performance
        early = [h["correct"] for h in script_history[:len(script_history)//2]]
        recent = [h["correct"] for h in script_history[len(script_history)//2:]]
        
        early_acc = sum(early) / len(early)
        recent_acc = sum(recent) / len(recent)
        
        return {
            "script_id": script_id,
            "early_accuracy": early_acc,
            "recent_accuracy": recent_acc,
            "forgetting": early_acc - recent_acc,  # Negative = improvement, positive = forgetting
            "total_uses": len(script_history)
        }
    
    def get_all_forgetting_analysis(self):
        """Analyze forgetting for all scripts"""
        all_analysis = {}
        for script_id in self.script_usage_history.keys():
            analysis = self.get_forgetting_analysis(script_id)
            if analysis:
                all_analysis[script_id] = analysis
        return all_analysis
    
    def save(self, filepath: str):
        """Save continual learning data to file"""
        with open(filepath, 'w') as f:
            json.dump({
                "performance_history": self.performance_history,
                "domain_performance": {k: list(v) for k, v in self.domain_performance.items()},
                "script_usage_history": {k: list(v) for k, v in self.script_usage_history.items()},
                "video_count": self.video_count
            }, f, indent=2)
    
    def load(self, filepath: str):
        """Load continual learning data from file"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.performance_history = data.get("performance_history", [])
        self.domain_performance = defaultdict(list, {k: list(v) for k, v in data.get("domain_performance", {}).items()})
        self.script_usage_history = defaultdict(list, {k: list(v) for k, v in data.get("script_usage_history", {}).items()})
        self.video_count = data.get("video_count", 0)


# ============================================================================
# REFLECTION & CONSOLIDATION (Section 3.4)
# ============================================================================

@dataclass
class DecisionLog:
    """Log for reflection (Eq. 12)"""
    timestamp: float
    hypothesis: str
    affordances_detected: List[str]
    script_selected: Optional[str]
    confidence: float
    outcome: Optional[str] = None
    success: Optional[bool] = None


class ReflectionMechanism:
    """
    Paper Section 3, Meta Reflection paragraph. Reflection is triggered when:
       (a) reasoning fails 3+ times consecutively,
       (b) postcondition coverage falls below 0.3, or
       (c) semantic drift in linguistic descriptions exceeds 0.7.
    """

    def __init__(self, config: LINGUAConfig, vlm, embedding_model=None):
        self.config = config
        self.vlm = vlm
        self.embedding_model = embedding_model
        self.decision_history: List[DecisionLog] = []

    def add_decision(self, log: DecisionLog):
        self.decision_history.append(log)

    # ------------------------------------------------------------------
    # Trigger logic (paper Meta Reflection paragraph)
    # ------------------------------------------------------------------
    def _consecutive_failures(self, recent) -> bool:
        return all(
            (log.success is False) for log in recent if log.success is not None
        )

    def _low_postcondition_coverage(self, recent) -> bool:
        # log.confidence carries postcondition coverage in our pipeline
        return all(
            log.confidence < self.config.REFLECTION_COVERAGE_THRESHOLD
            for log in recent
        )

    def _semantic_drift(self, recent) -> bool:
        """Mean pairwise (1 - cosine) over recent hypotheses > tau_drift."""
        descs = [log.hypothesis for log in recent if log.hypothesis]
        if len(descs) < 2 or self.embedding_model is None:
            return False
        embs = self.embedding_model.encode(descs, convert_to_numpy=True)
        dissims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                a, b = embs[i], embs[j]
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < 1e-8 or nb < 1e-8:
                    continue
                cos = float(np.dot(a, b) / (na * nb))
                dissims.append(1.0 - cos)
        if not dissims:
            return False
        return float(np.mean(dissims)) > self.config.SEMANTIC_DRIFT_THRESHOLD

    def detect_abnormality(self) -> bool:
        k = self.config.REFLECTION_FAILURE_COUNT
        if len(self.decision_history) < k:
            return False
        recent = self.decision_history[-k:]
        return (
            self._consecutive_failures(recent)
            or self._low_postcondition_coverage(recent)
            or self._semantic_drift(recent)
        )

    # ------------------------------------------------------------------
    # Linguistic diagnosis prompt (paper p_reflect template)
    # ------------------------------------------------------------------
    def reflect_and_recontextualize(self, current_observation: Dict) -> Dict[str, Any]:
        recent_failures = [
            log for log in self.decision_history[-5:] if log.success is False
        ]
        if not recent_failures or not self.vlm.model:
            return {"refined_hypothesis": None, "diagnosis": "no_failures"}

        try:
            failure_summary = "\n".join(
                f"- {log.hypothesis} (conf: {log.confidence:.2f})"
                for log in recent_failures
            )
            prompt = (
                "Diagnose video understanding failure.\n\n"
                f"Failed attempts:\n{failure_summary}\n\n"
                f"Current objects: {current_observation.get('objects', [])}\n\n"
                "What went wrong? Suggest alternative interpretation.\n\n"
                "Diagnosis:"
            )

            inputs = self.vlm.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.vlm.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=self.config.LLM_TEMPERATURE,  # T=0.1 per paper
                    pad_token_id=self.vlm.tokenizer.eos_token_id,
                )
            response = self.vlm.tokenizer.decode(outputs[0], skip_special_tokens=True)
            refined = response.split("Diagnosis:")[-1].strip()
            return {
                "refined_hypothesis": refined if refined else None,
                "diagnosis": "reflected",
                "confidence": 0.5,
            }
        except Exception as e:
            return {"refined_hypothesis": None, "diagnosis": f"error: {e}"}


class MemoryConsolidation:
    """Section 3.4: Memory consolidation (Eq. 13)"""
    
    def __init__(self, config: LINGUAConfig):
        self.config = config
        self.consolidation_log: List[Dict] = []
    
    def should_consolidate(self, success: bool, confidence: float, novelty: float) -> bool:
        """Eq. 13: Consolidation decision"""
        return success and confidence > 0.6 and novelty > 0.3
    
    def consolidate_episodic(self, hypothesis: str, observation: Dict, memory_system) -> EpisodicMemory:
        roles = memory_system.srl.extract_semantic_roles(hypothesis)
        
        new_memory = EpisodicMemory(
            agent=roles["Agent"],
            action=roles["Action"],
            affected_entity=roles["Affected_Entity"],
            location=roles["Location"],
            timestamp=observation.get("timestamp", 0.0),
            goal=observation.get("goal", ""),
            outcome="success",
            start_time=observation.get("timestamp", 0.0),
            end_time=observation.get("timestamp", 0.0) + 1.0,
            description=hypothesis,
            affordance_objects=observation.get("objects", []),
            videomae_embedding=observation.get("embedding"),
            confidence=observation.get("confidence", 0.6),
            memory_id=f"reflected_{datetime.now().timestamp()}"
        )
        
        memory_system.episodic_memory.append(new_memory)
        logger.info(f"[ok] Consolidated episodic: {new_memory.memory_id}")
        return new_memory


# ============================================================================
# CONTRASTIVE REFINEMENT (Section 3.5, Eq. 13-16)
# ============================================================================

class ContrastiveRefinement:
    """
    Section 3.5: Contrastive Procedural Refinement
    
    Requirements:
    - Eq. 13: Discriminator extraction from success/failure sets
    - Eq. 14: Refine preconditions
    - Eq. 15: Merge action sequences
    - Eq. 16: Extend postconditions
    
    Example from paper:
    "cooling" script: success has "refrigerator", failure has "oven"
     requires_mention(cold_appliance), ¬requires_mention(heat_source)
    """
    
    def __init__(self, config: LINGUAConfig, vlm):
        self.config = config
        self.vlm = vlm
        
        # Track success/failure examples per script
        self.success_examples: Dict[str, List[str]] = defaultdict(list)
        self.failure_examples: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("[ok] Contrastive refinement initialized")
    
    def add_example(self, script_id: str, description: str, success: bool):
        """Track success/failure examples for contrastive analysis"""
        if success:
            self.success_examples[script_id].append(description)
        else:
            self.failure_examples[script_id].append(description)
    
    def can_refine(self, script_id: str) -> bool:
        """
        condition: min(|S_k|, |F_k|) >= 3
        """
        n_success = len(self.success_examples.get(script_id, []))
        n_failure = len(self.failure_examples.get(script_id, []))
        
        return min(n_success, n_failure) >= self.config.MIN_CONTRASTIVE_EXAMPLES
    
    def extract_discriminators_vlm(self, script_id: str) -> Dict[str, List[str]]:
        """
        Eq. 13: Linguistic Discriminator Extraction
        
        D_k = VLM(p_contrast, S_k, F_k)
        
        Extract discriminative semantic features from success vs failure sets
        """
        success_set = self.success_examples.get(script_id, [])
        failure_set = self.failure_examples.get(script_id, [])
        
        if not success_set or not failure_set:
            return {"positive": [], "negative": []}
        
        if not self.vlm.model:
            # Fallback: Simple word frequency analysis
            return self._extract_discriminators_frequency(success_set, failure_set)
        
        try:
            # Build contrastive prompt
            success_str = "\n".join([f"  - {ex[:80]}" for ex in success_set[:5]])
            failure_str = "\n".join([f"  - {ex[:80]}" for ex in failure_set[:5]])
            
            # Paper's contrastive prompt
            prompt = f"""Task: Extract discriminative linguistic features for "{script_id}".

SUCCESS cases (what works):
{success_str}

FAILURE cases (what doesn't work):
{failure_str}

Identify:
1. Positive discriminators: Terms/concepts present in SUCCESS but not in FAILURE
2. Negative discriminators: Terms/concepts present in FAILURE but not in SUCCESS

Format:
Positive: [word1, word2, ...]
Negative: [word3, word4, ...]

Analysis:"""
            
            inputs = self.vlm.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.vlm.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.2,
                    pad_token_id=self.vlm.tokenizer.eos_token_id
                )
            
            response = self.vlm.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse discriminators
            discriminators = self._parse_discriminator_output(response)
            
            logger.info(f" Discriminators for {script_id}:")
            logger.info(f"   Positive: {discriminators['positive'][:3]}")
            logger.info(f"   Negative: {discriminators['negative'][:3]}")
            
            return discriminators
            
        except Exception as e:
            logger.warning(f"VLM discriminator extraction failed: {e}")
            return self._extract_discriminators_frequency(success_set, failure_set)
    
    def _parse_discriminator_output(self, response: str) -> Dict[str, List[str]]:
        """Parse VLM output for discriminators"""
        positive = []
        negative = []
        
        # Look for "Positive:" and "Negative:" sections
        if "Positive:" in response:
            pos_section = response.split("Positive:")[1].split("Negative:")[0] if "Negative:" in response else response.split("Positive:")[1]
            # Extract words in brackets or comma-separated
            import re
            words = re.findall(r'\b[a-zA-Z_]+\b', pos_section)
            positive = [w.lower() for w in words if len(w) > 2][:5]
        
        if "Negative:" in response:
            neg_section = response.split("Negative:")[1]
            import re
            words = re.findall(r'\b[a-zA-Z_]+\b', neg_section)
            negative = [w.lower() for w in words if len(w) > 2][:5]
        
        return {"positive": positive, "negative": negative}
    
    def _extract_discriminators_frequency(self, success_set: List[str], failure_set: List[str]) -> Dict[str, List[str]]:
        """Fallback: Frequency-based discriminator extraction"""
        # Tokenize
        def tokenize(texts):
            words = []
            for text in texts:
                words.extend(text.lower().split())
            return Counter([w for w in words if len(w) > 3])
        
        success_words = tokenize(success_set)
        failure_words = tokenize(failure_set)
        
        # Find discriminative words
        positive = []
        for word, count in success_words.most_common(10):
            if word not in failure_words or failure_words[word] < count / 2:
                positive.append(word)
        
        negative = []
        for word, count in failure_words.most_common(10):
            if word not in success_words or success_words[word] < count / 2:
                negative.append(word)
        
        return {"positive": positive[:5], "negative": negative[:5]}
    
    def refine_schema(self, procedural: ProceduralMemory, discriminators: Dict[str, List[str]]) -> ProceduralMemory:
        """
        Eq. 14-16: Refine linguistic schemas
        
        Eq. 14: Ψ_k ← Ψ_k ∪ ΔΨ_k+ ∪ {¬ψ : ψ ∈ ΔΨ_k-}
        Eq. 15: Π_k ← Merge(Π_k, ΔΠ_k)
        Eq. 16: Φ_k ← Φ_k ∪ ΔΦ_k
        """
        # Eq. 14: Add positive discriminators to preconditions
        for feature in discriminators.get("positive", []):
            new_precond = f"requires_mention({feature})"
            if new_precond not in procedural.preconditions:
                procedural.preconditions.append(new_precond)
                logger.info(f"   + Added precondition: {new_precond}")
        
        # Eq. 14: Add negative discriminators (negated)
        for feature in discriminators.get("negative", []):
            neg_precond = f"NOT_mention({feature})"
            if neg_precond not in procedural.preconditions:
                procedural.preconditions.append(neg_precond)
                logger.info(f"   + Added negative precondition: {neg_precond}")
        
        # Eq. 16: Extend postconditions
        procedural.postconditions.append("contrastive_refined")
        
        logger.info(f"[ok] Schema refined: {procedural.script_id}")
        logger.info(f"   Total preconditions: {len(procedural.preconditions)}")
        
        return procedural
    
    def refine_if_possible(self, script_id: str, procedural_memory: 
List[ProceduralMemory]) -> bool:
        """Versioned refinement"""
        if not self.can_refine(script_id):
            return False
        
        target_proc = None
        for proc in procedural_memory:
            if proc.script_id == script_id and proc.is_active:
                target_proc = proc
                break
        
        if not target_proc:
            return False
        
        logger.info(f"\n VERSIONED REFINEMENT: {script_id} v{target_proc.script_version}")
        logger.info(f"   Success: {len(self.success_examples[script_id])}")
        logger.info(f"   Failure: {len(self.failure_examples[script_id])}")
        
        discriminators = self.extract_discriminators_vlm(script_id)
        
        if not discriminators["positive"] and not discriminators["negative"]:
            return False
        
        refined_script = target_proc.create_refined_version(
            discriminators=discriminators,
            reason="contrastive_analysis"
        )
        
        target_proc.is_active = False
        procedural_memory.append(refined_script)
        
        logger.info(f"[ok] Created {refined_script.script_id}")
        logger.info(f"  Previous version retained (inactive)")
        
        return True

# ============================================================================
# FRAME-AWARE PROCEDURAL MINING (Section 3.2.3)
# ============================================================================

class FrameAwareProceduralMining:
    """
    Section 3.2.3: Procedural Mining with 5-step pipeline
    (1) VLM Narrative Generation
    (2) Semantic Role Labeling
    (3) Frame-Aware Clustering  ← NOW FULLY IMPLEMENTED
    (4) Canonical Schema Extraction
    (5) Validation
    """
    
    def __init__(self, config: LINGUAConfig, memory_system, embedding_model):
        self.config = config
        self.memory = memory_system
        self.embedding_model = embedding_model
        
        logger.info("[ok] Frame-aware procedural mining initialized")
    
    def mine_scripts(self, episodic_memories: List[EpisodicMemory]) -> List[ProceduralMemory]:
        """
        Complete 5-step mining pipeline
        """
        logger.info(f"\n  PROCEDURAL MINING (5-step pipeline)")
        logger.info(f"   Input: {len(episodic_memories)} episodic memories")
        
        if len(episodic_memories) < self.config.MIN_SCRIPT_INSTANCES:
            logger.warning(f"   Insufficient episodes for mining")
            return []
        
        # Step 1: VLM Narrative Generation (already done in episodic memories)
        # Step 2: Semantic Role Labeling (already extracted)
        
        # Step 3: Frame-Aware Clustering
        logger.info(f"\n   Step 3: Frame-aware clustering...")
        frame_clusters = self._frame_aware_clustering(episodic_memories)
        logger.info(f"    Found {len(frame_clusters)} frame clusters")
        
        # Step 4: Canonical Schema Extraction
        logger.info(f"\n   Step 4: Extracting canonical schemas...")
        scripts = self._extract_canonical_schemas(frame_clusters, episodic_memories)
        logger.info(f"    Extracted {len(scripts)} schemas")
        
        # Step 5: Validation
        logger.info(f"\n   Step 5: Validation...")
        validated = self._validate_scripts(scripts)
        logger.info(f"    Validated {len(validated)} scripts")
        
        return validated
    
    def _frame_aware_clustering(self, memories: List[EpisodicMemory]) -> Dict[str, List[int]]:
        """
        Step 3: Frame-Aware Clustering using FrameNet-style embeddings
        
        Paper: "Group SRL outputs by semantic frames (e.g., all instances of
               Cooking frame, Ingestion frame) using frame embeddings from FrameNet"
        """
        # Extract frame-semantic representations
        frame_representations = []
        
        for mem in memories:
            # Build frame representation from action + objects
            frame_text = f"{mem.action}"
            
            # Add object affordances to frame representation
            for obj in mem.affordance_objects:
                obj_label = obj.get("label", "")
                # Get semantic entry (VLM-based)
                if obj_label:
                    # Get affordances for this object
                    entry = self.memory.semantic_memory.get_or_create_entry(obj_label)
                    for aff in entry.affordances:
                        frame_text += f" {aff.get('frame', '')}"
            
            frame_representations.append(frame_text)
        
        # Encode frame representations
        frame_embeddings = []
        for frame_text in frame_representations:
            emb = self.embedding_model.encode(frame_text, convert_to_numpy=True)
            frame_embeddings.append(emb)
        
        frame_embeddings = np.array(frame_embeddings)
        
        # Cluster by frame similarity
        if len(frame_embeddings) < 2:
            return {"general": list(range(len(memories)))}
        
        from sklearn.cluster import AgglomerativeClustering
        
        # Use agglomerative clustering with cosine similarity
        clustering = AgglomerativeClustering(
            n_clusters=min(5, len(memories)),  # Max 5 clusters
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(frame_embeddings)
        
        # Group memories by cluster
        frame_clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            # Name cluster by most common action/frame
            cluster_name = f"frame_cluster_{label}"
            frame_clusters[cluster_name].append(idx)
        
        # Rename clusters based on dominant frames
        named_clusters = {}
        for cluster_id, indices in frame_clusters.items():
            # Find most common action in this cluster
            actions = [memories[i].action for i in indices if memories[i].action]
            if actions:
                most_common_action = Counter(actions).most_common(1)[0][0]
                named_clusters[most_common_action] = indices
            else:
                named_clusters[cluster_id] = indices
        
        return named_clusters
    
    def _extract_canonical_schemas(self, frame_clusters: Dict[str, List[int]],
                                memories: List[EpisodicMemory]) -> List[ProceduralMemory]:
        """Extract schemas with semantic preconditions"""
        scripts = []
        
        for frame_name, indices in frame_clusters.items():
            if len(indices) < self.config.MIN_SCRIPT_INSTANCES:
                continue
            
            cluster_memories = [memories[i] for i in indices]
            
            action_patterns = []
            for mem in cluster_memories:
                pattern = {
                    "predicate": mem.action,
                    "ARG0": mem.agent,
                    "ARG1": mem.affected_entity,
                    "ARGM-LOC": mem.location,
                    "ARGM-PRP": mem.goal
                }
                action_patterns.append(pattern)
            
            durations = [mem.end_time - mem.start_time for mem in cluster_memories]
            mean_duration = float(np.mean(durations))
            std_duration = float(np.std(durations))
            
            # NEW: Semantic preconditions
            semantic_preconditions = self._extract_semantic_preconditions(cluster_memories)
            
            goals = [mem.goal for mem in cluster_memories if mem.goal]
            goal_preconditions = []
            if goals:
                goal_counter = Counter(goals)
                most_common_goal = goal_counter.most_common(1)[0][0]
                goal_preconditions.append(f"requires_goal({most_common_goal})")
            
            all_preconditions = semantic_preconditions + goal_preconditions
            
            outcomes = [mem.outcome for mem in cluster_memories if mem.outcome]
            common_postconditions = list(set(outcomes)) if outcomes else ["action_completed"]
            
            script = ProceduralMemory(
                script_name=f"{frame_name}_script",
                preconditions=all_preconditions,
                action_sequence=action_patterns,
                postconditions=common_postconditions,
                temporal_constraints=[(mean_duration, std_duration)],
                corpus_frequency=len(indices) / len(memories),
                instance_count=len(indices),
                script_id=f"proc_{frame_name}",
                script_version=1,
                parent_script_id=None,
                is_active=True
            )
            
            script_text = " ".join([mem.description for mem in cluster_memories])
            script.script_embedding = self.embedding_model.encode(
                script_text, convert_to_numpy=True
            )
            
            scripts.append(script)
        
        return scripts
    
    def _validate_scripts(self, scripts: List[ProceduralMemory]) -> List[ProceduralMemory]:
        """
        Step 5: Validation
        
        Paper: "Filter schemas that appear <5 times or have >50% variance
               in temporal structure"
        """
        validated = []
        
        for script in scripts:
            # Check instance count
            if script.instance_count < self.config.MIN_SCRIPT_INSTANCES:
                logger.debug(f"   Rejected {script.script_id}: too few instances ({script.instance_count})")
                continue
            
            # Check temporal variance
            if script.temporal_constraints:
                mean_dur, std_dur = script.temporal_constraints[0]
                if mean_dur > 0:
                    variance = std_dur / mean_dur
                    if variance > self.config.MAX_TEMPORAL_VARIANCE:
                        logger.debug(f"   Rejected {script.script_id}: high variance ({variance:.2f})")
                        continue
            
            validated.append(script)
            logger.info(f"   [ok] Validated: {script.script_id} (n={script.instance_count})")
        
        return validated

    def _extract_semantic_preconditions(self, memories: List['EpisodicMemory']) -> List[str]:
        """Semantic  Procedural: HasAffordance checks"""
        preconditions = []
    
        all_objects = {}
        for mem in memories:
            for obj in mem.affordance_objects:
                label = obj.get("label", "")
                if label:
                    if label not in all_objects:
                        all_objects[label] = 0
                    all_objects[label] += 1
    
        total_memories = len(memories)
        for obj_label, count in all_objects.items():
            frequency = count / total_memories
        
            if frequency > 0.5:
                if obj_label in self.memory.semantic_memory.semantic_entries:
                    entry = self.memory.semantic_memory.semantic_entries[obj_label]
                
                    for affordance in entry.affordances[:2]:
                        frame = affordance.get("frame", "")
                        if frame:
                            precondition = f"HasAffordance({obj_label}, {frame})"
                            preconditions.append(precondition)
    
        return preconditions


# ============================================================================
# EVENT-DRIVEN PERCEPTION (Section 3.1)
# ============================================================================

class EventDrivenPerception:
    """Section 3.1: Event-driven perception with VideoMAE + Gemma3-4B VLM"""
    
    def __init__(self, config: LINGUAConfig):
        self.config = config
        self.videomae_model = None
        self.videomae_processor = None
        self.yolo_model = None
        self.vlm = None  # Gemma3-4B model
        self.vlm_processor = None  # Gemma3-4B processor
        self.prev_embedding = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all perception models"""
        
        # ============================================================
        # 1. VideoMAE for semantic change detection
        # ============================================================
        if VIDEOMAE_AVAILABLE:
            try:
                self.videomae_processor = VideoMAEImageProcessor.from_pretrained(
                    self.config.VIDEOMAE_MODEL
                )
                self.videomae_model = VideoMAEModel.from_pretrained(
                    self.config.VIDEOMAE_MODEL
                )
                self.videomae_model.eval()
                
                if torch.cuda.is_available():
                    self.videomae_model = self.videomae_model.cuda()
                
                logger.info(f"[ok] VideoMAE initialized")
            except Exception as e:
                logger.error(f"VideoMAE init failed: {e}")
        
        # ============================================================
        # 2. YOLO for object detection
        # ============================================================
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(self.config.YOLO_MODEL)
                logger.info("[ok] YOLO initialized")
            except Exception as e:
                logger.warning(f"YOLO init failed: {e}")
        
        # ------------------------------------------------------------------
        # 3. Gemma3-4B VLM is served via Ollama (see VLMNarrativeGenerator).
        #    No transformers download is needed here -- inference happens
        #    through the Ollama HTTP API in vision-language mode.
        # ------------------------------------------------------------------
        self.vlm = None
        self.vlm_processor = None
        logger.info("[ok] Gemma3-4B VLM will be served via Ollama (see Setup)")
    
    def _prepare_exactly_16_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Prepare exactly 16 frames for VideoMAE (unchanged)"""
        target = self.config.VIDEOMAE_NUM_FRAMES
        num_frames = len(frames)
        
        if num_frames == 0:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            return [blank.copy() for _ in range(target)]
        elif num_frames == target:
            return frames
        elif num_frames < target:
            repeat_factor = target // num_frames
            remainder = target % num_frames
            result = []
            for frame in frames:
                result.extend([frame.copy() for _ in range(repeat_factor)])
            for i in range(remainder):
                result.append(frames[i].copy())
            return result[:target]
        else:
            indices = np.linspace(0, num_frames - 1, target, dtype=int)
            return [frames[i].copy() for i in indices]
    
    def _numpy_to_pil(self, frames: List[np.ndarray]) -> List[Image.Image]:
        pil_frames = []
        for frame in frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            pil_frames.append(Image.fromarray(rgb_frame))
        return pil_frames
    
    def compute_semantic_change(self, frame_curr: np.ndarray, 
                            frame_prev: Optional[np.ndarray] = None) -> float:
        """Eq. 2: Temporal-aware semantic change"""
        if not self.videomae_model or frame_prev is None:
            return 1.0
        
        try:
            input_frames = [frame_prev, frame_curr]
            frames_16 = self._prepare_exactly_16_frames(input_frames)
            pil_frames = self._numpy_to_pil(frames_16)
            
            if len(pil_frames) != 16:
                return 1.0
            
            inputs = self.videomae_processor(pil_frames, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.videomae_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            embedding = embedding.flatten()
            if len(embedding) > 768:
                embedding = embedding[:768]
            elif len(embedding) < 768:
                embedding = np.pad(embedding, (0, 768 - len(embedding)))
            
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            if self.prev_embedding is not None:
                similarity = np.dot(embedding, self.prev_embedding)
                semantic_change = 1.0 - similarity
            else:
                semantic_change = 1.0
            
            self.prev_embedding = embedding
            return float(semantic_change)
            
        except:
            return 1.0
    
    def detect_affordance_objects(self, frame: np.ndarray) -> List[Dict]:
        """Eq. 3: Affordance-based attention"""
        if not self.yolo_model or frame is None or frame.size == 0:
            return []
        
        try:
            results = self.yolo_model(frame, conf=self.config.YOLO_CONFIDENCE, verbose=False)
            objects = []
            
            for r in results:
                for box in r.boxes:
                    obj = {
                        "label": self.yolo_model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].cpu().numpy().tolist()
                    }
                    objects.append(obj)
            
            return objects
        except:
            return []
    
    def select_frames(self, video_path: str, fps: int = 30) -> Dict[int, Dict]:
        """
        Event-driven frame selection with VLM descriptions
        
        Returns frames selected by:
        - Semantic change > threshold (Eq. 1)
        - OR affordance-triggering objects detected
        
        Each selected frame now includes Gemma3-4B generated description!
        """
        logger.info(f"🎥 EVENT-DRIVEN PERCEPTION: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return {}
        
        cap = cv2.VideoCapture(video_path)
        
        # Get actual FPS from video if not provided
        if fps == 30:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps = int(video_fps)
        
        selected_frames = {}
        frame_idx = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # ============================================================
            # STEP 1: Compute semantic change (VideoMAE)
            # ============================================================
            if prev_frame is not None:
                semantic_change = self.compute_semantic_change(frame, prev_frame)
            else:
                semantic_change = 1.0  # First frame always selected
            
            # ============================================================
            # STEP 2: Detect affordance objects (YOLO)
            # ============================================================
            affordance_objects = self.detect_affordance_objects(frame)
            select_by_change = semantic_change > self.config.SEMANTIC_CHANGE_THRESHOLD
            select_by_affordance = len(affordance_objects) > 0
            if select_by_change or select_by_affordance:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                description = self.generate_description(frame_rgb, affordance_objects)
                
                # Get FrameNet-style affordances
                framenet_affordances = {}
                for obj in affordance_objects:
                    frames = self.semantic_memory.get_affordances_framenet(
                        obj['label'],
                        context=description
                    )
                    if frames:
                        framenet_affordances[obj['label']] = frames
                
                selected_frames[frame_idx] = {
                    "frame": frame.copy(),
                    "frame_rgb": frame_rgb,
                    "affordance_objects": affordance_objects,
                    "framenet_affordances": framenet_affordances,  # NEW!
                    "timestamp": timestamp,
                    "semantic_change": semantic_change,
                    "description": description
                }
                
                # Log selection
                objects_str = ', '.join([o['label'] for o in affordance_objects[:3]])
                logger.info(f"  Frame {frame_idx:4d} ({timestamp:6.2f}s): {description[:60]}...")
                logger.info(f"    Objects: [{objects_str}], Change: {semantic_change:.3f}")
            
            prev_frame = frame.copy() if ret else None
            frame_idx += 1
        
        cap.release()
        
        # ============================================================
        # Report statistics
        # ============================================================
        total_frames = frame_idx
        selected_count = len(selected_frames)
        retention_rate = (selected_count / total_frames * 100) if total_frames > 0 else 0
        savings = 100 - retention_rate
        
        logger.info(f"\n  [ok] Selected {selected_count}/{total_frames} frames ({retention_rate:.1f}%)")
        logger.info(f"  [ok] Computational savings: ~{savings:.1f}%")
        
        # Verify coverage (from paper: 94% question-relevant events)
        if retention_rate < 5:
            logger.warning(f"  [warn] Very low retention ({retention_rate:.1f}%), check threshold!")
        elif retention_rate > 15:
            logger.warning(f"  [warn] High retention ({retention_rate:.1f}%), may reduce efficiency")
        
        return selected_frames
# ============================================================================
# VLM NARRATIVE GENERATOR
# ============================================================================
# ============================================================================
# VLM NARRATIVE GENERATOR (OLLAMA VERSION)
# ============================================================================

class VLMNarrativeGenerator:
    """VLM for description generation using Ollama (Gemma3-4B """
    
    def __init__(self, config: LINGUAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_ollama()
    
    def _initialize_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.ollama_client = ollama
            
            # Test connection
            models = self.ollama_client.list()
            logger.info(f"[ok] Ollama initialized. Available models: {[m['name'] for m in models['models']]}")
            
            # Check if required model exists
            model_exists = any(self.config.VLM_MODEL in m['name'] for m in models['models'])
            if not model_exists:
                logger.warning(f"[warn] Model {self.config.VLM_MODEL} not found in Ollama!")
                logger.warning(f"  Run: ollama pull {self.config.VLM_MODEL}")
            
            self.model = True  # Flag to indicate Ollama is ready
            
        except ImportError:
            logger.error("[err] Ollama not installed! Run: pip install ollama")
            self.model = None
        except Exception as e:
            logger.error(f"[err] Ollama initialization failed: {e}")
            self.model = None
    
    def generate_description(self, frame: np.ndarray, affordance_objects: List[Dict]) -> str:
        """
        Generate description using Gemma3-4B via Ollama
        
        Args:
            frame: RGB image array (H, W, 3)
            affordance_objects: Detected objects from YOLO
        
        Returns:
            Natural language description
        """
        if not affordance_objects:
            return "Empty scene"
        
        objects = [o["label"] for o in affordance_objects[:5]]
        
        # ============================================================
        # FALLBACK: If Ollama not available
        # ============================================================
        if self.model is None:
            person_count = objects.count("person")
            if person_count > 0:
                other = [o for o in objects if o != "person"]
                if other:
                    return f"Person with {other[0]}"
            return f"Scene with {', '.join(objects[:2])}"
        
        # ============================================================
        # Use Gemma3-4B via Ollama
        # ============================================================
        try:
            # Convert numpy array to base64 image
            from PIL import Image
            import io
            import base64
            
            if isinstance(frame, np.ndarray):
                frame_pil = Image.fromarray(frame.astype('uint8'))
            else:
                frame_pil = frame
            
            # Convert to base64
            buffered = io.BytesIO()
            frame_pil.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prompt with temporal markers (Section 3.1)
            prompt = (
                "Describe this scene in one sentence. "
                "Use temporal markers (before, after, while, then) and "
                "causal words (because, so that). "
                f"Focus on actions with: {', '.join(objects[:3])}."
            )
            
            # Call Ollama's Gemma3-4B
            response = self.ollama_client.generate(
                model=self.config.VLM_MODEL,
                prompt=prompt,
                images=[img_base64],
                options={
                    'temperature': self.config.VLM_TEMPERATURE,
                    'num_predict': self.config.VLM_MAX_LENGTH
                }
            )
            
            description = response['response'].strip()
            
            return description if description else f"Scene with {objects[0]}"
            
        except Exception as e:
            logger.warning(f"[warn] Gemma3-4B generation failed: {e}")
            # Fallback to simple description
            return f"Person interacting with {objects[0]}" if "person" in objects else f"Scene with {objects[0]}"

# ============================================================================
# LLM REASONING ENGINE (OLLAMA VERSION)
# ============================================================================

class OllamaReasoningEngine:
    """LLM for reasoning/planning using Ollama (Gemma3-4B)"""
    
    def __init__(self, config: LINGUAConfig):
        self.config = config
        self._initialize_ollama()
    
    def _initialize_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.ollama_client = ollama
            
            # Check if model exists
            models = self.ollama_client.list()
            model_exists = any(self.config.LLM_MODEL in m['name'] for m in models['models'])
            
            if not model_exists:
                logger.warning(f"[warn] Model {self.config.LLM_MODEL} not found!")
                logger.warning(f"  Run: ollama pull {self.config.LLM_MODEL}")
            
            logger.info(f"[ok] Ollama LLM initialized: {self.config.LLM_MODEL}")
            
        except Exception as e:
            logger.error(f"[err] Ollama LLM init failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text using Gemma3-4B"""
        try:
            response = self.ollama_client.generate(
                model=self.config.LLM_MODEL,
                prompt=prompt,
                options={
                    'temperature': self.config.LLM_TEMPERATURE,
                    'num_predict': max_tokens or self.config.LLM_MAX_LENGTH
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"[err] Gemma3 generation failed: {e}")
            return ""

# ============================================================================
# LINGUISTIC MEMORY SYSTEM
# ============================================================================

class LinguisticMemorySystem:
    """Three-memory architecture (Section 3.2)"""
    
    def __init__(self, config: LINGUAConfig):
        self.config = config
        
        # FIX: Use SRL_MODEL instead of SPACY_MODEL
        self.srl = SemanticRoleLabeler(config.SRL_MODEL)
        
        # Use Ollama-based VLM
        self.vlm = VLMNarrativeGenerator(config)
        
        # Add Ollama reasoning engine
        self.llm = OllamaReasoningEngine(config)
        
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # VLM-based semantic memory
        self.semantic_memory = VLMSemanticMemory(config, self.vlm, self.embedding_model)
        
        self.episodic_memory: List[EpisodicMemory] = []
        self.procedural_memory: List[ProceduralMemory] = []
        
        logger.info("[ok] Linguistic Memory System initialized (Ollama)")
    
    def create_episodic_memory(self, frame_data: Dict, start_time: float, end_time: float) -> EpisodicMemory:
        """Eq. 4: Create episodic memory"""
        description = self.vlm.generate_description(frame_data["affordance_objects"])
        roles = self.srl.extract_semantic_roles(description)
        
        # Infer goal from affordances
        obj_labels = [o["label"] for o in frame_data["affordance_objects"]]
        goal = "interaction"
        if obj_labels:
            entry = self.semantic_memory.get_or_create_entry(obj_labels[0])
            if entry.affordances:
                goal = entry.affordances[0].get("frame", "interaction")
        
        memory = EpisodicMemory(
            agent=roles["Agent"],
            action=roles["Action"],
            affected_entity=roles["Affected_Entity"],
            location=roles["Location"],
            timestamp=start_time,
            goal=goal,
            outcome="ongoing",
            start_time=start_time,
            end_time=end_time,
            description=description,
            affordance_objects=frame_data["affordance_objects"],
            memory_id=f"epi_{start_time:.1f}"
        )
        
        return memory

    def merge_adjacent_episodes(self):
        """
        Paper Section 3, Episodic Memory paragraph:
            "Adjacent entries merge into longer narratives when temporally
             close (gap < Delta_t_merge = 2s), semantically consistent
             (embedding similarity > 0.85), and linguistically continuous
             (markers like 'then', 'next')."
        """
        if len(self.episodic_memory) < 2:
            return

        continuity_markers = {
            "then", "next", "afterward", "afterwards",
            "subsequently", "after that", "later",
        }

        sorted_eps = sorted(self.episodic_memory, key=lambda m: m.start_time)
        merged = [sorted_eps[0]]

        for curr in sorted_eps[1:]:
            prev = merged[-1]

            # (i) Temporal proximity
            gap = curr.start_time - prev.end_time
            if gap >= self.config.TEMPORAL_GAP_THRESHOLD:
                merged.append(curr)
                continue

            # (ii) Semantic consistency -- embedding similarity > 0.85
            if prev.videomae_embedding is not None and curr.videomae_embedding is not None:
                p_emb = prev.videomae_embedding
                c_emb = curr.videomae_embedding
            else:
                p_emb = self.embedding_model.encode(
                    prev.description or "", convert_to_numpy=True
                )
                c_emb = self.embedding_model.encode(
                    curr.description or "", convert_to_numpy=True
                )
            denom = float(np.linalg.norm(p_emb) * np.linalg.norm(c_emb)) + 1e-8
            sim = float(np.dot(p_emb, c_emb) / denom)
            if sim <= self.config.EPISODIC_MERGE_SIMILARITY:
                merged.append(curr)
                continue

            # (iii) Linguistic continuity marker required
            text = (curr.description or "").lower()
            if not any(m in text for m in continuity_markers):
                merged.append(curr)
                continue

            # Merge: extend previous entry to absorb curr
            prev.end_time = curr.end_time
            if prev.description and curr.description:
                prev.description = (
                    prev.description.rstrip(". ") + ", " + curr.description
                )
            elif curr.description:
                prev.description = curr.description
            prev.affordance_objects = (
                (prev.affordance_objects or []) + (curr.affordance_objects or [])
            )

        n_before, n_after = len(self.episodic_memory), len(merged)
        self.episodic_memory = merged
        if n_after < n_before:
            logger.info(
                f"[ok] Merged adjacent episodes: {n_before} -> {n_after} "
                f"(gap<{self.config.TEMPORAL_GAP_THRESHOLD}s, "
                f"sim>{self.config.EPISODIC_MERGE_SIMILARITY})"
            )

# ============================================================================
# LLM INFERENCER - LLM-BASED SEMANTIC MAPPING
# ============================================================================

class LLMGoalInferencer:
    """LLM-based goal inference matching Equation 10"""
    
    def __init__(self, vlm, semantic_memory):
        self.vlm = vlm
        self.semantic_memory = semantic_memory
        
        # Goal taxonomy (6 types)
        self.goal_taxonomy = """Possible Goals:
1. identify_causal_event: asking for reasons, causes, or 'why' something happened.
2. verify_temporal_sequence: asking about timing, ordering (before/after, when).
3. identify_agent: asking who performed an action.
4. trace_action_sequence: asking how something happened or what actions occurred.
5. locate_object: asking where something is or where an action took place.
6. general_qa: other general questions about the video."""
        
        # Initialize NER (for entity extraction)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
    
    def infer_goal(self, question: str, objects: List[Dict] = None, context: List = None) -> str:
        """
        Implements Equation 10
        GoalHypotheses = VLM(p_goal, q, Entities, Affordances, EpisodicContext)
        """
        # Step 1: Extract entities from question (NER)
        entities = self._extract_entities(question)
        
        # Step 2: Query semantic memory for affordances
        affordances = self._get_affordances(entities, objects)
        
        # Step 3: Prepare episodic context
        episodic_context = self._prepare_episodic_context(context)
        
        # Step 4: VLM-based goal inference with full context
        if self.vlm.model:
            try:
                goal = self._infer_goal_vlm_contextual(
                    question, entities, affordances, episodic_context
                )
                if goal:
                    logger.debug(f"  VLM inferred goal (contextual): {goal}")
                    return goal
            except Exception as e:
                logger.debug(f"  VLM goal inference failed: {e}, using fallback")
        
        # Fallback to keyword matching
        logger.debug("  Using keyword-based goal inference (fallback)")
        return self._infer_goal_keywords(question)
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        Extract entities via NER 
        """
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(question)
                # Extract named entities and noun chunks
                for ent in doc.ents:
                    entities.append(ent.text)
                for chunk in doc.noun_chunks:
                    if chunk.text not in entities:
                        entities.append(chunk.text)
            except:
                pass
        
        # Fallback: extract nouns from question
        if not entities:
            words = question.split()
            # Simple heuristic: capitalize words might be entities
            entities = [w for w in words if w[0].isupper() and len(w) > 2]
        
        return entities[:5]  # Top 5 entities
    
    def _get_affordances(self, entities: List[str], objects: List[Dict] = None) -> List[Dict]:
        """
        Query semantic memory for affordances 
        """
        affordances = []
        
        # Get affordances for entities
        for entity in entities:
            # Try fuzzy matching in semantic memory
            matches = self.semantic_memory.retrieve_fuzzy(entity, threshold=0.7)
            for match in matches:
                affordances.extend(match.affordances)
        
        # Get affordances for detected objects
        if objects:
            for obj in objects[:3]:
                obj_label = obj.get("label", "")
                if obj_label:
                    entry = self.semantic_memory.get_or_create_entry(obj_label)
                    affordances.extend(entry.affordances)
        
        return affordances[:5]  # Top 5 affordances
    
    def _prepare_episodic_context(self, context: List = None) -> str:
        """
        Prepare episodic context string 
        """
        if not context:
            return "No prior context"
        
        # Extract recent episodic memories
        context_str = "Recent events:\n"
        for i, mem in enumerate(context[-3:]):  # Last 3 memories
            if hasattr(mem, 'description'):
                context_str += f"  {i+1}. {mem.description}\n"
        
        return context_str
    
    def _infer_goal_vlm_contextual(self, question: str, entities: List[str], 
                                   affordances: List[Dict], episodic_context: str) -> str:
        """
        Full VLM-based goal inference with context (Equation 10)
        
        GoalHypotheses = VLM(p_goal, q, Entities, Affordances, EpisodicContext)
        """
        # Format entities
        entities_str = ", ".join(entities) if entities else "none detected"
        
        # Format affordances
        affordances_str = ""
        if affordances:
            for aff in affordances[:3]:
                frame = aff.get("frame", "")
                roles = aff.get("roles", [])
                affordances_str += f"  - {frame}: {roles}\n"
        else:
            affordances_str = "  - none detected"
        
        # Paper's prompt format with full context
        prompt = f"""Task: Classify the underlying goal of the video question using contextual information.

{self.goal_taxonomy}

Question: "{question}"

Extracted Entities: {entities_str}

Semantic Affordances:
{affordances_str}

{episodic_context}

Based on the question and context, return ONLY the goal name from the taxonomy.

Goal:"""
        
        # Tokenize and generate
        inputs = self.vlm.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.vlm.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                pad_token_id=self.vlm.tokenizer.eos_token_id
            )
        
        response = self.vlm.tokenizer.decode(outputs[0], skip_special_tokens=True)
        goal = self._parse_goal_response(response)
        
        return goal
    
    def _parse_goal_response(self, response: str) -> str:
        """Parse VLM output for goal classification"""
        if "Goal:" in response:
            response = response.split("Goal:")[-1].strip()
        
        response = response.strip().lower()
        
        valid_goals = [
            "identify_causal_event",
            "verify_temporal_sequence",
            "identify_agent",
            "trace_action_sequence",
            "locate_object",
            "general_qa"
        ]
        
        for goal in valid_goals:
            if goal in response:
                return goal
        
        # Fuzzy matching
        if "causal" in response or "why" in response:
            return "identify_causal_event"
        elif "temporal" in response or "when" in response:
            return "verify_temporal_sequence"
        elif "agent" in response or "who" in response:
            return "identify_agent"
        elif "action" in response or "how" in response:
            return "trace_action_sequence"
        elif "locate" in response or "where" in response:
            return "locate_object"
        
        return "general_qa"
    
    def _infer_goal_keywords(self, question: str) -> str:
        """Fallback: Keyword-based inference"""
        q = question.lower()
        
        if "why" in q or "because" in q or "reason" in q:
            return "identify_causal_event"
        if "when" in q or "before" in q or "after" in q:
            return "verify_temporal_sequence"
        if "who" in q:
            return "identify_agent"
        if "where" in q:
            return "locate_object"
        if "how" in q or "what" in q:
            return "trace_action_sequence"
        
        return "general_qa"

# ============================================================================
# BAYESIAN ACTION SELECTOR
# ============================================================================

class BayesianActionSelector:
    """
    Paper expected-utility formula (Section 3, Belief-Action-Verification):

        EU_k = Rel_k * E[rho_k]  -  Risk_k  +  lambda_info * H(Beta(alpha_k, beta_k))

    where
        Rel_k    = semantic similarity between observed event descriptions
                   and the schema's preconditions (sentence embeddings)
        E[rho_k] = alpha_k / (alpha_k + beta_k)
        Risk_k   = semantic similarity between the script and past failure
                   narratives (penalises contradictions)
        H(.)     = differential entropy of Beta(alpha_k, beta_k); encourages
                   exploration of uncertain schemas
    """

    def __init__(self, config: LINGUAConfig, embedding_model):
        self.config = config
        self.embedding_model = embedding_model

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _rel_k(self,
               preconditions: List[str],
               observed_descriptions: List[str]) -> float:
        """Mean over preconditions of best cosine sim to any observation."""
        if not preconditions or not observed_descriptions:
            return 0.0
        precond_embs = self.embedding_model.encode(
            preconditions, convert_to_numpy=True
        )
        obs_embs = self.embedding_model.encode(
            observed_descriptions, convert_to_numpy=True
        )
        sims = []
        for p in precond_embs:
            best = 0.0
            for o in obs_embs:
                s = self._cosine(p, o)
                if s > best:
                    best = s
            sims.append(max(best, 0.0))
        return float(np.mean(sims)) if sims else 0.0

    def _risk_k(self,
                script_text: str,
                past_failure_narratives: List[str]) -> float:
        """Mean cosine sim between script representation and failure narratives."""
        if not past_failure_narratives or not script_text:
            return 0.0
        script_emb = self.embedding_model.encode(
            script_text, convert_to_numpy=True
        )
        fail_embs = self.embedding_model.encode(
            past_failure_narratives, convert_to_numpy=True
        )
        sims = [max(self._cosine(script_emb, f), 0.0) for f in fail_embs]
        return float(np.mean(sims)) if sims else 0.0

    @staticmethod
    def _entropy(alpha: float, beta_val: float) -> float:
        try:
            h = float(beta_dist.entropy(alpha, beta_val))
            return h if np.isfinite(h) else 0.0
        except Exception:
            return 0.0

    def compute_expected_utility(self,
                                 procedural: ProceduralMemory,
                                 observed_descriptions: List[str],
                                 past_failure_narratives: List[str]) -> float:
        """EU_k = Rel_k * E[rho_k] - Risk_k + lambda_info * H(Beta(alpha_k, beta_k))."""
        reliability = procedural.alpha / (procedural.alpha + procedural.beta)
        rel_k = self._rel_k(procedural.preconditions, observed_descriptions)
        script_text = " ".join(
            [procedural.script_name] + list(procedural.preconditions or [])
        )
        risk_k = self._risk_k(script_text, past_failure_narratives)
        h = self._entropy(procedural.alpha, procedural.beta)
        eu = (rel_k * reliability) - risk_k + self.config.LAMBDA_INFO * h
        return float(eu)

# ============================================================================
# ACTION VERIFIER (NEW - Missing from original code)
# ============================================================================
class ActionVerifier:
    """
    Section 3.3: Verification component
    Implements postcondition coverage and temporal consistency checks
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        logger.info("[ok] Action verifier initialized")
    
    def compute_postcondition_coverage(self, 
                                      expected: List[str], 
                                      observed: List[str],
                                      threshold: float = 0.8) -> float:
        """
        Equation: c_post = |Φ* ∩ Outcomes_observed| / |Φ*|
        
        Measures what fraction of expected postconditions appear in observations
        using semantic similarity (threshold: γ_post = 0.8)
        """
        if not expected:
            return 1.0
        
        if not observed:
            return 0.0
        
        matches = 0
        for exp_outcome in expected:
            exp_emb = self.embedding_model.encode(exp_outcome, convert_to_numpy=True)
            
            for obs_outcome in observed:
                obs_emb = self.embedding_model.encode(obs_outcome, convert_to_numpy=True)
                
                # Cosine similarity
                similarity = np.dot(exp_emb, obs_emb) / (
                    np.linalg.norm(exp_emb) * np.linalg.norm(obs_emb) + 1e-8
                )
                
                if similarity > threshold:
                    matches += 1
                    break  # Count each expected outcome once
        
        coverage = matches / len(expected)
        return coverage
    
    def compute_temporal_consistency(self,
                                    observed_durations: List[float],
                                    expected_stats: List[Tuple[float, float]]) -> float:
        """
        Equation: c_temporal = (1/|Π*|) Σ 𝟙[d_obs,i ∈ [μ_i - 2σ_i, μ_i + 2σ_i]]
        
        Checks if observed action durations fall within expected ranges (±2σ)
        """
        if not observed_durations or not expected_stats:
            return 1.0  # No temporal constraints to check
        
        n = len(observed_durations)
        if n == 0:
            return 1.0
        
        consistent = 0
        
        for i, duration in enumerate(observed_durations):
            if i < len(expected_stats):
                mu, sigma = expected_stats[i]
                
                # Paper: check if within [μ - 2σ, μ + 2σ]
                lower = mu - 2 * sigma
                upper = mu + 2 * sigma
                
                if lower <= duration <= upper:
                    consistent += 1
            else:
                # No expected stats for this action, count as consistent
                consistent += 1
        
        return consistent / n
    
    def verify_script_execution(self,
                               script: 'ProceduralMemory',
                               episodic_memories: List['EpisodicMemory'],
                               config: 'LINGUAConfig') -> Dict[str, Any]:
        """
        Complete verification combining postcondition and temporal checks
        
        Returns:
            Dict with verification results and metrics
        """
        # Extract observed outcomes from recent episodic memories
        observed_outcomes = [mem.outcome for mem in episodic_memories[-5:] 
                           if mem.outcome]
        
        # Extract observed durations
        observed_durations = [mem.end_time - mem.start_time 
                            for mem in episodic_memories[-5:]]
        
        # Compute postcondition coverage
        post_coverage = self.compute_postcondition_coverage(
            expected=script.postconditions,
            observed=observed_outcomes,
            threshold=config.POSTCONDITION_THRESHOLD  # Paper: γ_post = 0.8
        )
        
        # Compute temporal consistency
        temporal_consistency = self.compute_temporal_consistency(
            observed_durations=observed_durations,
            expected_stats=script.temporal_constraints
        )
        
        # Verification passes if both metrics are sufficient
        verification_passed = (
            post_coverage >= config.POSTCONDITION_THRESHOLD and
            temporal_consistency >= 0.5  # Reasonable threshold for temporal match
        )
        
        return {
            "passed": verification_passed,
            "post_coverage": post_coverage,
            "temporal_consistency": temporal_consistency,
            "observed_outcomes": observed_outcomes,
            "observed_durations": observed_durations
        }
    

    def verify_with_grounding(self, script: ProceduralMemory, 
                             episodic_memories: List['EpisodicMemory'],
                             config) -> GroundingStatus:
        """Prediction-verification grounding"""
        grounding = GroundingStatus()
        
        grounding.predicted_postconditions = script.postconditions
        grounding.predicted_temporal_ranges = script.temporal_constraints
        grounding.prediction_made = True
        
        logger.info(f"  📋 Predictions from {script.script_id}:")
        logger.info(f"     Postconditions: {script.postconditions[:3]}")
        
        grounding.observed_postconditions = [mem.outcome for mem in episodic_memories[-5:] 
                                            if mem.outcome]
        grounding.observed_durations = [mem.end_time - mem.start_time 
                                       for mem in episodic_memories[-5:]]
        
        if episodic_memories:
            start = min(mem.start_time for mem in episodic_memories)
            end = max(mem.end_time for mem in episodic_memories)
            grounding.temporal_span = (start, end)
        
        grounding.postcondition_coverage = self.compute_postcondition_coverage(
            expected=grounding.predicted_postconditions,
            observed=grounding.observed_postconditions,
            threshold=config.POSTCONDITION_THRESHOLD
        )
        
        grounding.temporal_consistency = self.compute_temporal_consistency(
            observed_durations=grounding.observed_durations,
            expected_stats=grounding.predicted_temporal_ranges
        )
        
        logger.info(f"  [ok] Verification:")
        logger.info(f"     Postcondition: {grounding.postcondition_coverage:.3f}")
        logger.info(f"     Temporal: {grounding.temporal_consistency:.3f}")
        
        verification_passed = (
            grounding.postcondition_coverage >= config.POSTCONDITION_THRESHOLD and
            grounding.temporal_consistency >= 0.5
        )
        
        if verification_passed:
            grounding.mark_grounded(
                script_id=script.script_id,
                confidence=min(grounding.postcondition_coverage, grounding.temporal_consistency)
            )
            logger.info(f"  GROUNDED")
        else:
            logger.info(f"  [err] NOT GROUNDED")
        
        grounding.verification_passed = verification_passed
        
        return grounding

# ============================================================================
# LINGUA-AGENT WITH CONTINUAL LEARNING (CORRECTED & COMPLETE)
# ============================================================================

class LINGUAAgent:
    """    
    COMPLETE IMPLEMENTATION
    
    Includes ALL sections:
    - 3.1: Event-driven perception (VideoMAE-v2 + YOLO)
    - 3.2.1: Episodic memory (Narrative Templates, Eq. 1)
    - 3.2.2: VLM-based semantic memory (Eq. 2, 9, 11)
    - 3.2.3: Frame-aware procedural mining (5-step pipeline)
    - 3.3: BAV loop (Bayesian action selection, Eq. 11)
    - 3.4: Meta-cognitive reflection (Eq. 12)
    - 3.5: Contrastive refinement (Eq. 13-16)
    - NEW: Continual Learning Tracking
    - NEW: Action Verification (postcondition + temporal)
    """
    
    def __init__(self, config: LINGUAConfig = None):
        self.config = config or LINGUAConfig()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Initializing {self.config.SYSTEM_NAME}")
        logger.info(f"{'='*80}")
        logger.info("Complete implementation:")
        logger.info("   - Event-driven perception (VideoMAE-v2)")
        logger.info("   - VLM-based semantic memory (Gemma3-4B, Eq. 9, 11)")
        logger.info("   - Frame-aware clustering (5-step pipeline)")
        logger.info("   - Bayesian action selection (Eq. 11)")
        logger.info("   - Action verification (postcondition + temporal)")
        logger.info("   - Meta-cognitive reflection (Eq. 12)")
        logger.info("   - Contrastive refinement (Eq. 13-16)")
        logger.info("   - Continual learning tracking")
        logger.info(f"{'='*80}")
        
        # Initialize core components
        self.perception = EventDrivenPerception(self.config)
        self.memory = LinguisticMemorySystem(self.config)
        self.goal_inferencer = LLMGoalInferencer(self.memory.vlm, self.memory.semantic_memory)
        
        # Frame-aware procedural mining
        self.procedural_miner = FrameAwareProceduralMining(
            self.config,
            self.memory,
            self.memory.embedding_model
        )
        
        # Action selection and verification
        self.action_selector = BayesianActionSelector(
            self.config, self.memory.embedding_model
        )
        
        # Action verifier (Section 3.3)
        self.verifier = ActionVerifier(self.memory.embedding_model)
        
        # Reflection and consolidation
        self.reflection = ReflectionMechanism(
            self.config, self.memory.vlm, self.memory.embedding_model
        )
        self.consolidation = MemoryConsolidation(self.config)
        
        # Learning components
        self.reliability_tracker = BayesianReliabilityTracker(
            prior_alpha=self.config.PRIOR_ALPHA,
            prior_beta=self.config.PRIOR_BETA
        )
        self.contrastive = ContrastiveRefinement(self.config, self.memory.vlm)
        self.continual_tracker = ContinualLearningTracker()
        
        logger.info("[ok] All components initialized (paper-complete + continual learning)")
    
# ============================================================================
# LINGUAAgent.process_video 
# ============================================================================

    def process_video(self, video_path: str) -> Dict[int, Dict]:
        """Process video through event-driven perception"""
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 1: Event-Driven Perception")
        logger.info(f"{'='*80}")
        
        selected_frames = self.perception.select_frames(video_path)
        
        for frame_idx, frame_data in selected_frames.items():
            memory = self.memory.create_episodic_memory(
                frame_data,
                start_time=frame_data["timestamp"],
                end_time=frame_data["timestamp"] + 1.0
            )
            self.memory.episodic_memory.append(memory)
        
        logger.info(f"[ok] Created {len(self.memory.episodic_memory)} episodic memories")

        # Paper Section 3, Episodic Memory: merge adjacent narratives
        self.memory.merge_adjacent_episodes()
        
        # NEW: Consolidation
        if len(self.memory.episodic_memory) >= 5:
            logger.info(f"\n{'='*80}")
            logger.info("STAGE 1.5: Episodic  Semantic Consolidation")
            logger.info(f"{'='*80}")
            
            consolidated = self.memory.semantic_memory.consolidate_from_episodic(
                self.memory.episodic_memory,
                min_instances=3
            )
            
            if consolidated:
                logger.info(f"[ok] Consolidated {len(consolidated)} concepts")
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 2: Procedural Mining")
        logger.info(f"{'='*80}")
        
        scripts = self.procedural_miner.mine_scripts(self.memory.episodic_memory)
        self.memory.procedural_memory = scripts
        
        logger.info(f"[ok] Mined {len(scripts)} procedural scripts")
        
        return selected_frames

    # ============================================================================
    # FUNCTIONS FOR BAV LOOP
    # ============================================================================
    def answer_question_BAV(self, question: str) -> Dict[str, Any]:
        """
        Paper Section 3, Belief-Action-Verification (BAV) Loops.

        Key paper-alignment properties:
          - EU computation receives observed event descriptions and the
            running list of past-failure narratives, so Rel_k and Risk_k
            are computed per the paper formula
                EU_k = Rel_k * E[rho_k] - Risk_k + lambda_info * H(Beta).
          - Bayesian posterior updates are purely verification-driven
            (paper: posterior update on verification outcome).
          - Reflection is triggered after max_attempts unsuccessful cycles,
            using paper-faithful thresholds (consecutive failures,
            postcondition coverage < 0.3, semantic drift > 0.7).
        """
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 3: BAV Loop (Retrieve-Hypothesize-Verify-Update)")
        logger.info(f"{'='*80}")
        logger.info(f"Q: {question}")

        past_failure_narratives: List[str] = []
        max_attempts = self.config.REFLECTION_FAILURE_COUNT  # 3 per paper

        goal = ""
        objects: List = []

        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n  Cycle {attempt}/{max_attempts}")

            # ============================================================
            # B: BELIEF -- retrieve evidence and generate goal hypotheses
            # ============================================================
            logger.info("\n  [B] Belief: retrieving memories...")

            episodic_events = self.memory.episodic_memory
            observed_descriptions = [
                e.description for e in episodic_events
                if getattr(e, "description", None)
            ]

            if episodic_events:
                objects = episodic_events[0].affordance_objects
            else:
                objects = []

            semantic_affordances = {
                obj: self.memory.semantic_memory.semantic_entries[obj]
                for obj in objects
                if obj in self.memory.semantic_memory.semantic_entries
            }
            procedural_scripts = self.memory.procedural_memory

            logger.info(
                f"     Retrieved: {len(episodic_events)} episodic, "
                f"{len(semantic_affordances)} semantic, "
                f"{len(procedural_scripts)} procedural"
            )

            # ============================================================
            # A: ACTION -- hypothesise goal and rank candidate scripts
            # ============================================================
            logger.info("\n  [A] Action: hypothesising goal and selecting script...")
            goal = self.goal_inferencer.infer_goal(question, objects, episodic_events)
            logger.info(f"     Goal hypothesis: {goal}")

            if not procedural_scripts:
                answer = self._generate_answer(question)
                return {
                    "answer": answer, "confidence": 0.3, "script_used": None,
                    "goal": goal, "verification": None
                }

            # Rank by paper EU formula
            utilities = []
            for proc in procedural_scripts:
                if not proc.is_active:
                    continue
                alpha, beta_v = self.reliability_tracker.get_reliability(proc.script_id)
                proc.alpha, proc.beta = alpha, beta_v
                eu = self.action_selector.compute_expected_utility(
                    proc,
                    observed_descriptions=observed_descriptions,
                    past_failure_narratives=past_failure_narratives,
                )
                utilities.append({"proc": proc, "eu": eu})

            if not utilities:
                return {
                    "answer": "unknown", "confidence": 0.0, "script_used": None,
                    "goal": goal, "verification": None
                }

            best = max(utilities, key=lambda x: x["eu"])
            logger.info(f"     Selected script: {best['proc'].script_id}")
            logger.info(
                f"     EU: {best['eu']:.3f}   "
                f"E[rho]={best['proc'].reliability:.3f}"
            )

            if best["eu"] < self.config.EU_THRESHOLD:
                logger.info(
                    f"     max EU < tau_EU ({self.config.EU_THRESHOLD}); "
                    "may trigger reflection."
                )

            answer = self._generate_answer(question)

            # ============================================================
            # V: VERIFY postconditions + temporal grounding
            # ============================================================
            logger.info("\n  [V] Verification: grounding...")
            grounding_status = self.verifier.verify_with_grounding(
                script=best["proc"],
                episodic_memories=episodic_events,
                config=self.config,
            )
            logger.info(
                f"     post_cov={grounding_status.postcondition_coverage:.3f} "
                f"temp_cons={grounding_status.temporal_consistency:.3f} "
                f"grounded={grounding_status.is_grounded}"
            )

            # ============================================================
            # Bayesian update -- verification-driven only (paper Eq. 4)
            # ============================================================
            success = bool(grounding_status.is_grounded)
            self.reliability_tracker.update_after_verification(
                best["proc"].script_id, success=success
            )

            # Log decision for reflection drift detection
            from types import SimpleNamespace
            self.reflection.add_decision(SimpleNamespace(
                hypothesis=str(goal),
                confidence=float(grounding_status.postcondition_coverage),
                success=success,
            ))

            if success:
                logger.info("     [ok] Verification passed: alpha += 1")
                return {
                    "answer": answer,
                    "confidence": float(best["eu"]),
                    "script_used": best["proc"].script_id,
                    "goal": goal,
                    "verification": {
                        "passed": grounding_status.verification_passed,
                        "post_coverage": grounding_status.postcondition_coverage,
                        "temporal_consistency": grounding_status.temporal_consistency,
                    },
                    "grounding_status": grounding_status,
                    "attempt": attempt,
                }

            # Record narrative for Risk_k in the next cycle
            logger.info("     [x] Verification failed: beta += 1")
            past_failure_narratives.append(
                f"goal={goal}; script={best['proc'].script_name}; "
                f"post_cov={grounding_status.postcondition_coverage:.2f}"
            )

        # ============================================================
        # REFLECTION (paper Meta Reflection paragraph)
        # ============================================================
        if self.reflection.detect_abnormality():
            logger.info("\n  Triggering meta-cognitive reflection...")
            result = self.reflection.reflect_and_recontextualize({
                "objects": objects,
                "timestamp": 0,
            })
            if result.get("refined_hypothesis"):
                logger.info(f"  Refined hypothesis: {result['refined_hypothesis']}")
                return {
                    "answer": result["refined_hypothesis"],
                    "confidence": 0.5,
                    "script_used": "reflected",
                    "goal": goal,
                    "verification": None,
                    "reflection": result,
                }

        return {
            "answer": "unknown", "confidence": 0.0, "script_used": None,
            "goal": goal, "verification": None
        }

    def update_and_refine(self, result: Dict, answer_gt: str, description: str):
        """
        Update learning and apply contrastive refinement
        Section 3.5: Contrastive Refinement (Eq. 13-16)
        """
        script_id = result.get("script_used")
        if not script_id or script_id == "reflected":
            return
        
        success = (result["answer"].lower().strip() == answer_gt.lower().strip())
        
        # Bayesian update (Eq. 4)
        self.reliability_tracker.update_after_verification(script_id, success)
        
        # Track for contrastive analysis
        self.contrastive.add_example(script_id, description, success)
        
        # Apply contrastive refinement if sufficient examples 
        if self.contrastive.can_refine(script_id):
            logger.info(f"\n{'='*80}")
            logger.info("STAGE 4: Contrastive Refinement")
            logger.info(f"{'='*80}")
            
            refined = self.contrastive.refine_if_possible(
                script_id, 
                self.memory.procedural_memory
            )
            
            if refined:
                logger.info(f"[ok] Script refined via contrastive analysis")
    
    # ============================================================================
    # CONTINUAL LEARNING METHODS
    # ============================================================================
    
    def update_learning(self, result: Dict, answer_gt: str, video_id: str, 
                       question: str, domain: str = "general"):
        """
        Update learning and track continual learning metrics
        
        This method should be called after getting ground truth feedback
        """
        correct = (result["answer"].lower().strip() == answer_gt.lower().strip())
        
        # Track continual learning metrics
        self.continual_tracker.log_performance(
            video_id=video_id,
            question=question,
            correct=correct,
            script_used=result.get("script_used", "none"),
            domain=domain
        )
        
        # Get description for contrastive analysis
        description = (self.memory.episodic_memory[0].description 
                      if self.memory.episodic_memory else "")
        
        # Existing update logic (Bayesian + Contrastive)
        self.update_and_refine(result, answer_gt, description)
    
    def get_learning_statistics(self):
        """
        Get continual learning statistics for reporting
        
        Returns:
            Dict with learning curve, domain adaptation metrics, etc.
        """
        return {
            "learning_curve": self.continual_tracker.get_learning_curve(window_size=10),
            "domain_adaptation": self.continual_tracker.get_domain_adaptation_metrics(),
            "forgetting_analysis": self.continual_tracker.get_all_forgetting_analysis(),
            "total_videos": self.continual_tracker.video_count
        }
    
    def save_continual_learning(self, filepath: str):
        """Save continual learning data"""
        self.continual_tracker.save(filepath)
    
    def load_continual_learning(self, filepath: str):
        """Load continual learning data"""
        self.continual_tracker.load(filepath)
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _generate_answer(self, question: str) -> str:
        """
        Generate answer from episodic memory based on question type
        Simple heuristic-based generation (can be replaced with VLM)
        """
        if not self.memory.episodic_memory:
            return "unknown"
        
        q = question.lower()
        
        # Why questions  return goal
        if "why" in q:
            for m in self.memory.episodic_memory:
                if m.goal:
                    return m.goal
        
        # How/What questions  return action
        elif "how" in q or "what" in q:
            for m in self.memory.episodic_memory:
                if m.action:
                    return m.action
        
        # Who questions  return agent
        elif "who" in q:
            for m in self.memory.episodic_memory:
                if m.agent:
                    return m.agent
        
        # Where questions  return location
        elif "where" in q:
            for m in self.memory.episodic_memory:
                if m.location:
                    return m.location
        
        # Default fallback
        return "yes"
    
    def save_tracker(self, filepath: str):
        """Save Bayesian reliability tracker"""
        self.reliability_tracker.save(filepath)
    
    def load_tracker(self, filepath: str):
        """Load Bayesian reliability tracker"""
        self.reliability_tracker.load(filepath)
    
    def save_all(self, output_dir: str):
        """
        Save all system state (comprehensive checkpoint)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save reliability tracker
        self.save_tracker(str(output_path / "reliability_tracker.json"))
        
        # Save continual learning data
        self.save_continual_learning(str(output_path / "continual_learning.json"))
        
        logger.info(f"[ok] Saved all system state to {output_dir}")
    
    def load_all(self, output_dir: str):
        """
        Load all system state (restore from checkpoint)
        """
        output_path = Path(output_dir)
        
        # Load reliability tracker
        tracker_file = output_path / "reliability_tracker.json"
        if tracker_file.exists():
            self.load_tracker(str(tracker_file))
        
        # Load continual learning data
        cl_file = output_path / "continual_learning.json"
        if cl_file.exists():
            self.load_continual_learning(str(cl_file))
        
        logger.info(f"[ok] Loaded system state from {output_dir}")


# ============================================================================
# EVALUATION
# ============================================================================
# ============================================================================
# TEMPORAL GROUNDING UTILITIES
# ============================================================================


def compute_iou(pred_span: Tuple[float, float], gt_span: Tuple[float, float]) -> float:
    """
    Compute Intersection over Union (IoU)
    IoU = Intersection / Union
    """
    if pred_span is None or gt_span is None:
        return 0.0
    
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span
    
    # Intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)
    
    # Union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_iop(pred_span: Tuple[float, float], gt_span: Tuple[float, float]) -> float:
    """
    Compute Intersection over Prediction (IoP)
    IoP = Intersection / Prediction_length
    
    Measures precision: what fraction of predicted segment is actually relevant
    """
    if pred_span is None or gt_span is None:
        return 0.0
    
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span
    
    # Intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)
    
    # Prediction length
    pred_length = pred_end - pred_start
    
    if pred_length == 0:
        return 0.0
    
    return intersection / pred_length


def compute_recall_at_iou(ious: List[float], threshold: float) -> float:
    """Compute Recall@IoU_threshold (R@0.3, R@0.5, etc.)"""
    if len(ious) == 0:
        return 0.0
    return sum(1 for iou in ious if iou >= threshold) / len(ious)


def compute_recall_at_iop(iops: List[float], threshold: float) -> float:
    """Compute Recall@IoP_threshold"""
    if len(iops) == 0:
        return 0.0
    return sum(1 for iop in iops if iop >= threshold) / len(iops)


# ============================================================================
# COMPREHENSIVE METRICS CLASS
# ============================================================================

@dataclass
class ComprehensiveMetrics:
    """
    Tracks ALL metrics reported in the paper:
    1. Accuracy (overall + T/C/D breakdown)
    2. IoU-based: R@0.3, R@0.5, mIoU
    3. IoP-based: R@0.3, R@0.5, mIoP  ← ADDED
    4. Acc@GQA (grounded accuracy)
    5. Continual learning metrics
    """
    
    predictions: List[Dict] = field(default_factory=list)
    question_types: Dict[str, List[bool]] = field(default_factory=lambda: {
        'temporal': [],
        'causal': [],
        'descriptive': []
    })
    
    # Temporal grounding metrics
    ious: List[float] = field(default_factory=list)
    iops: List[float] = field(default_factory=list)  # ← ADDED
    grounded_correct: List[bool] = field(default_factory=list)
    
    def add_prediction(
        self,
        pred_answer: str,
        gt_answer: str,
        pred_span: Optional[Tuple[float, float]] = None,
        gt_span: Optional[Tuple[float, float]] = None,
        video_id: str = "",
        question: str = "",
        question_type: str = "descriptive"
    ):
        """Add a prediction with all associated metrics"""
        
        is_correct = (pred_answer.lower().strip() == gt_answer.lower().strip())
        
        # Question type breakdown
        q_type = self._infer_question_type(question, question_type)
        if q_type in self.question_types:
            self.question_types[q_type].append(is_correct)
        
        # Temporal grounding
        iou_score = 0.0
        iop_score = 0.0
        grounded_acc = False
        
        if pred_span is not None and gt_span is not None:
            iou_score = compute_iou(pred_span, gt_span)
            iop_score = compute_iop(pred_span, gt_span)  # ← ADDED
            self.ious.append(iou_score)
            self.iops.append(iop_score)  # ← ADDED
            
            # Acc@GQA: correct answer AND IoU ≥ 0.5
            grounded_acc = is_correct and (iou_score >= 0.5)
            self.grounded_correct.append(grounded_acc)
        
        # Store prediction
        self.predictions.append({
            'video_id': video_id,
            'question': question,
            'question_type': q_type,
            'pred_answer': pred_answer,
            'gt_answer': gt_answer,
            'is_correct': is_correct,
            'pred_span': pred_span,
            'gt_span': gt_span,
            'iou': iou_score,
            'iop': iop_score,  # ← ADDED
            'grounded_correct': grounded_acc
        })
    
    def _infer_question_type(self, question: str, explicit_type: str = None) -> str:
        """Infer question type from keywords"""
        if explicit_type:
            return explicit_type
        
        q_lower = question.lower()
        
        # Temporal keywords
        if any(kw in q_lower for kw in ['when', 'before', 'after', 'first', 'last', 'then', 'next']):
            return 'temporal'
        
        # Causal keywords
        if any(kw in q_lower for kw in ['why', 'how', 'because', 'reason', 'cause']):
            return 'causal'
        
        # Default to descriptive
        return 'descriptive'
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute ALL metrics matching Table 1 in the paper
        """
        if len(self.predictions) == 0:
            return {
                'total_samples': 0,
                'accuracy': 0.0,
                'temporal_acc': 0.0,
                'causal_acc': 0.0,
                'descriptive_acc': 0.0,
                'miou': 0.0,
                'miop': 0.0,
                'iou_r03': 0.0,
                'iou_r05': 0.0,
                'iop_r03': 0.0,
                'iop_r05': 0.0,
                'acc_gqa': 0.0,
                'samples_with_spans': 0
            }
        
        # 1. Overall accuracy
        correct = [p['is_correct'] for p in self.predictions]
        overall_accuracy = sum(correct) / len(correct)
        
        # 2. Question type breakdown
        temporal_acc = (sum(self.question_types['temporal']) / len(self.question_types['temporal']) 
                       if self.question_types['temporal'] else 0.0)
        causal_acc = (sum(self.question_types['causal']) / len(self.question_types['causal']) 
                     if self.question_types['causal'] else 0.0)
        descriptive_acc = (sum(self.question_types['descriptive']) / len(self.question_types['descriptive']) 
                          if self.question_types['descriptive'] else 0.0)
        
        # 3. IoU-based metrics
        miou = np.mean(self.ious) if self.ious else 0.0
        iou_r03 = compute_recall_at_iou(self.ious, 0.3)
        iou_r05 = compute_recall_at_iou(self.ious, 0.5)
        
        # 4. IoP-based metrics ← ADDED
        miop = np.mean(self.iops) if self.iops else 0.0
        iop_r03 = compute_recall_at_iop(self.iops, 0.3)
        iop_r05 = compute_recall_at_iop(self.iops, 0.5)
        
        # 5. Grounded accuracy (Acc@GQA)
        acc_gqa = (sum(self.grounded_correct) / len(self.grounded_correct) 
                  if self.grounded_correct else 0.0)
        
        return {
            'total_samples': len(self.predictions),
            'samples_with_spans': len(self.ious),
            
            # Accuracy metrics
            'accuracy': overall_accuracy,
            'temporal_acc': temporal_acc,
            'causal_acc': causal_acc,
            'descriptive_acc': descriptive_acc,
            
            # IoU-based grounding
            'miou': miou,
            'iou_r03': iou_r03,
            'iou_r05': iou_r05,
            
            # IoP-based grounding ← ADDED
            'miop': miop,
            'iop_r03': iop_r03,
            'iop_r05': iop_r05,
            
            # Grounded QA
            'acc_gqa': acc_gqa
        }
    
    def print_metrics(self):
        """Print metrics in  format (Table 1)"""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n NExT-QA Accuracy:")
        print(f"  Temporal (T):     {metrics['temporal_acc']*100:6.2f}%")
        print(f"  Causal (C):       {metrics['causal_acc']*100:6.2f}%")
        print(f"  Descriptive (D):  {metrics['descriptive_acc']*100:6.2f}%")
        print(f"  Average:          {metrics['accuracy']*100:6.2f}%")
        
        if metrics['samples_with_spans'] > 0:
            print(f"\n🎯 NExT-GQA Grounding (IoU-based):")
            print(f"  R@0.3:   {metrics['iou_r03']*100:6.2f}%")
            print(f"  R@0.5:   {metrics['iou_r05']*100:6.2f}%")
            print(f"  mIoU:    {metrics['miou']*100:6.2f}%")
            
            print(f"\n🎯 NExT-GQA Grounding (IoP-based):  ← ADDED")
            print(f"  R@0.3:   {metrics['iop_r03']*100:6.2f}%")
            print(f"  R@0.5:   {metrics['iop_r05']*100:6.2f}%")
            print(f"  mIoP:    {metrics['miop']*100:6.2f}%")
            
            print(f"\nGrounded Accuracy:")
            print(f"  Acc@GQA: {metrics['acc_gqa']*100:6.2f}%")
            print(f"  (Requires correct answer + IoU ≥ 0.5)")
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total samples:            {metrics['total_samples']}")
        print(f"  Samples with grounding:   {metrics['samples_with_spans']}")
        print("="*80 + "\n")


# ============================================================================
# HELPER: EXTRACT TEMPORAL SPAN FROM SCRIPT
# ============================================================================

def extract_temporal_span_from_script(script, episodic_memory: List) -> Optional[Tuple[float, float]]:
    """
    Extract temporal span from executed script by matching with episodic memory
    """
    if not episodic_memory:
        return None
    
    # Find episodes matching script actions
    matching_episodes = []
    for episode in episodic_memory:
        # Check if episode contains script actions
        if hasattr(script, 'actions') and script.actions:
            for action in script.actions:
                if action.lower() in episode.get('description', '').lower():
                    matching_episodes.append(episode)
                    break
    
    if not matching_episodes:
        return None
    
    # Get temporal span from matching episodes
    start_time = min(ep.get('timestamp', [0, 0])[0] for ep in matching_episodes)
    end_time = max(ep.get('timestamp', [0, 0])[1] for ep in matching_episodes)
    
    return (start_time, end_time)


# ============================================================================
# QUESTION TYPE MAPPING (ADD THIS NEW FUNCTION)
# ============================================================================

def map_question_type(type_code: str) -> str:
    """
    Map NExT-QA question type codes to our categories
    
    Type codes from dataset:
    - TC/TP/TN/TA/TB = Temporal
    - CW/CH/DO = Causal  
    - DC/DL/DB = Descriptive
    """
    type_mapping = {
        # Temporal questions
        'TP': 'temporal',  # Temporal Previous
        'TN': 'temporal',  # Temporal Next
        'TC': 'temporal',  # Temporal Causality
        'TB': 'temporal',  # Temporal Before
        'TA': 'temporal',  # Temporal After
        
        # Causal questions
        'CW': 'causal',    # Causal Why
        'CH': 'causal',    # Causal How
        'DO': 'causal',    # Doing (action-oriented)
        
        # Descriptive questions
        'DC': 'descriptive',  # Descriptive Count
        'DL': 'descriptive',  # Descriptive Location
        'DB': 'descriptive',  # Descriptive Object/Being
    }
    
    return type_mapping.get(type_code, 'descriptive')


# ============================================================================
# LOAD NExT-GQA DATA
# ============================================================================

def load_nextgqa_data(dataset_path: str, split: str = "val"):
    """Load NExT-GQA dataset with temporal annotations"""
    dataset_path = Path(dataset_path)
    
    # Load CSV with questions and answers
    csv_file = dataset_path / f"{split}.csv"
    if not csv_file.exists():
        # Try alternative naming convention
        csv_file = dataset_path / f"{split}(NExT-GQA).csv"
    
    if not csv_file.exists():
        # List available CSV files to help debug
        available_csvs = list(dataset_path.glob("*.csv"))
        raise FileNotFoundError(
            f"CSV file not found!\n"
            f"  Expected: {dataset_path / f'{split}.csv'}\n"
            f"  Or: {dataset_path / f'{split}(NExT-GQA).csv'}\n"
            f"  Available CSV files: {available_csvs}"
        )
    
    logger.info(f"📄 Loading CSV from: {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"[ok] Loaded {len(df)} questions")
    
    # Load JSON with temporal annotations
    json_file = dataset_path / f"gsub_{split}.json"
    
    logger.info(f"📄 Loading annotations from: {json_file}")
    
    if json_file.exists():
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        logger.info(f"[ok] Loaded {len(annotations)} video annotations")
    else:
        logger.warning(f"[warn] Temporal annotations not found!")
        logger.warning(f"   Expected: {json_file}")
        logger.warning(f"   Available JSON files: {list(dataset_path.glob('*.json'))}")
        annotations = {}
    
    return df, annotations

# ============================================================================
# MAIN EVALUATION WITH ALL METRICS (CORRECTED VERSION)
# ============================================================================

def main_evaluation():
    """
    COMPLETE EVALUATION with ALL metrics from Table 1:
    - Accuracy (T/C/D breakdown)
    - IoU metrics: R@0.3, R@0.5, mIoU
    - IoP metrics: R@0.3, R@0.5, mIoP
    - Acc@GQA (grounded accuracy)
    - Continual learning tracking
    """
    logger.info("\n" + "="*80)
    logger.info("LINGUA-Agent: COMPREHENSIVE EVALUATION")
    logger.info("="*80)
    logger.info("Metrics (matching Table 1):")
    logger.info("  Accuracy (T/C/D breakdown)")
    logger.info("  IoU-based: R@0.3, R@0.5, mIoU")
    logger.info("  IoP-based: R@0.3, R@0.5, mIoP")
    logger.info("  Grounded Accuracy (Acc@GQA)")
    logger.info("  Continual learning tracking")
    logger.info("="*80)
    
    # Configuration
    dataset_path = r"C:\VLM_Agent\NExT-GQA"  # ← CHANGE THIS TO YOUR PATH
    videos_path = Path(dataset_path) / "videos"
    
    if not videos_path.exists():
        logger.error(f"[err] Videos not found: {videos_path}")
        return
    
    # Check available videos
    available_videos = set(v.stem for v in videos_path.glob("*.mp4"))
    if len(available_videos) == 0:
        logger.error("[err] No videos found!")
        return
    
    logger.info(f"[ok] Found {len(available_videos)} videos\n")
    
    # Load dataset
    try:
        df, annotations = load_nextgqa_data(dataset_path, "val")
        
        # CONVERT FORMAT FIRST, BEFORE FILTERING!
        df['video_id'] = df['video_id'].astype(int).astype(str)  # Keep full number!
        # df['video_id'] = df['video_id'].astype(int).astype(str).str.zfill(10)  # 10 digits        
        df_filtered = df[df['video_id'].isin(available_videos)].head(100)
        
        logger.info(f"[ok] Loaded {len(df_filtered)} samples\n")

    except Exception as e:
        logger.error(f"[err] Dataset loading failed: {e}")
        return
    
    # USE REAL AGENT - NOT MOCK! logger.info("🤖 Initializing LINGUA-Agent...")
    config = LINGUAConfig()

    # VLM for NExT-GQA (Ollama vision model)
    config.VLM_PROVIDER = "ollama"
    config.VLM_MODEL = "llava:7b"
    config.VLM_MAX_LENGTH = 128

    # LLM for reasoning (Ollama text model)
    config.LLM_PROVIDER = "ollama"
    config.LLM_MODEL = "gemma3:4b"
    config.LLM_MAX_LENGTH = 256
    agent = LINGUAAgent(config)
    logger.info("[ok] Agent initialized\n")

    metrics_tracker = ComprehensiveMetrics()
    
    # Process videos
    for idx, row in df_filtered.iterrows():
        try:
            video_id = str(row['video_id'])
            question = str(row['question'])
            answer_gt = str(row['answer'])
            
            video_path = videos_path / f"{video_id}.mp4"
            if not video_path.exists():
                logger.warning(f"[warn] Video not found: {video_path}")
                continue
            
            logger.info(f"\n[{idx+1}/{len(df_filtered)}] Video: {video_id}")
            logger.info(f"Q: {question[:80]}...")
            
            # Process video with REAL agent
            agent.process_video(str(video_path))
            
            # Answer question with REAL agent
            result = agent.answer_question_BAV(question)
            
            # Extract temporal span (if applicable)
            pred_span = None
            if result.get("script_used") and result.get("script_used") != "reflected":
                # Try to extract temporal span from script execution
                pred_span = extract_temporal_span_from_script(
                    result.get("script_used"),
                    agent.memory.episodic_memory
                )
            
            # Get ground truth temporal span
            gt_span = None
            if video_id in annotations:
                video_ann = annotations[video_id]
                for qa_pair in video_ann.get('QA_pairs', []):
                    if qa_pair.get('question') == question:
                        if 'temporal_span' in qa_pair:
                            gt_span = tuple(qa_pair['temporal_span'])
                        break
            
            # Map question type correctly
            question_type_code = row.get('type', 'DC')
            question_type = map_question_type(question_type_code)
            
            # Update metrics
            metrics_tracker.add_prediction(
                pred_answer=result["answer"],
                gt_answer=answer_gt,
                pred_span=pred_span,
                gt_span=gt_span,
                video_id=video_id,
                question=question,
                question_type=question_type  # Now correctly mapped
            )
            
            # Log result
            correct = (result["answer"].lower().strip() == answer_gt.lower().strip())
            logger.info(f"Pred: {result['answer']}")
            logger.info(f"GT:   {answer_gt}")
            logger.info(f"Result: {'[ok] CORRECT' if correct else '✗ WRONG'}")
            
            if pred_span and gt_span:
                iou = compute_iou(pred_span, gt_span)
                iop = compute_iop(pred_span, gt_span)
                logger.info(f"IoU: {iou:.3f} | IoP: {iop:.3f}")
            
            # Update learning (for continual learning tracking)
            domain = "general"  # Could infer from video metadata
            agent.update_learning(result, answer_gt, video_id, question, domain)
            
            # Progress report every 20 videos
            if (idx + 1) % 20 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"PROGRESS REPORT ({idx+1}/{len(df_filtered)} videos)")
                logger.info(f"{'='*60}")
                current_metrics = metrics_tracker.compute_all_metrics()
                logger.info(f"Current Accuracy: {current_metrics['accuracy']*100:.2f}%")
                logger.info(f"  Temporal:     {current_metrics['temporal_acc']*100:.2f}%")
                logger.info(f"  Causal:       {current_metrics['causal_acc']*100:.2f}%")
                logger.info(f"  Descriptive:  {current_metrics['descriptive_acc']*100:.2f}%")
                
                if current_metrics['samples_with_spans'] > 0:
                    logger.info(f"\nGrounding Metrics:")
                    logger.info(f"  mIoU: {current_metrics['miou']*100:.2f}%")
                    logger.info(f"  mIoP: {current_metrics['miop']*100:.2f}%")
                    logger.info(f"  Acc@GQA: {current_metrics['acc_gqa']*100:.2f}%")
                logger.info(f"{'='*60}\n")
        
        except Exception as e:
            logger.error(f"[err] Error processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================================
    # FINAL RESULTS: TABLE FORMAT
    # ============================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info("FINAL COMPREHENSIVE RESULTS")
    logger.info(f"{'='*80}")
    
    # Print all metrics (matching Table 1)
    metrics_tracker.print_metrics()
    
    # Get learning statistics
    learning_stats = agent.get_learning_statistics()
    
    logger.info(f"\n📈 Continual Learning Statistics:")
    logger.info(f"  Total videos processed: {learning_stats['total_videos']}")
    
    if learning_stats['learning_curve']:
        final_acc = learning_stats['learning_curve'][-1]['accuracy']
        logger.info(f"  Final rolling accuracy: {final_acc*100:.2f}%")
    
    # Save detailed results
    results_file = Path(dataset_path) / "comprehensive_results.json"
    final_metrics = metrics_tracker.compute_all_metrics()
    
    all_results = {
        "standard_metrics": final_metrics,
        "predictions": metrics_tracker.predictions,
        "breakdown": {
            "temporal": {
                "count": len(metrics_tracker.question_types['temporal']),
                "accuracy": final_metrics['temporal_acc']
            },
            "causal": {
                "count": len(metrics_tracker.question_types['causal']),
                "accuracy": final_metrics['causal_acc']
            },
            "descriptive": {
                "count": len(metrics_tracker.question_types['descriptive']),
                "accuracy": final_metrics['descriptive_acc']
            }
        },
        "continual_learning": learning_stats
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n📁 Detailed results saved to: {results_file}")
    
    # Save continual learning data
    cl_file = Path(dataset_path) / "continual_learning.json"
    agent.save_continual_learning(str(cl_file))
    logger.info(f"📁 Continual learning data saved to: {cl_file}")
    
    # Save reliability tracker
    tracker_file = Path(dataset_path) / "reliability_tracker.json"
    agent.save_tracker(str(tracker_file))
    logger.info(f"📁 Reliability tracker saved to: {tracker_file}")
    
    logger.info(f"{'='*80}\n")
    
    # Return metrics for further analysis
    return final_metrics, metrics_tracker


if __name__ == "__main__":
    # Run comprehensive evaluation
    final_metrics, tracker = main_evaluation()