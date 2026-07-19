"""
Configuration parameters for window-based experiments.
Modify these values to tune the baseline and proposed method behaviors.
"""

class Config:
    # === General Simulation Parameters ===
    WINDOW_SIZE_SECONDS = 45 * 60  # 45 minutes by default

    # === LLM Backend Parameters ===
    DEFAULT_BRAIN_MODEL = "qwen3:8b"
    DEFAULT_VLM_MODEL = "qwen3-vl:8B"
    DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:8b"
    
    # General limit for full-window multimodal summary to avoid context explosion
    MAX_IMAGES_PER_MULTIMODAL_CALL = 40
    LLM_MAX_RETRIES = 1
    LLM_MIN_SUMMARY_CHARS = 24
    LLM_MIN_SUMMARY_WORDS = 6
    VLM_HTTP_TIMEOUT_S = 600
    # Use a longer timeout for contact-sheet requests because local large images
    # and montage decoding can take significantly longer than normal multi-image calls.
    VLM_CONTACT_SHEET_HTTP_TIMEOUT_S = 1800
    # Branch-specific retry controls for multimodal generation.
    # Value means "extra retries" so total attempts = 1 + value.
    VLM_MULTI_IMAGE_MAX_RETRIES = 0
    VLM_CONTACT_SHEET_MAX_RETRIES = 1
    VLM_SINGLE_IMAGE_MAX_RETRIES = 1
    # Strict multi-image mode: when >=2 images exist, each VLM request must include >=2 images.
    VLM_STRICT_MULTI_IMAGE = True
    # If strict multi-image direct calls all fail, send a contact-sheet composed from all snapshots.
    VLM_ENABLE_CONTACT_SHEET_FALLBACK = True
    # If image count is above this threshold, skip direct multi-image and use
    # contact-sheet first for better stability on large local screenshot batches.
    VLM_CONTACT_SHEET_FORCE_MIN_IMAGES = 11

    # === Proposed Method (HSR Window) Parameters ===
    # Layer 1 Filtering
    INVALID_WINDOW_TITLES = ['unknown', 'untitled', 'new tab', '_blank']
    MIN_VALID_DURATION_SEC = 2.0
    
    # Layer 3 Summarization
    PROPOSED_MAX_IMAGES_PER_TASK = 3
    PROPOSED_TASK_MEMORY_WINDOW = 3
    # Number of recent memory items injected into each VLM prompt.
    PROPOSED_TASK_MEMORY_PROMPT_ITEMS = 1

    # === Online Session Method Parameters ===
    # Experimental session-mode setting: use effectively unbounded limits so
    # the method can see the full session context when sessions are short.
    ONLINE_BRAIN_HISTORY_LENGTH = 4
    ONLINE_MAX_VISIBLE_TASKS = 999999
    ONLINE_TASK_MEMORY_WINDOW = 999999
    ONLINE_TASK_MEMORY_PROMPT_ITEMS = 1
    ONLINE_GLOBAL_MEMORY_PROMPT_ITEMS = 4
    ONLINE_FORCE_SAMPLE_ON_NEW_TASK = True
    # Keep periodic fallback sparse so adaptive sampling remains meaningful.
    ONLINE_PERIODIC_SAMPLE_EVERY = 100
    ONLINE_SUMMARY_MAX_MEMORY_ITEMS = 999999
    ONLINE_RECENT_TASK_GAP = 6
    ONLINE_TITLE_OVERLAP_THRESHOLD = 0.5

    # === Baseline Experiment Parameters ===
    # Uniform Sampling Baseline
    UNIFORM_SAMPLE_RATE = 5  # Sample 1 out of every 5 events

    # Change-Point Segmentation Baseline
    CHANGEPOINT_TITLE_SIMILARITY_THRESHOLD = 0.25
    CHANGEPOINT_DURATION_RELATIVE_JUMP = 2.5
    CHANGEPOINT_KPM_ABS_JUMP = 20.0
    CHANGEPOINT_CPM_ABS_JUMP = 80.0
    
    # Global Timeline Segments Baseline
    TIMELINE_SEGMENT_LENGTH = 20  # Number of events per segment
