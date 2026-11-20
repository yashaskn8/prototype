"""One process-wide lock for constructing local transformer models.

Loading two checkpoints at the same time from different threads (for example
BGE-small in the startup preload while a request's bounded worker loads V5)
crashed the Windows process with a native access violation. Construction is
rare and short, so it is serialized; inference is not affected.
"""
import threading

MODEL_LOAD_LOCK = threading.RLock()
