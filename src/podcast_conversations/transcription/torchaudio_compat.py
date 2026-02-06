"""Compatibility shims for torchaudio API changes.

This module provides backwards compatibility for pyannote.audio and whisperx
which use deprecated torchaudio APIs that were removed in torchaudio 2.1+.

Import this module BEFORE importing any modules that use pyannote.audio.
"""

import logging

logger = logging.getLogger(__name__)

def apply_torchaudio_compat_shims():
    """Apply compatibility shims for torchaudio 2.1+ with older pyannote.audio."""
    try:
        import torchaudio
        
        # AudioMetaData was renamed/removed in torchaudio 2.1+
        if not hasattr(torchaudio, 'AudioMetaData'):
            from collections import namedtuple
            AudioMetaData = namedtuple('AudioMetaData', [
                'sample_rate', 'num_frames', 'num_channels', 
                'bits_per_sample', 'encoding'
            ])
            torchaudio.AudioMetaData = AudioMetaData
            logger.debug("added torchaudio.AudioMetaData")
        
        # list_audio_backends() was removed in torchaudio 2.1+
        if not hasattr(torchaudio, 'list_audio_backends'):
            def list_audio_backends():
                # Return soundfile as default backend
                return ['soundfile']
            torchaudio.list_audio_backends = list_audio_backends
            logger.debug("added torchaudio.list_audio_backends()")
        
        # get_audio_backend() was removed in torchaudio 2.1+
        if not hasattr(torchaudio, 'get_audio_backend'):
            def get_audio_backend():
                # Default backend in modern torchaudio
                return 'soundfile'
            torchaudio.get_audio_backend = get_audio_backend
            logger.debug("added torchaudio.get_audio_backend()")
        
        logger.info("torchaudio compatibility shims applied successfully")
        
    except Exception as e:
        logger.warning(f"failed to apply torchaudio compatibility shims: {e}")


def apply_torch_weights_only_fix():
    """Fix PyTorch 2.6+ weights_only default for pyannote models.
    
    PyTorch 2.6 changed the default of weights_only from False to True for security.
    However, pyannote models use omegaconf configs that require weights_only=False.
    Since these are trusted models from HuggingFace, we globally disable the restriction.
    """
    try:
        import torch
        
        # Monkey-patch torch.load to use weights_only=False by default
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            # Set weights_only=False if not explicitly specified
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        logger.info("patched torch.load to use weights_only=False for pyannote models")
        
    except Exception as e:
        logger.warning(f"failed to apply torch weights_only fix: {e}")


# Apply shims immediately on import
apply_torchaudio_compat_shims()
apply_torch_weights_only_fix()
