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
    
    This fix uses two approaches:
    1. Add omegaconf classes to torch safe globals (preferred, works with weights_only=True)
    2. Patch torch.load as fallback (forces weights_only=False)
    
    Since these are trusted models from HuggingFace, both approaches are safe.
    """
    try:
        import torch
        
        # approach 1: add omegaconf classes to safe globals (PyTorch 2.6+).
        # this is the recommended approach as it allows weights_only=True to work.
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf
            from omegaconf.listconfig import ListConfig as ListConfigType
            from omegaconf.dictconfig import DictConfig as DictConfigType
            
            safe_classes = [ListConfig, DictConfig, ListConfigType, DictConfigType]
            
            # try to add OmegaConf itself if it's used.
            try:
                safe_classes.append(OmegaConf)
            except Exception:
                pass
            
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(safe_classes)
                logger.info("added omegaconf classes to torch safe globals")
            else:
                # older pytorch, fallback to patching.
                raise AttributeError("add_safe_globals not available")
                
        except ImportError:
            logger.debug("omegaconf not installed, skipping safe globals registration")
        except Exception as e:
            logger.debug(f"safe globals approach failed: {e}, falling back to patch")
            
            # approach 2: patch torch.load as fallback.
            original_torch_load = torch.load
            
            def patched_torch_load(f, *args, **kwargs):
                # force weights_only=False if it would otherwise be True.
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                elif kwargs.get('weights_only') is True:
                    kwargs['weights_only'] = False
                    logger.debug("overriding weights_only=True to False for pyannote compatibility")
                return original_torch_load(f, *args, **kwargs)
            
            torch.load = patched_torch_load
            logger.info("patched torch.load to force weights_only=False")
        
    except Exception as e:
        logger.warning(f"failed to apply torch weights_only fix: {e}")


# Apply shims immediately on import
apply_torchaudio_compat_shims()
apply_torch_weights_only_fix()
