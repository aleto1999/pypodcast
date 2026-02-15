"""Compatibility shims for torchaudio API changes.

This module provides backwards compatibility for pyannote.audio and whisperx
which use deprecated torchaudio APIs that were removed in torchaudio 2.1+.

Import this module BEFORE importing any modules that use pyannote.audio.
"""

import logging
import os

logger = logging.getLogger(__name__)


def setup_cuda_library_path():
    """Setup CUDA library paths from nvidia python packages.
    
    PyTorch expects CUDA libraries to be in LD_LIBRARY_PATH or bundled.
    When using nvidia-*-cu12 packages from PyPI, we need to add their
    library directories to LD_LIBRARY_PATH before torch is imported.
    
    Also preloads cuDNN 8.x libraries for ctranslate2 compatibility
    (ctranslate2 dynamically loads libcudnn_ops_infer.so.8).
    """
    try:
        import ctypes
        import site
        from pathlib import Path
        
        nvidia_packages = [
            "nvidia.cuda_runtime",
            "nvidia.cublas",
            "nvidia.cudnn",
        ]
        
        lib_dirs = []
        
        for pkg_name in nvidia_packages:
            try:
                parts = pkg_name.split(".")
                mod = __import__(pkg_name)
                for part in parts[1:]:
                    mod = getattr(mod, part)
                
                pkg_path = Path(mod.__file__).parent
                
                lib_candidates = [
                    pkg_path / "lib",
                    pkg_path,
                ]
                
                for candidate in lib_candidates:
                    if candidate.exists() and list(candidate.glob("*.so*")):
                        lib_dirs.append(str(candidate))
                        break
                        
            except (ImportError, AttributeError):
                continue
        
        # search site-packages for cuDNN 8.x .so files from nvidia-cudnn-cu11.
        # nvidia-cudnn-cu11 and nvidia-cudnn-cu12 both use the nvidia.cudnn
        # namespace, so we scan for libcudnn_ops_infer.so.8 in all nvidia dirs.
        site_dirs = site.getsitepackages() + [site.getusersitepackages()]
        for site_dir in site_dirs:
            site_path = Path(site_dir)
            if not site_path.exists():
                continue
            for cudnn_lib in site_path.rglob("libcudnn_ops_infer.so.8*"):
                lib_dir = str(cudnn_lib.parent)
                if lib_dir not in lib_dirs:
                    lib_dirs.append(lib_dir)
                    # preload cuDNN 8 .so files so ctranslate2 can find them.
                    for lib in sorted(cudnn_lib.parent.glob("libcudnn*.so.8*")):
                        try:
                            ctypes.cdll.LoadLibrary(str(lib))
                            logger.debug(f"preloaded {lib.name}")
                        except OSError:
                            pass
                break
        
        if lib_dirs:
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            new_paths = ":".join(lib_dirs)
            
            if current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{current_ld_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = new_paths
            
            logger.info(f"added {len(lib_dirs)} nvidia library paths to LD_LIBRARY_PATH")
            return True
        else:
            logger.debug("no nvidia cuda libraries found in python packages")
            return False
            
    except Exception as e:
        logger.debug(f"failed to setup cuda library path: {e}")
        return False

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


# Apply fixes immediately on import (before torch is imported)
setup_cuda_library_path()  # must be first - sets up environment
apply_torchaudio_compat_shims()
apply_torch_weights_only_fix()
