import argparse
import logging
import os
import gc
import re
from typing import List, Tuple

import cv2
import numpy as np
import scipy.fftpack
import scipy.ndimage
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("EVM_Magnifier")


class EulerianMagnifier:
    """
    Eulerian Video Magnifier using Laplacian spatial pyramids and Ideal Temporal Bandpass Filtering.
    This class vectorizes operations over the entire 4D video tensor (T, H, W, C) to ensure optimal
    performance without slow frame-by-frame loops for core mathematical convolutions.
    """

    def __init__(self, alpha: float = 10.0, low_omega: float = 5.0, high_omega: float = 25.0, fps: int = 200, levels: int = 4):
        """
        Initializes the EVM processing class.

        Args:
            alpha (float): Magnification factor to apply to the motion.
            low_omega (float): Lower frequency bound in Hz for the temporal ideal bandpass filter.
            high_omega (float): Upper frequency bound in Hz for the temporal ideal bandpass filter.
            fps (int): Frame rate of the dataset (e.g., 200 for CASME II).
            levels (int): Number of depth levels for the Laplacian spatial pyramid decomposition.
        """
        self.alpha = alpha
        self.low_omega = low_omega
        self.high_omega = high_omega
        self.fps = fps
        self.levels = levels

    def _blur(self, tensor: np.ndarray) -> np.ndarray:
        """
        Applies a 1D Gaussian blur kernel [1, 4, 6, 4, 1] / 16 across both spatial axes (Height, Width).
        Using mode 'reflect' safely protects original structural boundary artifacts.

        Args:
            tensor (np.ndarray): The 4D NumPy float tensor to blur.

        Returns:
            np.ndarray: The spatially blurred 4D video tensor.
        """
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        blurred = scipy.ndimage.convolve1d(tensor, kernel, axis=1, mode='reflect')
        blurred = scipy.ndimage.convolve1d(blurred, kernel, axis=2, mode='reflect')
        return blurred

    def _upsample(self, downsampled: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Upsamples a tensor geographically by zero-stuffing, then blurring.
        The kernel multiplier of 4 (energy conservation ratio) is achieved via /8.0 kernel division.

        Args:
            downsampled (np.ndarray): The compressed scale component tensor spatial pyramid.
            target_shape (Tuple[int, ...]): The strict resolution expected to reconstruct back upwards.

        Returns:
            np.ndarray: Upscaled tensor perfectly matching target resolution constraint sizes.
        """
        up = np.zeros(target_shape, dtype=np.float32)
        up[:, ::2, ::2, :] = downsampled
        
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 8.0
        blurred = scipy.ndimage.convolve1d(up, kernel, axis=1, mode='reflect')
        blurred = scipy.ndimage.convolve1d(blurred, kernel, axis=2, mode='reflect')
        return blurred

    def _build_laplacian_pyramid(self, video_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Constructs a Laplacian mathematical pyramid efficiently for the 4D video tensor.

        Args:
            video_tensor (np.ndarray): Zero padded normalized base inputs prior to hierarchy loops.

        Returns:
            List[np.ndarray]: A sequence comprising spatially isolated scaled representations concluding
            in the fully blurred baseline. Length equals self.levels + 1.
        """
        pyramid = []
        current = video_tensor
        for _ in range(self.levels):
            blurred = self._blur(current)
            downsampled = blurred[:, ::2, ::2, :]
            
            upsampled = self._upsample(downsampled, current.shape)
            laplacian = current - upsampled
            pyramid.append(laplacian)
            
            current = downsampled
            
        pyramid.append(current) # Base Gaussian image map
        return pyramid

    def magnify(self, video_tensor: np.ndarray) -> np.ndarray:
        """
        The principal invocation engine handling padding protection, frequency FFT masking,
        and temporal reconstruction summation all at optimal matrix bandwidth loads.

        Args:
            video_tensor (np.ndarray): The raw 4D target sequence shape (T, H, W, C).

        Returns:
            np.ndarray: The finalized micro-expression augmented tensor bounded natively to byte constraints.
        """
        tensor_float = video_tensor.astype(np.float32)
        
        T, H, W, C = tensor_float.shape
        # Dynamically protect limits ensuring 100% boundary division parity explicitly using border reflection
        divisor = 2 ** self.levels
        pad_h = (divisor - (H % divisor)) % divisor
        pad_w = (divisor - (W % divisor)) % divisor
        
        if pad_h > 0 or pad_w > 0:
            logger.info(f"Dynamically padding spatial dimensions (H:{H}->{H+pad_h}, W:{W}->{W+pad_w}) for boundary parity.")
            tensor_float = np.pad(tensor_float, ((0,0), (0, pad_h), (0, pad_w), (0,0)), mode='reflect')
            
        T_pad, H_pad, W_pad, C_pad = tensor_float.shape
            
        logger.info(f"Building spatial pyramids (shape: {tensor_float.shape}, levels: {self.levels})")
        pyramid = self._build_laplacian_pyramid(tensor_float)
        
        logger.info(f"Applying Temporal Bandpass FFT [{self.low_omega}Hz - {self.high_omega}Hz] at {self.fps} FPS")
        freqs = scipy.fftpack.fftfreq(T_pad, d=1.0/self.fps)
        mask = (np.abs(freqs) >= self.low_omega) & (np.abs(freqs) <= self.high_omega)
        mask_bc = mask.reshape(-1, 1, 1, 1)

        # To optimize huge memory RAM pools computationally we accumulate collapsing inside the loop.
        current = pyramid[-1] 

        for i in range(self.levels - 1, -1, -1):
            level_tensor = pyramid[i]
            
            # Identify sub-band motion physically located per respective isolated scale.
            fft_tensor = scipy.fftpack.fft(level_tensor, axis=0)
            fft_tensor = np.where(mask_bc, fft_tensor, 0)
            magnified_motion = scipy.fftpack.ifft(fft_tensor, axis=0).real
            magnified_motion *= self.alpha
            
            # Reconstruction addition sequence
            current = self._upsample(current, level_tensor.shape)
            current = current + level_tensor + magnified_motion
            
            # Free memory
            del level_tensor, fft_tensor, magnified_motion
            gc.collect()
            
        # Free unused pyramid levels mapping
        del pyramid
        
        # Discard transient padding mapping back safely guaranteeing unharmed vector features!
        result = current[:, :H, :W, :]
        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)


class ImageSequenceOrchestrator:
    """
    Handles robust I/O operational wrappers abstractly decoding standard multimedia codecs correctly
    into raw tensors processing directories seamlessly batching workflows appropriately.
    """
    def __init__(self, input_dir: str, output_dir: str, magnifier: EulerianMagnifier):
        """
        Args:
            input_dir (str): Base raw samples folder location path.
            output_dir (str): Destination path folder routing.
            magnifier (EulerianMagnifier): Configured computation hardware engine sequence hook.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.magnifier = magnifier
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _read_image_sequence(self, leaf_dir: str, image_files: List[str]) -> np.ndarray:
        frames = []
        for img_file in image_files:
            img_path = os.path.join(leaf_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                raise IOError(f"Failed opening physical image trace: {img_path}")
            # Ensure memory contiguous arrays from cv2
            frames.append(np.ascontiguousarray(frame))
        return np.array(frames, dtype=np.float32)

    def _write_video(self, tensor: np.ndarray, path: str):
        T, H, W, C = tensor.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, float(self.magnifier.fps), (W, H))
        
        if not out.isOpened():
            logger.error(f"Failed to open video writer for {path}")
            return
            
        try:
            for i in range(T):
                out.write(tensor[i])
        finally:
            out.release()
        
    def process_directory(self):
        """
        Recursively loops through all subdirectories locating target image files, duplicating their
        exact relative path hierarchies inside the designated output directory wrapper gracefully.
        """
        valid_exts = {'.jpg', '.jpeg', '.png'}
        
        # 1. Discover all leaf directories
        leaf_dirs = {}
        for root_dir, _, filenames in os.walk(self.input_dir):
            images_in_dir = [f for f in filenames if os.path.splitext(f)[1].lower() in valid_exts]
            if images_in_dir:
                leaf_dirs[root_dir] = images_in_dir
                    
        if not leaf_dirs:
            logger.warning(f"No valid image sequence samples discovered recursively within: '{self.input_dir}'")
            return
            
        logger.info(f"Discovered exactly {len(leaf_dirs)} leaf directories intended to amplify safely.")
        
        # 2. Process with progress tracking
        for leaf_dir, image_files in tqdm(leaf_dirs.items(), desc="Magnifying EVM Sequences"):
            rel_dir = os.path.relpath(leaf_dir, self.input_dir)
            out_path = os.path.join(self.output_dir, f"{rel_dir}.avi")
            
            # Ensure the relative subfolder directory structure physically exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # 3. Check cached output state
            if os.path.exists(out_path):
                logger.info(f"Skipping cached artifact mapping: {out_path}")
                continue
                
            # 4. Fault-tolerant sequence execution
            try:
                # Alphanumeric sorting to ensure img9.jpg comes before img10.jpg
                sorted_files = sorted(image_files, key=lambda s: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)])
                
                vid_tensor = self._read_image_sequence(leaf_dir, sorted_files)
                logger.info(f"Matrix footprint loaded structurally for '{rel_dir}': {vid_tensor.shape}")
                
                mag_tensor = self.magnifier.magnify(vid_tensor)
                self._write_video(mag_tensor, out_path)
                
                # 5. Aggressive Memory Management
                del vid_tensor
                del mag_tensor
                gc.collect()
                
            except Exception as e:
                logger.error(f"Computation failure terminating vector graph payload strictly at {rel_dir}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Eulerian Video Magnification Processor.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path pointing unmagnified video subjects target records.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dedicated storage bucket folder generated sequences routing.")
    parser.add_argument('--alpha', type=float, default=10.0, help="EVM explicit magnification strength configuration ratio. (Default: 10.0)")
    parser.add_argument('--low_omega', type=float, default=5.0, help="Temporal filtering lower cutoff target hz physically isolating thresholds. (Default: 5.0hz)")
    parser.add_argument('--high_omega', type=float, default=25.0, help="Temporal filtering upper cutoff target hz physically masking limits. (Default: 25.0hz)")
    parser.add_argument('--fps', type=int, default=200, help="Assumed original recording hardware frequency timeline rate. (Default: 200 via CASME II baseline)")
    parser.add_argument('--levels', type=int, default=4, help="Mathematical laplacian hierarchy recursion breakdown depth sizes. (Default: 4)")

    args = parser.parse_args()
    
    logger.info("Initializing Eulerian Matrix Modulator Algorithm core settings payload...")
    magnifier = EulerianMagnifier(
        alpha=args.alpha,
        low_omega=args.low_omega,
        high_omega=args.high_omega,
        fps=args.fps,
        levels=args.levels
    )
    
    orchestrator = ImageSequenceOrchestrator(args.input_dir, args.output_dir, magnifier)
    orchestrator.process_directory()


if __name__ == '__main__':
    main()
