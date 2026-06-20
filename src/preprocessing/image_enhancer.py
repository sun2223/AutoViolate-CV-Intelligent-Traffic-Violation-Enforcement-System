"""
Task 1 - Image Preprocessing & Enhancement Pipeline
====================================================
Handles: low-light, rain, shadows, motion blur, dynamic normalisation.
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, List


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _brightness(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


class LowLightEnhancer:
    """Adaptive gamma + CLAHE on LAB L-channel (Zero-DCE inspired)."""
    def __init__(self, clip_limit=3.0, tile_grid=(8, 8), dark_thresh=80.0):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        self.dark_thresh = dark_thresh

    def _gamma(self, img: np.ndarray) -> np.ndarray:
        br = _brightness(img)
        g = max(1.0, min(3.0, 2.0 * (1.0 - br / self.dark_thresh)))
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self.clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        img = _to_bgr(img)
        if _brightness(img) < self.dark_thresh:
            img = self._gamma(img)
        return self._clahe(img)


class RainRemover:
    """Guided-filter base/detail decomposition to suppress rain streaks."""
    def __init__(self, radius=15, eps=0.01):
        self.radius, self.eps = radius, eps

    def _guided(self, guide, src):
        try:
            import cv2.ximgproc as xi
            return xi.guidedFilter(guide.astype(np.float32), src.astype(np.float32), self.radius, self.eps)
        except Exception:
            return cv2.bilateralFilter(src, 9, 75, 75)

    def remove(self, img: np.ndarray) -> np.ndarray:
        img = _to_bgr(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        chs = []
        for c in cv2.split(img):
            cf = c.astype(np.float32) / 255.0
            base = self._guided(gray, cf)
            chs.append(np.clip((base + 0.3 * (cf - base)) * 255, 0, 255).astype(np.uint8))
        return cv2.merge(chs)


class ShadowRemover:
    """LAB luminance lift for shadow regions."""
    def remove(self, img: np.ndarray) -> np.ndarray:
        img = _to_bgr(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        l_mean = cv2.dilate(l, kernel)
        mask = (l < l_mean * 0.6).astype(np.float32)
        l = np.clip(l + 30.0 * mask, 0, 255)
        lab = cv2.merge([l, np.clip(a, 0, 255), np.clip(b, 0, 255)]).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class MotionDeblurrer:
    """Wiener deconvolution for linear motion blur."""
    def __init__(self, kernel_size=21):
        self.kernel_size = kernel_size

    def _kernel(self, size):
        k = np.zeros((size, size), dtype=np.float32)
        k[size // 2, :] = 1.0 / size
        return k

    def _wiener(self, ch, kernel, snr=50.0):
        C = np.fft.fft2(ch.astype(np.float64))
        K = np.fft.fft2(kernel, s=ch.shape)
        G = (np.conj(K) / (np.abs(K) ** 2 + 1.0 / snr)) * C
        return np.clip(np.abs(np.fft.ifft2(G)), 0, 255).astype(np.uint8)

    def deblur(self, img: np.ndarray) -> np.ndarray:
        img = _to_bgr(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_len = min(self.kernel_size, max(5, int(cv2.Sobel(gray, cv2.CV_64F, 1, 0).var() / (cv2.Sobel(gray, cv2.CV_64F, 0, 1).var() + 1e-6) * 10)))
        k = self._kernel(blur_len)
        return cv2.merge([self._wiener(c, k) for c in cv2.split(img)])


class TrafficImagePreprocessor:
    """
    Master preprocessor – runs adaptive pipeline based on scene conditions.
    Usage: preprocessor = TrafficImagePreprocessor.from_config(cfg)
           tensor = preprocessor(bgr_image)  # returns float32 HWC
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, enhance_ll=True, remove_rain=False, remove_shadow=True,
                 deblur=True, target_size=(640, 640), blur_thresh=100.0):
        self.ll      = LowLightEnhancer() if enhance_ll else None
        self.rain    = RainRemover() if remove_rain else None
        self.shadow  = ShadowRemover() if remove_shadow else None
        self.deblur  = MotionDeblurrer() if deblur else None
        self.size    = target_size
        self.blur_th = blur_thresh

    @classmethod
    def from_config(cls, cfg):
        pp = cfg.get("preprocessing", {})
        return cls(
            enhance_ll=pp.get("low_light_enhance", True),
            remove_rain=pp.get("derain", False),
            remove_shadow=pp.get("shadow_removal", True),
            deblur=pp.get("deblur", True),
            target_size=tuple(pp.get("target_size", [640, 640])),
        )

    def _is_blurry(self, img):
        return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < self.blur_th

    def _has_rain(self, img):
        gy = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1)
        return float(np.var(gy)) > 3000.0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img is not None
        if self.ll:     img = self.ll.enhance(img)
        if self.shadow: img = self.shadow.remove(img)
        if self.rain and self._has_rain(img): img = self.rain.remove(img)
        if self.deblur and self._is_blurry(img): img = self.deblur.deblur(img)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (img - self.MEAN) / self.STD

    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        return np.stack([self(i) for i in images], axis=0)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = cv2.imread(path) if path else np.random.randint(0, 60, (720, 1280, 3), dtype=np.uint8)
    p = TrafficImagePreprocessor()
    out = p(img)
    print(f"Input {img.shape} → Output {out.shape} [{out.min():.3f}, {out.max():.3f}]")
