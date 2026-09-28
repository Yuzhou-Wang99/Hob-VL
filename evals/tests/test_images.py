"""Offline image integrity, orientation, size-limit, and path tests."""

import hashlib
import io
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evals.data import MAX_IMAGE_BYTES, prepare_image, safe_image_path


class ImagePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hob_vl-evals-images-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def save(self, image, name="scene.png", **options):
        path = self.root / name
        image.save(path, **options)
        return path

    def assert_hashes_and_sizes(self, path, asset):
        original = path.read_bytes()
        self.assertEqual(asset.info["original_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(asset.info["prepared_sha256"], hashlib.sha256(asset.data).hexdigest())
        self.assertEqual(asset.info["original_bytes"], len(original))
        self.assertEqual(asset.info["prepared_bytes"], len(asset.data))
        with Image.open(io.BytesIO(asset.data)) as transmitted:
            transmitted.load()
            self.assertEqual(asset.info["prepared_dimensions"], list(transmitted.size))

    def test_small_png_preserves_original_bytes_and_dimensions(self):
        path = self.save(Image.new("RGB", (37, 19), "purple"))
        original = path.read_bytes()
        asset = prepare_image(path)
        self.assertEqual(asset.data, original)
        self.assertEqual(asset.mime_type, "image/png")
        self.assertFalse(asset.info["transformed"])
        self.assertEqual(asset.info["original_dimensions"], [37, 19])
        self.assertEqual(asset.info["prepared_dimensions"], [37, 19])
        self.assert_hashes_and_sizes(path, asset)

    def test_oversized_rectangle_is_fitted_without_cropping(self):
        image = Image.new("RGB", (400, 200))
        image.paste("red", (0, 0, 40, 40))
        image.paste("green", (360, 0, 400, 40))
        image.paste("blue", (0, 160, 40, 200))
        image.paste("yellow", (360, 160, 400, 200))
        path = self.save(image)
        asset = prepare_image(path, max_image_edge=100)
        self.assertTrue(asset.info["transformed"])
        self.assertEqual(asset.info["original_dimensions"], [400, 200])
        self.assertEqual(asset.info["prepared_dimensions"], [100, 50])
        with Image.open(io.BytesIO(asset.data)) as transmitted:
            # All four original corners survive; a center crop would lose them.
            self.assertEqual(transmitted.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(transmitted.getpixel((99, 0)), (0, 128, 0))
            self.assertEqual(transmitted.getpixel((0, 49)), (0, 0, 255))
            self.assertEqual(transmitted.getpixel((99, 49)), (255, 255, 0))
        self.assert_hashes_and_sizes(path, asset)

    def test_exif_orientation_normalizes_pixels_even_without_a_size_change(self):
        for orientation, size, top_left, bottom_right in [
            (6, [20, 40], (255, 0, 0), (0, 0, 255)),
            (3, [40, 20], (0, 0, 255), (255, 0, 0)),
        ]:
            with self.subTest(orientation=orientation):
                image = Image.new("RGB", (40, 20), "red")
                image.paste("blue", (20, 0, 40, 20))
                exif = Image.Exif()
                exif[274] = orientation
                path = self.save(image, name=f"orientation-{orientation}.png", exif=exif)
                asset = prepare_image(path)
                self.assertTrue(asset.info["transformed"])
                self.assertEqual(asset.info["original_dimensions"], [40, 20])
                self.assertEqual(asset.info["prepared_dimensions"], size)
                with Image.open(io.BytesIO(asset.data)) as transmitted:
                    self.assertEqual(transmitted.getexif().get(274, 1), 1)
                    self.assertEqual(transmitted.getpixel((0, 0)), top_left)
                    self.assertEqual(transmitted.getpixel((size[0] - 1, size[1] - 1)), bottom_right)
                self.assert_hashes_and_sizes(path, asset)

    def test_high_entropy_png_is_deterministically_reduced_below_real_byte_cap(self):
        # Dimensions already fit the default edge: only the encoded-byte cap
        # can force this image to shrink. Fixed-seed pixels avoid fixture files.
        pixels = random.Random(314159).randbytes(1200 * 1200 * 3)
        path = self.save(Image.frombytes("RGB", (1200, 1200), pixels), name="noise.png")
        self.assertEqual(MAX_IMAGE_BYTES, 4 * 1024 * 1024)
        self.assertGreater(path.stat().st_size, MAX_IMAGE_BYTES)
        asset = prepare_image(path)
        repeated = prepare_image(path)
        self.assertLessEqual(len(asset.data), MAX_IMAGE_BYTES)
        self.assertTrue(asset.info["transformed"])
        width, height = asset.info["prepared_dimensions"]
        self.assertLess(width, 1200)
        self.assertEqual(width, height)
        self.assertEqual(asset.data, repeated.data)
        self.assertEqual(asset.info, repeated.info)
        self.assert_hashes_and_sizes(path, asset)

    def test_corrupt_and_truncated_images_are_rejected(self):
        valid = self.save(Image.new("RGB", (10, 10), "red")).read_bytes()
        for index, content in enumerate([b"not an image", valid[:32]]):
            with self.subTest(index=index):
                path = self.root / f"corrupt-{index}.png"
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "Cannot decode image"):
                    prepare_image(path)

    def test_multiframe_png_is_rejected(self):
        first = Image.new("RGB", (10, 10), "red")
        second = Image.new("RGB", (10, 10), "blue")
        path = self.save(first, name="animated.png", save_all=True,
                         append_images=[second], duration=100, loop=0)
        with Image.open(path) as image:
            self.assertEqual(image.n_frames, 2)
        with self.assertRaisesRegex(ValueError, "Animated/multi-frame"):
            prepare_image(path)

    def test_valid_but_unsupported_image_type_is_rejected(self):
        path = self.save(Image.new("RGB", (10, 10), "red"), name="scene.bmp")
        with self.assertRaisesRegex(ValueError, "Unsupported image type: BMP"):
            prepare_image(path)


class ImagePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hob_vl-evals-paths-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_windows_separators_and_spaces_resolve_to_actual_file(self):
        relative = "images/photos/example image (2).png"
        expected = self.root / relative
        expected.parent.mkdir(parents=True)
        Image.new("RGB", (2, 2), "blue").save(expected)
        for spelling in [relative, relative.replace("/", "\\")]:
            with self.subTest(spelling=spelling):
                resolved = safe_image_path(self.root, spelling)
                self.assertEqual(resolved, expected)
                self.assertTrue(resolved.is_file())
                self.assertEqual(prepare_image(resolved).data, expected.read_bytes())

    def test_absolute_drive_relative_unc_and_escaping_paths_are_rejected(self):
        for relative in [
            "../outside.png", "..\\outside.png", "images/../../outside.png",
            "C:/images/scene.png", "C:\\images\\scene.png", "C:scene.png",
            "\\\\server\\share\\scene.png", "//server/share/scene.png",
            str(self.root / "absolute.png"),
        ]:
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    safe_image_path(self.root, relative)


if __name__ == "__main__":
    unittest.main()
