"""Tests for output file writing and layer export."""

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from my_amv.output import (
    OutputWriter,
    save_layered_png,
    save_layers,
    save_topology_obj,
    save_topology_svg,
)
from my_amv.types import DepthArray, EdgeArray, LayerKey, RGBArray


class TestOutputWriter:
    """Test OutputWriter class."""

    def test_save_layers_creates_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        layers = {"test": np.zeros((10, 10, 3), dtype=np.uint8)}
        output_dir = tmp_path / "subdir" / "layers"

        writer = OutputWriter()
        writer.save_layers(layers, output_dir)

        assert output_dir.exists()
        assert (output_dir / "test.png").exists()

    def test_save_layers_rgb_png(self, tmp_path):
        """Should save RGB arrays as 24-bit PNG files."""
        # Create a test RGB array
        rgb: RGBArray = np.full((10, 10, 3), 128, dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]  # Red pixel
        rgb[0, 1] = [0, 255, 0]  # Green pixel
        rgb[0, 2] = [0, 0, 255]  # Blue pixel

        layers = {"rgb_layer": rgb}
        output_dir = tmp_path / "layers"

        writer = OutputWriter()
        writer.save_layers(layers, output_dir)

        output_file = output_dir / "rgb_layer.png"
        assert output_file.exists()

        # Verify the output is lossless by round-tripping
        img = Image.open(output_file)
        arr = np.array(img)
        np.testing.assert_array_equal(arr, rgb)

    def test_save_layers_multiple(self, tmp_path):
        """Should save multiple layers to separate files."""
        layers = {
            "layer1": np.zeros((10, 10, 3), dtype=np.uint8),
            "layer2": np.ones((10, 10, 3), dtype=np.uint8) * 255,
            "layer3": np.full((10, 10, 3), 128, dtype=np.uint8),
        }
        output_dir = tmp_path / "layers"

        writer = OutputWriter()
        writer.save_layers(layers, output_dir)

        assert (output_dir / "layer1.png").exists()
        assert (output_dir / "layer2.png").exists()
        assert (output_dir / "layer3.png").exists()

    def test_save_layers_depth_as_16bit(self, tmp_path):
        """Should save depth arrays as 16-bit grayscale PNGs."""
        # Create a depth map
        depth: DepthArray = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)

        # Need to wrap in dict as RGBArray type (but actually depth)
        # The function checks dtype to determine format
        layers = {"depth": depth}  # type: ignore
        output_dir = tmp_path / "layers"

        writer = OutputWriter()
        writer.save_layers(layers, output_dir)

        output_file = output_dir / "depth.png"
        assert output_file.exists()

        # Verify it's 16-bit
        img = Image.open(output_file)
        assert img.mode == "I;16"

    def test_save_layered_png(self, tmp_path):
        """Should save multiple layers as frames in a single PNG."""
        layers = {
            "layer1": np.zeros((10, 10, 3), dtype=np.uint8),
            "layer2": np.full((10, 10, 3), 255, dtype=np.uint8),
        }
        output_path = tmp_path / "layered.png"

        writer = OutputWriter()
        writer.save_layered_png(layers, output_path)

        assert output_path.exists()

        # Verify it's a valid PNG
        img = Image.open(output_path)
        assert img.format == "PNG"
        assert img.size == (10, 10)

    def test_save_layered_png_empty_raises(self, tmp_path):
        """Should raise ValueError for empty layers dict."""
        writer = OutputWriter()
        output_path = tmp_path / "layered.png"

        with pytest.raises(ValueError, match="Cannot save empty layers"):
            writer.save_layered_png({}, output_path)

    def test_save_layered_png_creates_directory(self, tmp_path):
        """Should create output directory if needed."""
        layers = {"test": np.zeros((10, 10, 3), dtype=np.uint8)}
        output_path = tmp_path / "subdir" / "layered.png"

        writer = OutputWriter()
        writer.save_layered_png(layers, output_path)

        assert output_path.exists()

    def test_save_topology_svg(self, tmp_path):
        """Should save edge topology as SVG file."""
        # Create a simple edge image (a square)
        edges: EdgeArray = np.zeros((100, 100), dtype=np.uint8)
        edges[20:80, 20] = 255  # Left edge
        edges[20:80, 80] = 255  # Right edge
        edges[20, 20:80] = 255  # Top edge
        edges[80, 20:80] = 255  # Bottom edge

        output_path = tmp_path / "topology.svg"

        writer = OutputWriter()
        writer.save_topology_svg(edges, output_path)

        assert output_path.exists()

        # Verify it's valid XML/SVG
        content = output_path.read_text()
        assert "<svg" in content
        assert "</svg>" in content
        assert "path" in content

    def test_save_topology_svg_creates_directory(self, tmp_path):
        """Should create output directory if needed."""
        edges: EdgeArray = np.zeros((10, 10), dtype=np.uint8)
        output_path = tmp_path / "subdir" / "topology.svg"

        writer = OutputWriter()
        writer.save_topology_svg(edges, output_path)

        assert output_path.exists()

    def test_save_topology_svg_custom_stroke_width(self, tmp_path):
        """Should respect custom stroke_width parameter."""
        edges: EdgeArray = np.ones((10, 10), dtype=np.uint8) * 255
        output_path = tmp_path / "topology.svg"

        writer = OutputWriter()
        writer.save_topology_svg(edges, output_path, stroke_width=2.5)

        content = output_path.read_text()
        assert 'stroke-width="2.5"' in content

    def test_save_topology_obj(self, tmp_path):
        """Should save 2.5D topology as OBJ file."""
        # Create a simple edge and depth map
        edges: EdgeArray = np.zeros((50, 50), dtype=np.uint8)
        edges[20:30, 20:30] = 255  # Square of edges

        depth: DepthArray = np.zeros((50, 50), dtype=np.float32)
        depth[20:30, 20:30] = 0.5  # Depth for the square

        output_path = tmp_path / "topology.obj"

        writer = OutputWriter()
        writer.save_topology_obj(depth, edges, output_path)

        assert output_path.exists()

        # Verify OBJ format
        content = output_path.read_text()
        assert "v " in content  # Vertices
        assert "# Wavefront OBJ" in content

    def test_save_topology_obj_no_edges(self, tmp_path):
        """Should create empty OBJ file when no edges found."""
        edges: EdgeArray = np.zeros((10, 10), dtype=np.uint8)
        depth: DepthArray = np.zeros((10, 10), dtype=np.float32)
        output_path = tmp_path / "topology.obj"

        writer = OutputWriter()
        writer.save_topology_obj(depth, edges, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "No edges found" in content

    def test_save_topology_obj_creates_directory(self, tmp_path):
        """Should create output directory if needed."""
        edges: EdgeArray = np.ones((10, 10), dtype=np.uint8)
        depth: DepthArray = np.zeros((10, 10), dtype=np.float32)
        output_path = tmp_path / "subdir" / "topology.obj"

        writer = OutputWriter()
        writer.save_topology_obj(depth, edges, output_path)

        assert output_path.exists()


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_save_layers_function(self, tmp_path):
        """Module-level save_layers should work correctly."""
        layers = {"test": np.zeros((10, 10, 3), dtype=np.uint8)}
        output_dir = tmp_path / "layers"

        save_layers(layers, output_dir)

        assert (output_dir / "test.png").exists()

    def test_save_layered_png_function(self, tmp_path):
        """Module-level save_layered_png should work correctly."""
        layers = {"test": np.zeros((10, 10, 3), dtype=np.uint8)}
        output_path = tmp_path / "layered.png"

        save_layered_png(layers, output_path)

        assert output_path.exists()

    def test_save_topology_svg_function(self, tmp_path):
        """Module-level save_topology_svg should work correctly."""
        edges: EdgeArray = np.zeros((10, 10), dtype=np.uint8)
        output_path = tmp_path / "topology.svg"

        save_topology_svg(edges, output_path)

        assert output_path.exists()

    def test_save_topology_obj_function(self, tmp_path):
        """Module-level save_topology_obj should work correctly."""
        edges: EdgeArray = np.zeros((10, 10), dtype=np.uint8)
        depth: DepthArray = np.zeros((10, 10), dtype=np.float32)
        output_path = tmp_path / "topology.obj"

        save_topology_obj(depth, edges, output_path)

        assert output_path.exists()
