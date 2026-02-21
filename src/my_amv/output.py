"""Output file writing and layer export.

This module provides functions for saving processed layers and topology data
to various file formats including PNG, SVG, and OBJ.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from typing_extensions import override

from my_amv.types import DepthArray, EdgeArray, LayerKey, RGBArray


class OutputWriter:
    """Writer for exporting layers and topology to various file formats."""

    def save_layers(
        self, layers: dict[LayerKey, RGBArray], output_dir: Path
    ) -> None:
        """Save each layer as a lossless PNG file.

        Depth arrays are saved as 16-bit grayscale PNGs.
        RGB arrays are saved as 24-bit RGB PNGs.

        Args:
            layers: Dictionary mapping layer names to RGB arrays
            output_dir: Directory to save the layer files

        Raises:
            OSError: If output directory cannot be created or written to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for layer_name, arr in layers.items():
            # Convert layer name to safe filename
            filename = f"{layer_name}.png"
            filepath = output_dir / filename

            # Determine if this is a depth map (float32) or RGB (uint8)
            if arr.dtype == np.float32:
                # Depth map - save as 16-bit grayscale
                # Normalize to 0-65535 range
                depth_normalized = (arr * 65535).astype(np.uint16)
                img = Image.fromarray(depth_normalized, mode="I;16")
            else:
                # RGB array - save as 24-bit RGB
                img = Image.fromarray(arr, mode="RGB")

            img.save(filepath, compression=0)  # compression=0 for lossless

    def save_layered_png(
        self, layers: dict[LayerKey, RGBArray], output_path: Path
    ) -> None:
        """Stack layers as separate frames in a multi-image PNG file.

        This creates a single PNG file with multiple frames (one per layer).
        Note: Not all image viewers support multi-frame PNGs.

        Args:
            layers: Dictionary mapping layer names to RGB arrays
            output_path: Path to output PNG file

        Raises:
            OSError: If output file cannot be written
            ValueError: If layers dict is empty
        """
        if not layers:
            raise ValueError("Cannot save empty layers dict")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the first layer to determine image dimensions
        first_layer = next(iter(layers.values()))
        first_img = Image.fromarray(first_layer, mode="RGB")

        # Save all layers as frames in a single file
        first_img.save(
            output_path,
            format="PNG",
            save_all=True,
            append_images=[
                Image.fromarray(arr, mode="RGB") for arr in list(layers.values())[1:]
            ],
            compression=0,  # Lossless
        )

    def save_topology_svg(
        self,
        edge_image: EdgeArray,
        output_path: Path,
        stroke_width: float = 1.0,
    ) -> None:
        """Save edge topology as an SVG file with vector paths.

        Uses cv2.findContours to detect contours and approxPolyDP to
        simplify them. The result is a clean SVG with no fill.

        Args:
            edge_image: Edge/contour array (grayscale)
            output_path: Path to output SVG file
            stroke_width: Width of stroke lines in SVG units

        Raises:
            OSError: If output file cannot be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import svgwrite
        except ImportError:
            raise ImportError(
                "svgwrite is required for SVG output. "
                "Install it with: pip install svgwrite"
            )

        # Find contours
        contours, _ = cv2.findContours(
            edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get image dimensions
        height, width = edge_image.shape

        # Create SVG document
        dwg = svgwrite.Drawing(
            str(output_path),
            size=(width, height),
            profile="tiny",
        )

        # Add each contour as a path
        for contour in contours:
            # Simplify the contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 2:
                continue

            # Build path data string
            path_data = self._contour_to_path_data(approx)

            # Add path to SVG
            dwg.add(
                dwg.path(
                    d=path_data,
                    stroke="black",
                    stroke_width=stroke_width,
                    fill="none",
                )
            )

        dwg.save()

    def _contour_to_path_data(self, contour: np.ndarray) -> str:
        """Convert OpenCV contour to SVG path data string.

        Args:
            contour: OpenCV contour array

        Returns:
            SVG path data string (e.g., "M x y L x y L x y Z")
        """
        if len(contour) == 0:
            return ""

        # Start with first point
        points = [f"M {contour[0][0][0]} {contour[0][0][1]}"]

        # Add lines to remaining points
        for i in range(1, len(contour)):
            x, y = contour[i][0]
            points.append(f"L {x} {y}")

        # Close path
        points.append("Z")

        return " ".join(points)

    def save_topology_obj(
        self,
        depth: DepthArray,
        edges: EdgeArray,
        output_path: Path,
    ) -> None:
        """Save topology as a 2.5D Wavefront OBJ mesh.

        Creates a mesh where edge points are lifted by their depth value.
        The result is a 2.5D mesh representing the topology.

        Args:
            depth: Depth map with values 0.0-1.0
            edges: Edge/contour array (grayscale)
            output_path: Path to output OBJ file

        Raises:
            OSError: If output file cannot be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find edge points
        edge_points = np.argwhere(edges > 0)

        if len(edge_points) == 0:
            # No edges found, create empty file
            output_path.write_text("# No edges found\n")
            return

        # Extract coordinates and depth values
        # Flip y-axis because image coordinates are (row, col) = (y, x)
        # but OBJ coordinates are (x, y, z)
        y_coords = edge_points[:, 0]
        x_coords = edge_points[:, 1]

        # Get depth values at edge points
        z_coords = depth[y_coords, x_coords]

        # Scale depth for OBJ (typical range 0-1 in depth, scale to reasonable Z)
        z_scale = 100.0
        z_coords = z_coords * z_scale

        # Build OBJ file content
        lines = [
            "# Wavefront OBJ file generated by my_amv",
            f"# {len(edge_points)} vertices",
            "",
        ]

        # Write vertices
        for x, y, z in zip(x_coords, y_coords, z_coords):
            # Invert y because OBJ has y-up, images have y-down
            lines.append(f"v {x} {-y} {z}")

        # Write edges as lines
        # For simplicity, we'll create point-like lines (pairs of nearby points)
        # A more sophisticated approach would trace actual contours
        lines.append("")
        lines.append(f"# {len(edge_points)} line segments")

        # Create lines between adjacent edge points (simplified connectivity)
        # In a real implementation, this would use contour tracing
        for i in range(min(len(edge_points) - 1, 10000)):  # Limit output size
            # Connect to next point if close enough
            if i + 1 < len(edge_points):
                dist = np.linalg.norm(edge_points[i] - edge_points[i + 1])
                if dist < 5:  # Threshold for connectivity
                    lines.append(f"l {i + 1} {i + 2}")  # OBJ is 1-indexed

        output_path.write_text("\n".join(lines))


# Convenience functions for backward compatibility

def save_layers(layers: dict[LayerKey, RGBArray], output_dir: Path) -> None:
    """Save each layer as a lossless PNG file.

    Convenience function that creates an OutputWriter and calls save_layers.

    Args:
        layers: Dictionary mapping layer names to RGB arrays
        output_dir: Directory to save the layer files
    """
    writer = OutputWriter()
    writer.save_layers(layers, output_dir)


def save_layered_png(layers: dict[LayerKey, RGBArray], output_path: Path) -> None:
    """Stack layers as separate frames in a multi-image PNG file.

    Convenience function that creates an OutputWriter and calls save_layered_png.

    Args:
        layers: Dictionary mapping layer names to RGB arrays
        output_path: Path to output PNG file
    """
    writer = OutputWriter()
    writer.save_layered_png(layers, output_path)


def save_topology_svg(
    edge_image: EdgeArray,
    output_path: Path,
    stroke_width: float = 1.0,
) -> None:
    """Save edge topology as an SVG file.

    Convenience function that creates an OutputWriter and calls save_topology_svg.

    Args:
        edge_image: Edge/contour array (grayscale)
        output_path: Path to output SVG file
        stroke_width: Width of stroke lines in SVG units
    """
    writer = OutputWriter()
    writer.save_topology_svg(edge_image, output_path, stroke_width)


def save_topology_obj(
    depth: DepthArray,
    edges: EdgeArray,
    output_path: Path,
) -> None:
    """Save topology as a 2.5D Wavefront OBJ mesh.

    Convenience function that creates an OutputWriter and calls save_topology_obj.

    Args:
        depth: Depth map with values 0.0-1.0
        edges: Edge/contour array (grayscale)
        output_path: Path to output OBJ file
    """
    writer = OutputWriter()
    writer.save_topology_obj(depth, edges, output_path)
