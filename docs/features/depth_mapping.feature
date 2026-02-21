Feature: Depth Map Generation
  As an artist exploring dimensional composition,
  I want to generate depth maps from photographs or video frames of people,
  So that I can use depth as a shading guide or to create layered compositions.

  Background:
    Given the depth mapping pipeline is installed and configured
    And an image or video frame containing a person is available as input

  # --- Basic Depth Map Generation ---

  Scenario: Generate a depth map from a single photograph
    Given an input image "portrait.jpg" containing a single person
    When I run depth map generation with default settings
    Then the output is a 16-bit grayscale PNG
    And pixels closer to the camera are brighter
    And pixels farther from the camera are darker
    And the output resolution matches the input resolution

  Scenario: Depth map captures relative ordering of body parts
    Given an input image of a person with one arm extended toward the camera
    When I run depth map generation
    Then the extended hand appears brighter than the torso
    And the torso appears brighter than the background
    And the depth gradient across the arm is smooth and continuous

  Scenario: Depth map separates person from background
    Given an input image with a person in the foreground and a wall behind them
    When I run depth map generation
    Then there is a clear depth discontinuity between the person and the wall
    And the person's depth values are uniformly closer than the background

  # --- Algorithm Selection ---

  Scenario: Generate depth map using MiDaS
    Given an input image "portrait.jpg"
    When I run depth map generation using the "midas" algorithm
    Then the output is a valid depth map
    And the output metadata records the algorithm as "midas"

  Scenario: Generate depth map using Depth Anything
    Given an input image "portrait.jpg"
    When I run depth map generation using the "depth_anything" algorithm
    Then the output is a valid depth map
    And the output metadata records the algorithm as "depth_anything"

  Scenario: Compare depth algorithms side by side
    Given an input image "portrait.jpg"
    When I run depth map generation using algorithms "midas", "depth_anything", and "zoedepth"
    Then the output includes one depth map per algorithm
    And each output is labeled with the algorithm name
    And all outputs use the same depth value range for comparison

  # --- Output Formats ---

  Scenario: Export depth map as 16-bit grayscale PNG
    Given an input image "portrait.jpg"
    When I run depth map generation with output format "png16"
    Then the output is a 16-bit grayscale PNG file
    And the depth values span the full 0-65535 range

  Scenario: Export depth map as EXR for high dynamic range
    Given an input image "portrait.jpg"
    When I run depth map generation with output format "exr"
    Then the output is a 32-bit float EXR file
    And the depth values are stored as linear float values

  Scenario: Export depth map with color visualization
    Given an input image "portrait.jpg"
    When I run depth map generation with output format "color"
    Then the output is a color-mapped PNG using a perceptually uniform colormap
    And near regions are warm colors and far regions are cool colors

  # --- Depth Map Quality ---

  Scenario: Depth map edges align with person contours
    Given an input image "portrait.jpg"
    When I run depth map generation
    Then depth discontinuities align with the person's silhouette edges
    And there is no significant depth bleeding beyond the person's boundary

  Scenario: Depth map is smooth within continuous surfaces
    Given an input image of a person's torso
    When I run depth map generation
    Then depth values vary smoothly across the torso surface
    And there are no abrupt depth jumps within a single continuous body surface

  Scenario: Depth map handles challenging poses
    Given an input image of a person with crossed arms
    When I run depth map generation
    Then the overlapping arm is correctly estimated as closer than the body behind it
    And the depth ordering of overlapping limbs is plausible

  # --- Integration with Edge Detection ---

  Scenario: Generate depth map masked to person region only
    Given an input image with a person and background
    When I run depth map generation with person isolation enabled
    Then the depth map contains depth values only for the person region
    And the background region has zero or null depth values

  Scenario: Depth map can be combined with edge map as layers
    Given an input image "portrait.jpg"
    When I run depth map generation
    And I run edge detection at detail level 5
    Then both outputs have identical dimensions
    And both outputs can be loaded as separate layers in a PSD-compatible editor

  # --- Error Handling ---

  Scenario: Graceful handling of very small input images
    Given an input image at 64x64 resolution
    When I run depth map generation
    Then the pipeline produces a depth map at 64x64 resolution
    And a warning indicates that small input size may reduce depth accuracy

  Scenario: Graceful handling when no person is detected
    Given an input image of an empty room with no people
    When I run depth map generation with person isolation enabled
    Then the pipeline returns an empty depth map
    And a warning message indicates no person was detected

  Scenario: Missing model weights produce clear error
    Given the depth model weights file is not present on disk
    When I attempt to run depth map generation
    Then the pipeline returns an error indicating which model weights are missing
    And the error message includes instructions for downloading the weights
