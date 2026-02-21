Feature: Edge Detection and Silhouette Extraction
  As a visual artist preparing reference layers,
  I want to detect edges and extract silhouettes of people from photographs or video frames,
  So that I can use clean structural outlines as a foundation for drawing and painting.

  Background:
    Given the edge detection pipeline is installed and configured
    And an image or video frame containing a person is available as input

  # --- Silhouette Extraction ---

  Scenario: Extract a clean person silhouette from a photograph
    Given an input image "portrait.jpg" containing a single person
    When I run silhouette extraction with default settings
    Then the output is a PNG with a binary alpha mask
    And the person region is opaque
    And the background region is transparent
    And the silhouette boundary is smooth with no jagged edges

  Scenario: Silhouette extraction isolates person from cluttered background
    Given an input image with a person against a complex background
    When I run silhouette extraction
    Then the output mask contains only the person
    And no background objects appear in the silhouette
    And the person's outline is continuous with no gaps

  Scenario: Silhouette handles partial occlusion
    Given an input image where the person is partially behind an object
    When I run silhouette extraction
    Then the visible portions of the person are included in the silhouette
    And the occluding object is excluded from the mask

  # --- Edge Detection with Detail Levels ---

  Scenario Outline: Generate edge map at configurable detail levels
    Given an input image "portrait.jpg" containing a single person
    When I run edge detection with detail level <level>
    Then the output is a PNG with black lines on a transparent background
    And the output contains <expected_content>
    And the output does not contain <excluded_content>

    Examples:
      | level | expected_content                          | excluded_content                     |
      | 1     | outer body silhouette only                | interior contours or texture         |
      | 3     | silhouette with head and limb separation  | clothing folds or facial features    |
      | 5     | major structural lines and limb contours  | skin texture or fabric weave         |
      | 7     | clothing folds and facial feature outlines | skin pores or fine hair strands      |
      | 10    | fine detail including fingers and hair     | background texture noise             |

  Scenario: Edge map excludes background edges at all detail levels
    Given an input image with a person against a textured background
    When I run edge detection at detail level 5
    Then the edge map contains only edges belonging to the person
    And no background texture or object edges appear in the output

  # --- Classical vs AI Algorithm Selection ---

  Scenario: Run edge detection with a classical algorithm
    Given an input image "portrait.jpg"
    When I run edge detection using the "canny" algorithm
    And I set the low threshold to 50 and high threshold to 150
    Then the output contains detected edges using Canny edge detection
    And the output is a PNG file at the same resolution as the input

  Scenario: Run edge detection with an AI-based algorithm
    Given an input image "portrait.jpg"
    When I run edge detection using the "hed" algorithm
    Then the output contains detected edges using HED deep learning model
    And the output is a PNG file at the same resolution as the input

  Scenario: Compare multiple algorithms side by side
    Given an input image "portrait.jpg"
    When I run edge detection using algorithms "canny", "hed", and "pidinet"
    Then the output includes one edge map per algorithm
    And each output is labeled with the algorithm name
    And all outputs use the same input resolution

  # --- Output Format and Quality ---

  Scenario: Output resolution matches input resolution
    Given an input image at 3840x2160 resolution
    When I run edge detection with default settings
    Then the output image is 3840x2160 pixels
    And the output is saved in lossless PNG format

  Scenario: Output lines are clean and anti-aliased
    Given an input image "portrait.jpg"
    When I run edge detection at detail level 5
    Then the edge lines are anti-aliased
    And no single-pixel noise dots appear in the output
    And edge lines have consistent visual weight

  # --- Person Isolation ---

  Scenario: Edge detection focuses on person region only
    Given an input image with a person and a bounding box around the person
    When I run edge detection with the region of interest set to the bounding box
    Then only edges within the bounding box region are detected
    And the rest of the output is empty

  Scenario: Automatic person detection when no region specified
    Given an input image containing a single person with no region of interest specified
    When I run edge detection with person isolation enabled
    Then the pipeline automatically detects the person
    And only edges belonging to the detected person are included in the output

  # --- Error Handling ---

  Scenario: Graceful handling when no person is detected
    Given an input image containing only a landscape with no people
    When I run edge detection with person isolation enabled
    Then the pipeline returns an empty edge map
    And a warning message indicates no person was detected in the image

  Scenario: Unsupported image format produces clear error
    Given an input file "photo.bmp" in an unsupported format
    When I attempt to run edge detection
    Then the pipeline returns an error message listing supported formats
    And no output file is created

  Scenario: Invalid detail level produces clear error
    Given an input image "portrait.jpg"
    When I run edge detection with detail level 15
    Then the pipeline returns an error indicating valid range is 1 to 10
    And no output file is created
